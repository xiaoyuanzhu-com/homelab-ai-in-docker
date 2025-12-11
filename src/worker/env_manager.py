"""EnvironmentManager - On-demand worker environment installation.

Handles:
- Checking if a worker environment is installed
- Installing environments on first use via `uv sync --frozen`
- Tracking installation status and progress
- Environment deletion to free disk space
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class EnvStatus(str, Enum):
    """Status of a worker environment."""

    NOT_INSTALLED = "not_installed"  # Template exists, .venv does not
    INSTALLING = "installing"  # Currently running uv sync
    READY = "ready"  # Installed and ready to use
    OUTDATED = "outdated"  # Installed but template has changed
    FAILED = "failed"  # Installation failed
    NOT_FOUND = "not_found"  # Template doesn't exist


@dataclass
class EnvInfo:
    """Information about a worker environment."""

    env_id: str
    status: EnvStatus
    template_path: Optional[Path] = None
    venv_path: Optional[Path] = None
    size_mb: Optional[float] = None
    install_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    python_version: Optional[str] = None


@dataclass
class InstallProgress:
    """Progress tracking for environment installation."""

    env_id: str
    started_at: float
    completed_at: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    log_lines: list[str] = field(default_factory=list)


class EnvironmentManager:
    """
    Manages on-demand worker environment installation.

    Environments are installed from templates in envs/ directory.
    Each template contains pyproject.toml and uv.lock files.
    Installation creates .venv in the template directory (or HAID_ENVS_DIR).
    """

    def __init__(self, envs_dir: Optional[Path] = None, data_envs_dir: Optional[Path] = None):
        """
        Initialize environment manager.

        Args:
            envs_dir: Directory containing environment templates (default: project/envs/)
            data_envs_dir: Optional directory for installed envs (for Docker volumes)
                          If set, .venv is created here instead of in template dir
        """
        if envs_dir is None:
            # Default to project_root/envs/
            project_root = Path(__file__).resolve().parent.parent.parent
            envs_dir = project_root / "envs"

        self.envs_dir = envs_dir
        self.data_envs_dir = data_envs_dir  # Optional: /haid/data/envs for Docker

        # Track ongoing installations
        self._installing: Dict[str, InstallProgress] = {}
        self._install_locks: Dict[str, asyncio.Lock] = {}

    def _get_template_dir(self, env_id: str) -> Path:
        """Get the template directory for an environment."""
        return self.envs_dir / env_id

    def _get_venv_dir(self, env_id: str) -> Path:
        """
        Get the .venv directory for an environment.

        If data_envs_dir is set (Docker mode), .venv goes there.
        Otherwise, .venv is inside the template directory.
        """
        if self.data_envs_dir:
            return self.data_envs_dir / env_id / ".venv"
        return self._get_template_dir(env_id) / ".venv"

    def _get_install_lock(self, env_id: str) -> asyncio.Lock:
        """Get or create installation lock for an environment."""
        if env_id not in self._install_locks:
            self._install_locks[env_id] = asyncio.Lock()
        return self._install_locks[env_id]

    def get_env_status(self, env_id: str) -> EnvInfo:
        """
        Get status of a worker environment.

        Args:
            env_id: Environment identifier (e.g., "transformers", "whisper")

        Returns:
            EnvInfo with current status
        """
        template_dir = self._get_template_dir(env_id)
        venv_dir = self._get_venv_dir(env_id)

        # Check if template exists
        if not template_dir.exists():
            return EnvInfo(env_id=env_id, status=EnvStatus.NOT_FOUND)

        pyproject = template_dir / "pyproject.toml"
        if not pyproject.exists():
            return EnvInfo(
                env_id=env_id,
                status=EnvStatus.NOT_FOUND,
                error_message=f"Missing pyproject.toml in {template_dir}",
            )

        # Check if currently installing
        if env_id in self._installing:
            progress = self._installing[env_id]
            if progress.completed_at is None:
                return EnvInfo(
                    env_id=env_id,
                    status=EnvStatus.INSTALLING,
                    template_path=template_dir,
                    install_time_seconds=time.time() - progress.started_at,
                )

        # Check if .venv exists
        if not venv_dir.exists():
            return EnvInfo(
                env_id=env_id,
                status=EnvStatus.NOT_INSTALLED,
                template_path=template_dir,
            )

        # Check if python binary exists (valid installation)
        python_bin = venv_dir / "bin" / "python"
        if not python_bin.exists():
            return EnvInfo(
                env_id=env_id,
                status=EnvStatus.FAILED,
                template_path=template_dir,
                venv_path=venv_dir,
                error_message="Missing python binary in .venv",
            )

        # Check if template has changed since installation
        is_outdated = self._is_template_changed(env_id)

        # Calculate size
        size_mb = self._get_dir_size_mb(venv_dir)

        # Get Python version
        python_version = None
        try:
            result = subprocess.run(
                [str(python_bin), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                python_version = result.stdout.strip().replace("Python ", "")
        except Exception:
            pass

        return EnvInfo(
            env_id=env_id,
            status=EnvStatus.OUTDATED if is_outdated else EnvStatus.READY,
            template_path=template_dir,
            venv_path=venv_dir,
            size_mb=size_mb,
            python_version=python_version,
        )

    def _get_dir_size_mb(self, path: Path) -> float:
        """Calculate directory size in MB."""
        try:
            total = 0
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
            return round(total / (1024 * 1024), 1)
        except Exception:
            return 0.0

    def _get_installed_dir(self, env_id: str) -> Path:
        """Get the directory where installed copies of template files are stored.

        In Docker mode: data_envs_dir/env_id (separate from template)
        In local dev mode: venv_dir (store copies alongside .venv)
        """
        if self.data_envs_dir:
            return self.data_envs_dir / env_id
        # Local dev: store installed copies in .venv to track what was installed
        return self._get_venv_dir(env_id)

    def _is_template_changed(self, env_id: str) -> bool:
        """Check if template files differ from installed copies.

        Compares tracked files (pyproject.toml, post_install.sh) between
        template and installed copies.
        Returns False if no installed copy exists (new install, not outdated).
        """
        template_dir = self._get_template_dir(env_id)
        installed_dir = self._get_installed_dir(env_id)

        # Check each tracked file
        tracked_files = ["pyproject.toml", "post_install.sh"]

        for filename in tracked_files:
            template_file = template_dir / filename
            installed_file = installed_dir / f".installed_{filename}"

            # If template has the file but installed doesn't, outdated
            if template_file.exists() and not installed_file.exists():
                return True

            # If both exist, compare contents
            if template_file.exists() and installed_file.exists():
                if template_file.read_bytes() != installed_file.read_bytes():
                    return True

        return False

    def _mark_files_installed(self, env_id: str, template_dir: Path, files: list[str]) -> None:
        """Copy template files to installed dir to track what version was installed.

        Args:
            env_id: Environment identifier
            template_dir: Source template directory
            files: List of filenames to mark as installed
        """
        installed_dir = self._get_installed_dir(env_id)
        installed_dir.mkdir(parents=True, exist_ok=True)

        for filename in files:
            template_file = template_dir / filename
            if template_file.exists():
                installed_file = installed_dir / f".installed_{filename}"
                shutil.copy(template_file, installed_file)

    def list_environments(self) -> Dict[str, EnvInfo]:
        """
        List all available environments and their status.

        Returns:
            Dict of env_id -> EnvInfo
        """
        envs = {}

        if not self.envs_dir.exists():
            return envs

        for item in self.envs_dir.iterdir():
            if item.is_dir() and (item / "pyproject.toml").exists():
                env_id = item.name
                envs[env_id] = self.get_env_status(env_id)

        return envs

    async def ensure_installed(self, env_id: str) -> EnvInfo:
        """
        Ensure an environment is installed and ready.

        This is the main entry point called by the coordinator before spawning workers.
        Handles installation, updates when templates change, and post-install scripts.

        Args:
            env_id: Environment identifier

        Returns:
            EnvInfo with READY status

        Raises:
            ValueError: If environment template not found
            RuntimeError: If installation fails
        """
        status = self.get_env_status(env_id)

        if status.status == EnvStatus.NOT_FOUND:
            raise ValueError(f"Environment template '{env_id}' not found")

        # If already READY, nothing to do (all tracked files match)
        if status.status == EnvStatus.READY:
            return status

        # OUTDATED means template changed - need to reinstall
        if status.status == EnvStatus.OUTDATED:
            logger.info(f"Environment '{env_id}' is outdated, will reinstall")

        # Need to install - acquire lock to prevent concurrent installs
        lock = self._get_install_lock(env_id)

        async with lock:
            # Re-check status after acquiring lock
            status = self.get_env_status(env_id)
            if status.status == EnvStatus.READY:
                # Another task installed it while we waited
                return status

            # Install (or reinstall if outdated) the environment
            return await self._install_env(env_id)

    async def _install_env(self, env_id: str) -> EnvInfo:
        """
        Install a worker environment.

        Args:
            env_id: Environment identifier

        Returns:
            EnvInfo with installation result

        Raises:
            RuntimeError: If installation fails
        """
        template_dir = self._get_template_dir(env_id)
        venv_dir = self._get_venv_dir(env_id)

        logger.info(f"Installing environment '{env_id}' from {template_dir}")

        # Track progress
        progress = InstallProgress(env_id=env_id, started_at=time.time())
        self._installing[env_id] = progress

        try:
            # If using data_envs_dir, we need to set up the directory structure
            if self.data_envs_dir:
                env_data_dir = self.data_envs_dir / env_id
                env_data_dir.mkdir(parents=True, exist_ok=True)

                # Copy template files to data dir
                shutil.copy(template_dir / "pyproject.toml", env_data_dir / "pyproject.toml")
                if (template_dir / "uv.lock").exists():
                    shutil.copy(template_dir / "uv.lock", env_data_dir / "uv.lock")
                if (template_dir / ".python-version").exists():
                    shutil.copy(template_dir / ".python-version", env_data_dir / ".python-version")
                if (template_dir / "post_install.sh").exists():
                    shutil.copy(template_dir / "post_install.sh", env_data_dir / "post_install.sh")

                cwd = env_data_dir
            else:
                cwd = template_dir

            # Build clean environment without VIRTUAL_ENV from main process
            clean_env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
            clean_env["UV_NO_PROGRESS"] = "1"

            # Generate lockfile if missing
            lockfile = cwd / "uv.lock"
            if not lockfile.exists():
                logger.info(f"Generating lockfile for '{env_id}'")
                lock_cmd = ["uv", "lock"]
                lock_proc = await asyncio.create_subprocess_exec(
                    *lock_cmd,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=clean_env,
                )
                lock_output, _ = await lock_proc.communicate()
                if lock_proc.returncode != 0:
                    error = f"uv lock failed with code {lock_proc.returncode}"
                    logger.error(f"Environment '{env_id}' lock failed: {error}")
                    logger.error(f"Output: {lock_output.decode()}")
                    raise RuntimeError(f"Failed to generate lockfile for '{env_id}': {error}")
                logger.info(f"Lockfile generated for '{env_id}'")

            # Run uv sync --frozen
            cmd = ["uv", "sync", "--frozen"]

            logger.info(f"Running: {' '.join(cmd)} in {cwd}")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=clean_env,
            )

            # Stream output
            output_lines = []
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                line_str = line.decode().rstrip()
                output_lines.append(line_str)
                progress.log_lines.append(line_str)
                logger.debug(f"[{env_id}] {line_str}")

            await proc.wait()

            progress.completed_at = time.time()
            duration = progress.completed_at - progress.started_at

            if proc.returncode != 0:
                error = f"uv sync failed with code {proc.returncode}"
                progress.error = error
                logger.error(f"Environment '{env_id}' installation failed: {error}")
                logger.error(f"Output:\n" + "\n".join(output_lines[-20:]))
                raise RuntimeError(f"Failed to install environment '{env_id}': {error}")

            progress.success = True
            logger.info(f"Environment '{env_id}' installed in {duration:.1f}s")

            # Mark pyproject.toml as installed
            self._mark_files_installed(env_id, template_dir, ["pyproject.toml"])

            # Run post-install script if it exists
            await self._run_post_install(env_id, template_dir, cwd, progress)

            return self.get_env_status(env_id)

        except Exception as e:
            progress.completed_at = time.time()
            progress.error = str(e)
            raise

        finally:
            # Keep progress for a bit for status queries, then clean up
            # (In practice, we keep it until the next install attempt)
            pass

    async def _run_post_install(
        self, env_id: str, template_dir: Path, cwd: Path, progress: InstallProgress
    ) -> None:
        """
        Run post-install script if it exists (during installation with progress tracking).

        Looks for post_install.sh in the template directory and runs it
        with the environment's Python/venv activated.

        Args:
            env_id: Environment identifier
            template_dir: Template directory (contains post_install.sh)
            cwd: Working directory for installation (template or data dir)
            progress: Progress tracker for logging
        """
        post_install_script = template_dir / "post_install.sh"
        if not post_install_script.exists():
            return

        logger.info(f"Running post-install script for '{env_id}'")

        venv_dir = self._get_venv_dir(env_id)
        venv_bin = venv_dir / "bin"

        # Build environment with venv activated
        env = {
            **os.environ,
            "VIRTUAL_ENV": str(venv_dir),
            "PATH": f"{venv_bin}:{os.environ.get('PATH', '')}",
        }
        # Remove PYTHONHOME if set (can interfere with venv)
        env.pop("PYTHONHOME", None)

        proc = await asyncio.create_subprocess_exec(
            "bash",
            str(post_install_script),
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )

        output_lines = []
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            line_str = line.decode().rstrip()
            output_lines.append(line_str)
            progress.log_lines.append(f"[post_install] {line_str}")
            logger.debug(f"[{env_id}:post_install] {line_str}")

        await proc.wait()

        if proc.returncode != 0:
            error = f"post_install.sh failed with code {proc.returncode}"
            logger.error(f"Environment '{env_id}' post-install failed: {error}")
            logger.error(f"Output:\n" + "\n".join(output_lines[-20:]))
            raise RuntimeError(f"Post-install failed for '{env_id}': {error}")

        # Mark post_install.sh as installed
        self._mark_files_installed(env_id, template_dir, ["post_install.sh"])
        logger.info(f"Post-install completed for '{env_id}'")

    async def delete_env(self, env_id: str) -> bool:
        """
        Delete an installed environment to free disk space.

        Args:
            env_id: Environment identifier

        Returns:
            True if deleted, False if not found

        Raises:
            RuntimeError: If environment is currently installing
        """
        status = self.get_env_status(env_id)

        if status.status == EnvStatus.INSTALLING:
            raise RuntimeError(f"Cannot delete environment '{env_id}' while installing")

        if status.status in (EnvStatus.NOT_INSTALLED, EnvStatus.NOT_FOUND):
            return False

        venv_dir = self._get_venv_dir(env_id)
        if venv_dir.exists():
            logger.info(f"Deleting environment '{env_id}' at {venv_dir}")
            shutil.rmtree(venv_dir)

            # If using data_envs_dir, also clean up the copied files
            if self.data_envs_dir:
                env_data_dir = self.data_envs_dir / env_id
                if env_data_dir.exists():
                    shutil.rmtree(env_data_dir)

            return True

        return False

    def get_install_progress(self, env_id: str) -> Optional[InstallProgress]:
        """Get installation progress for an environment."""
        return self._installing.get(env_id)


# Global instance
_env_manager: Optional[EnvironmentManager] = None


def get_env_manager() -> EnvironmentManager:
    """Get or create the global EnvironmentManager."""
    global _env_manager
    if _env_manager is None:
        # Check for HAID_ENVS_DIR environment variable (Docker mode)
        data_envs_dir = os.environ.get("HAID_ENVS_DIR")
        _env_manager = EnvironmentManager(
            data_envs_dir=Path(data_envs_dir) if data_envs_dir else None
        )
    return _env_manager
