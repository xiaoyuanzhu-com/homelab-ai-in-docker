"""EnvironmentManager - On-demand worker environment installation.

Handles:
- Checking if a worker environment is installed
- Installing environments on first use via `uv sync --frozen`
- Tracking installation status and progress
- Environment deletion to free disk space
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

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
        template_hash = self._compute_template_hash(template_dir)
        installed_hash = self._get_installed_hash(env_id)
        # Outdated if we have a hash AND it differs (missing hash = legacy install, treat as OK)
        is_outdated = installed_hash is not None and installed_hash != template_hash

        # If no hash file exists (legacy install), create one to track future changes
        if installed_hash is None:
            self._save_installed_hash(env_id, template_hash)

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

    def _compute_template_hash(self, template_dir: Path) -> str:
        """Compute hash of pyproject.toml and uv.lock to detect changes."""
        hasher = hashlib.sha256()

        for filename in ["pyproject.toml", "uv.lock"]:
            filepath = template_dir / filename
            if filepath.exists():
                hasher.update(filename.encode())
                hasher.update(filepath.read_bytes())

        return hasher.hexdigest()[:16]  # Short hash is enough

    def _get_installed_hash_file(self, env_id: str) -> Path:
        """Get path to the file storing the installed template hash."""
        venv_dir = self._get_venv_dir(env_id)
        return venv_dir.parent / ".template_hash"

    def _get_installed_hash(self, env_id: str) -> Optional[str]:
        """Get the hash of the template used for the current installation."""
        hash_file = self._get_installed_hash_file(env_id)
        if hash_file.exists():
            return hash_file.read_text().strip()
        return None

    def _save_installed_hash(self, env_id: str, template_hash: str) -> None:
        """Save the template hash after successful installation."""
        hash_file = self._get_installed_hash_file(env_id)
        hash_file.parent.mkdir(parents=True, exist_ok=True)
        hash_file.write_text(template_hash)

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
        Ensure an environment is installed, installing if necessary.

        This is the main entry point called by the coordinator before spawning workers.

        Args:
            env_id: Environment identifier

        Returns:
            EnvInfo with READY status

        Raises:
            ValueError: If environment template not found
            RuntimeError: If installation fails
        """
        status = self.get_env_status(env_id)

        if status.status == EnvStatus.READY:
            return status

        if status.status == EnvStatus.NOT_FOUND:
            raise ValueError(f"Environment template '{env_id}' not found")

        # OUTDATED means template changed - need to reinstall
        if status.status == EnvStatus.OUTDATED:
            logger.info(f"Environment '{env_id}' is outdated, will reinstall")

        # Need to install - acquire lock to prevent concurrent installs
        lock = self._get_install_lock(env_id)

        async with lock:
            # Re-check status after acquiring lock
            status = self.get_env_status(env_id)
            if status.status == EnvStatus.READY:
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

                # Copy pyproject.toml and uv.lock to data dir
                shutil.copy(template_dir / "pyproject.toml", env_data_dir / "pyproject.toml")
                if (template_dir / "uv.lock").exists():
                    shutil.copy(template_dir / "uv.lock", env_data_dir / "uv.lock")
                if (template_dir / ".python-version").exists():
                    shutil.copy(template_dir / ".python-version", env_data_dir / ".python-version")

                cwd = env_data_dir
            else:
                cwd = template_dir

            # Run uv sync --frozen
            cmd = ["uv", "sync", "--frozen"]

            logger.info(f"Running: {' '.join(cmd)} in {cwd}")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "UV_NO_PROGRESS": "1"},
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

            # Save template hash to detect future changes
            template_hash = self._compute_template_hash(template_dir)
            self._save_installed_hash(env_id, template_hash)

            return self.get_env_status(env_id)

        except Exception as e:
            progress.completed_at = time.time()
            progress.error = str(e)
            raise

        finally:
            # Keep progress for a bit for status queries, then clean up
            # (In practice, we keep it until the next install attempt)
            pass

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
