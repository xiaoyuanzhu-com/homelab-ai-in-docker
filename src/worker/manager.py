"""Manager for isolated OCR workers.

Spawns per-model workers that linger for an idle timeout and serve HTTP locally.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from urllib import request as urlrequest
from urllib.error import URLError, HTTPError

from src.db.settings import get_setting_int


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class _Worker:
    model_key: str
    model_id: str
    language: Optional[str]
    port: int
    proc: subprocess.Popen
    last_active: float = field(default_factory=time.time)
    shutdown_task: Optional[asyncio.Task] = None


class OCRWorkerManager:
    def __init__(self) -> None:
        self._workers: Dict[str, _Worker] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _model_key(model_id: str, language: Optional[str]) -> str:
        return f"{model_id}::lang={language or ''}"

    async def _spawn_worker(self, model_id: str, language: Optional[str]) -> _Worker:
        port = _find_free_port()
        idle = get_setting_int("model_idle_timeout_seconds", 5)
        cmd = [
            sys.executable,
            "-m",
            "src.worker.image_ocr_worker",
            "--model-id",
            model_id,
            "--port",
            str(port),
            "--idle-timeout",
            str(idle),
        ]
        if language:
            cmd.extend(["--language", language])

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        proc = subprocess.Popen(cmd, env=env)
        worker = _Worker(
            model_key=self._model_key(model_id, language),
            model_id=model_id,
            language=language,
            port=port,
            proc=proc,
        )

        # Wait until healthz responds or timeout
        await self._wait_ready(worker, timeout_s=120)
        return worker

    async def _wait_ready(self, worker: _Worker, timeout_s: int = 60) -> None:
        url = f"http://127.0.0.1:{worker.port}/healthz"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if worker.proc.poll() is not None:
                raise RuntimeError("Worker exited before becoming ready")
            try:
                with urlrequest.urlopen(url, timeout=1) as resp:
                    if resp.status == 200:
                        return
            except (URLError, HTTPError, ConnectionError):
                await asyncio.sleep(0.2)
        raise TimeoutError("Worker failed to become ready in time")

    async def _ensure_worker(self, model_id: str, language: Optional[str]) -> _Worker:
        async with self._lock:
            key = self._model_key(model_id, language)
            w = self._workers.get(key)
            if w is not None and w.proc.poll() is None:
                return w
            # Clean stale
            if w is not None:
                self._workers.pop(key, None)
            w = await self._spawn_worker(model_id, language)
            self._workers[key] = w
            return w

    async def _schedule_shutdown(self, worker: _Worker) -> None:
        idle = get_setting_int("model_idle_timeout_seconds", 5)
        # Cancel previous
        if worker.shutdown_task and not worker.shutdown_task.done():
            worker.shutdown_task.cancel()

        async def _watchdog():
            try:
                await asyncio.sleep(idle)
                # Double-check still idle
                if (time.time() - worker.last_active) >= idle:
                    await self._terminate_worker(worker)
            except asyncio.CancelledError:
                pass

        worker.shutdown_task = asyncio.create_task(_watchdog())

    async def _terminate_worker(self, worker: _Worker) -> None:
        if worker.proc.poll() is not None:
            return
        # Best-effort graceful shutdown via HTTP
        try:
            url = f"http://127.0.0.1:{worker.port}/shutdown"
            req = urlrequest.Request(url, method="POST")
            with urlrequest.urlopen(req, timeout=1) as _:
                pass
        except Exception:
            pass

        # Send SIGTERM, then SIGKILL fallback
        try:
            worker.proc.terminate()
        except Exception:
            pass
        try:
            worker.proc.wait(timeout=5)
        except Exception:
            try:
                worker.proc.kill()
            except Exception:
                pass
        # Remove from registry
        self._workers.pop(worker.model_key, None)

    async def infer(self, model_id: str, language: Optional[str], image_b64: str) -> Dict:
        w = await self._ensure_worker(model_id, language)
        # Send inference request (sync HTTP in thread)
        url = f"http://127.0.0.1:{w.port}/infer"
        import json as _json
        payload = _json.dumps({"image": image_b64}).encode()

        def _do_request():
            req = urlrequest.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urlrequest.urlopen(req, timeout=60) as resp:
                import json
                return json.loads(resp.read().decode())

        data = await asyncio.to_thread(_do_request)
        w.last_active = time.time()
        await self._schedule_shutdown(w)
        return data


# Singleton manager for importers
manager = OCRWorkerManager()
