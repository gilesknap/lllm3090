"""Lifecycle for the llama.cpp server.

One GPU means one engine, so this module owns a single process tracked by a
pidfile. Start blocks until the server actually answers -- the HTTP port binds
long before the weights finish uploading, so "the port is open" is not the same
as "the model is ready", and reporting the former as the latter is the most
common way to make a slow load look like a failure.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from . import config


def _get(url: str, timeout: float = 3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def server_binary() -> Path:
    return config.LLAMA_DIR / "llama-server"


def pid() -> int | None:
    """PID of the running engine, or None. Stale pidfiles are ignored."""
    try:
        value = int(config.ENGINE_PID.read_text().strip())
    except Exception:
        return None
    return value if Path(f"/proc/{value}").exists() else None


def served_model() -> str | None:
    """Model id the engine reports, or None if it is not answering yet."""
    data = _get(f"{config.ENGINE_URL}/v1/models", timeout=2)
    if not data:
        return None
    entries = data.get("data") or data.get("models") or []
    if not entries:
        return None
    first = entries[0]
    return first.get("id") or first.get("name")


def healthy() -> bool:
    return _get(f"{config.ENGINE_URL}/health", timeout=2) is not None


def stop(timeout: int = 40) -> tuple[bool, str]:
    """Terminate the engine and wait for the VRAM to actually come back."""
    target = pid()
    if target is None:
        config.ENGINE_PID.unlink(missing_ok=True)
        return True, "no engine running"
    try:
        os.kill(target, signal.SIGTERM)
    except ProcessLookupError:
        config.ENGINE_PID.unlink(missing_ok=True)
        return True, "engine already gone"
    for _ in range(timeout * 2):
        if not Path(f"/proc/{target}").exists():
            config.ENGINE_PID.unlink(missing_ok=True)
            return True, f"stopped engine (pid {target})"
        time.sleep(0.5)
    try:
        os.kill(target, signal.SIGKILL)
    except ProcessLookupError:
        pass
    config.ENGINE_PID.unlink(missing_ok=True)
    return True, f"killed engine (pid {target})"


def start(model_path: str, name: str, ctx: int, wait: int = 300) -> tuple[bool, str]:
    """Launch llama-server and block until it answers.

    The KV cache is quantised to ``q8_0``, which halves its cost per token for
    close to no quality loss and is what makes long context affordable on 24 GB.
    """
    binary = server_binary()
    if not binary.exists():
        return False, (
            f"llama-server not found at {binary}; run 'llm3090 install-engine'"
        )
    if not Path(model_path).exists():
        return False, f"model file not found: {model_path}"

    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LD_LIBRARY_PATH=str(config.LLAMA_DIR))
    with config.ENGINE_LOG.open("wb") as log:
        proc = subprocess.Popen(
            [
                str(binary),
                "--model", model_path,
                "--alias", name,
                "--host", "127.0.0.1",
                "--port", str(config.ENGINE_PORT),
                "--n-gpu-layers", "999",
                "--ctx-size", str(ctx),
                "-fa", "on",
                "--cache-type-k", "q8_0",
                "--cache-type-v", "q8_0",
                "--jinja",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,   # survives a panel restart
        )
    config.ENGINE_PID.write_text(str(proc.pid))

    for _ in range(wait):
        if proc.poll() is not None:
            config.ENGINE_PID.unlink(missing_ok=True)
            tail = ""
            if config.ENGINE_LOG.exists():
                tail = config.ENGINE_LOG.read_text(errors="replace")[-400:]
            return False, f"engine exited with {proc.returncode}: {tail}"
        if healthy():
            return True, f"engine ready: {name} @ {ctx} context"
        time.sleep(1)
    return False, "engine did not become ready in time"


def status() -> dict:
    """Everything the panel needs to render engine state in one call."""
    running = pid()
    return {
        "running": running is not None,
        "pid": running,
        "port": config.ENGINE_PORT,
        "answering": healthy() if running else False,
        "model": served_model() if running else None,
    }
