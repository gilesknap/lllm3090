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
import re
import signal
import subprocess
import time
import urllib.request
from importlib import resources
from pathlib import Path

from . import config, gguf

#: Colour and cursor escapes llama.cpp writes when it thinks it has a terminal.
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

#: First four bytes of every GGUF file. The engine reads no other container --
#: the older GGML and GGJT magics were dropped from llama.cpp long before the
#: builds this installs -- so this is the whole test.
GGUF_MAGIC = b"GGUF"


def clean(line: str) -> str:
    """Strip ANSI colour and collapse a progress-bar redraw to its last frame."""
    line = ANSI.sub("", line)
    if "\r" in line:
        line = line.split("\r")[-1]
    return line.rstrip("\n")


#: How much of the end of the log is worth reading to find its last lines.
#: An engine that has been up for a day has logged every request it served.
TAIL_BYTES = 256 * 1024


def tail(lines: int = 200) -> list[str]:
    """The last ``lines`` readable lines of the engine log.

    This lives with the code that writes the log rather than with either front
    end, because both read it: the panel streams it to a browser over SSE, and
    the terminal UI -- which is on the same machine by construction, since the
    panel binds loopback -- simply re-reads the end of the file.

    Split on newlines alone, deliberately. Python's universal newlines turn a
    carriage return into a line ending, which silently makes a progress-bar
    redraw into hundreds of near-identical lines -- the exact thing
    :func:`clean` exists to collapse.
    """
    log = config.ENGINE_LOG
    if not log.exists():
        return []
    start_at = max(0, log.stat().st_size - TAIL_BYTES)
    with log.open("r", errors="replace", newline="") as f:
        f.seek(start_at)
        buf = f.read().split("\n")
    if start_at:
        # The seek landed inside a line; that fragment is not one.
        buf = buf[1:]
    # Empties are dropped before the count, not after, so asking for twenty
    # lines of log gets twenty lines of log rather than whatever survives.
    readable = [text for text in (clean(line) for line in buf) if text.strip()]
    # `readable[-0:]` is the whole log, which is the opposite of what asking for
    # no lines means. The panel takes this count straight off a query string.
    return readable[-lines:] if lines > 0 else []


def _get(url: str, timeout: float = 3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def server_binary() -> Path:
    return config.LLAMA_DIR / "llama-server"


#: Answers to :func:`supports`, keyed by the binary that gave them.
#:
#: Keyed on modification time as well as path, because ``install-engine
#: --force`` replaces the binary underneath a panel that has been running for
#: days, and an answer cached from the old one would outlive it.
_SUPPORTS: dict[tuple[str, float, str], bool] = {}


def supports(flag: str) -> bool:
    """Whether the installed llama-server accepts ``flag``.

    The engine build is pinned, but the *installed* build is whatever is on
    disk: ``install-engine`` skips replacement when a binary is already there,
    and ``setup`` calls it that way, so upgrading lllm3090 does not upgrade
    llama.cpp. A flag this project learned about in one release can therefore
    meet a binary from an earlier one.

    Passing an unknown flag is not a soft failure -- llama-server exits, and
    the panel reports "starting" over a process that is already gone. Asking
    first costs one ``--help`` per binary per process.
    """
    binary = server_binary()
    try:
        key = (str(binary), binary.stat().st_mtime, flag)
    except OSError:
        return False
    if key not in _SUPPORTS:
        try:
            out = subprocess.run(
                [str(binary), "--help"],
                capture_output=True, text=True, timeout=30, check=False,
                env=dict(os.environ, LD_LIBRARY_PATH=str(config.LLAMA_DIR)),
            )
            _SUPPORTS[key] = flag in (out.stdout + out.stderr)
        except Exception:
            # Unable to ask is not permission to assume. The flag is an
            # optimisation; the start is not.
            _SUPPORTS[key] = False
    return _SUPPORTS[key]


def alive(target: int) -> bool:
    """Is this PID still a process?

    ``/proc`` rather than ``kill(pid, 0)``: the panel and the CLI both call
    this, and only one of them is the engine's parent, so a check that depends
    on signal permissions would answer differently depending on who asked.
    """
    return Path(f"/proc/{target}").exists()


def pid() -> int | None:
    """PID of the running engine, or None. Stale pidfiles are ignored."""
    try:
        value = int(config.ENGINE_PID.read_text().strip())
    except Exception:
        return None
    return value if alive(value) else None


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


def served_slots() -> int | None:
    """How many conversations the running engine can hold at once, or None.

    Asked rather than derived from the ``--parallel`` it was started with: a
    start can override that, and this project is not necessarily the thing
    that started the engine answering on this port. ``/props`` reports what it
    is really serving.
    """
    data = _get(f"{config.ENGINE_URL}/props", timeout=2)
    if not isinstance(data, dict):
        return None
    slots = data.get("total_slots")
    return slots if isinstance(slots, int) and slots > 0 else None


def healthy() -> bool:
    return _get(f"{config.ENGINE_URL}/health", timeout=2) is not None


def not_a_gguf(path: Path) -> str | None:
    """Why ``path`` is not a GGUF, or ``None`` if it is one.

    A pre-flight rather than a formality. llama.cpp rejects a non-GGUF itself,
    but it does so a second or two after launch, in the log -- and the panel
    starts the engine with ``wait=0``, so by then it has already told the user
    "starting", written a pidfile, and gone back to polling a process that is
    about to exit. Reading four bytes here turns that into a refusal with a
    reason, before anything is launched.
    """
    try:
        with path.open("rb") as f:
            magic = f.read(4)
    except OSError as e:
        return f"could not read {path}: {e}"
    if magic != GGUF_MAGIC:
        return (
            f"{path} is not a GGUF file (starts with {magic!r}, "
            f"expected {GGUF_MAGIC!r})"
        )
    return None


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
        if not alive(target):
            config.ENGINE_PID.unlink(missing_ok=True)
            return True, f"stopped engine (pid {target})"
        time.sleep(0.5)
    try:
        os.kill(target, signal.SIGKILL)
    except ProcessLookupError:
        pass
    config.ENGINE_PID.unlink(missing_ok=True)
    return True, f"killed engine (pid {target})"


def start(
    model_path: str,
    name: str,
    ctx: int,
    parallel: int = 1,
    wait: int = 300,
    chat_template: str | None = None,
    mmproj: str | None = None,
) -> tuple[bool, str]:
    """Launch llama-server and block until it answers.

    ``wait`` is how many seconds to poll for readiness; 0 returns as soon as the
    process is launched.

    ``ctx`` is the whole KV pool and ``parallel`` is how many conversations
    share it, so each slot gets ``ctx // parallel`` tokens. What to pass is
    decided by :func:`lllm3090.catalog.plan`, which fills one conversation to
    the model's ceiling before opening a second; an agent that wants room for a
    subagent has to ask for it.

    The cache is quantised to ``q8_0``, which halves its cost per token for
    close to no quality loss and is what makes long context affordable on 24 GB.
    """
    binary = server_binary()
    if not binary.exists():
        return False, (
            f"llama-server not found at {binary}; run 'lllm3090 install-engine'"
        )
    if not Path(model_path).exists():
        return False, f"model file not found: {model_path}"
    bad = not_a_gguf(Path(model_path))
    if bad:
        return False, bad
    if mmproj:
        if not Path(mmproj).exists():
            return False, f"projector not found: {mmproj}"
        # The projector is a GGUF too, and a wrong one fails the same way.
        bad = not_a_gguf(Path(mmproj))
        if bad:
            return False, bad

    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LD_LIBRARY_PATH=str(config.LLAMA_DIR))
    # O_TRUNC so each run starts with its own log, and O_APPEND so every write
    # lands at the end of the file rather than at the engine's own offset. The
    # second matters because the log can be emptied from under a running
    # engine: without it, the next line would be written at the offset it had
    # reached, behind a hole of NULs the length of what was cleared.
    fd = os.open(
        config.ENGINE_LOG,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_APPEND,
        0o644,
    )
    with os.fdopen(fd, "wb") as log:
        proc = subprocess.Popen(
            [
                str(binary),
                "--model", model_path,
                "--alias", name,
                "--host", "127.0.0.1",
                "--port", str(config.ENGINE_PORT),
                "--n-gpu-layers", "999",
                "--ctx-size", str(ctx),
                "--parallel", str(parallel),
                "-fa", "on",
                "--cache-type-k", "q8_0",
                "--cache-type-v", "q8_0",
                "--jinja",
                # Multi-token prediction, when the checkpoint carries the head.
                #
                # The model drafts its own next few tokens and verifies them in
                # one pass, so accepted drafts cost a fraction of a forward
                # pass. Measured on the reference 3090: Qwen3.8-27B 34.9 ->
                # 56.6 tok/s (1.62x), Qwen3.6-35B-A3B-MTP 130.5 -> 171.8
                # (1.32x), 179.9 on code editing.
                #
                # Read from the file rather than the catalogue on purpose: only
                # the checkpoint on disk decides whether the head is there, and
                # llama.cpp refuses to start with this flag against one that
                # lacks it. See lllm3090.gguf.
                #
                # Both halves are checked, because both can be false on a
                # working install: the engine build is pinned but not upgraded
                # in place, so a binary older than --spec-type survives an
                # lllm3090 upgrade and would exit on an argument it has never
                # heard of.
                *(
                    ["--spec-type", "draft-mtp"]
                    if supports("--spec-type") and gguf.has_mtp(model_path)
                    else []
                ),
                # Vision: the projector is a separate GGUF that turns image
                # input into embeddings the model can attend to.
                *(["--mmproj", mmproj] if mmproj else []),
                *(
                    ["--chat-template-file", str(
                        resources.files("lllm3090.data").joinpath(chat_template)
                    )]
                    if chat_template
                    else []
                ),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,   # survives a panel restart
        )
    config.ENGINE_PID.write_text(str(proc.pid))

    if wait == 0:
        # Fire and forget. The panel uses this: holding an HTTP request open for
        # a multi-minute model load makes the panel unresponsive to everything
        # else, and makes a systemd stop hang until it is SIGKILLed. Callers
        # poll status() instead, which already distinguishes loading from ready.
        return True, f"starting {name} -- {ctx // parallel} tokens x {parallel} slots"

    for _ in range(wait):
        if proc.poll() is not None:
            config.ENGINE_PID.unlink(missing_ok=True)
            tail = ""
            if config.ENGINE_LOG.exists():
                tail = config.ENGINE_LOG.read_text(errors="replace")[-400:]
            return False, f"engine exited with {proc.returncode}: {tail}"
        if healthy():
            per = ctx // parallel
            return True, (
                f"engine ready: {name} -- {per} tokens x {parallel} slots "
                f"(pool {ctx})"
            )
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
