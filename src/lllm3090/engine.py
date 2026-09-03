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
import urllib.error
import urllib.request
from importlib import resources
from pathlib import Path

from . import config, engines, gguf, speculation

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

    This lives with the code that writes the log rather than with the panel,
    because the log is the engine's rather than any front end's: the panel
    streams it to a browser over SSE, and ``lllm3090 status`` and a shell on
    the same box read the same file.

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
    """The llama-server that will serve, which is not always the installed one.

    Resolved per call rather than held in a constant: the panel is a
    long-running process, and an engine chosen in the browser has to reach the
    next start without a restart of the thing the browser is talking to.
    """
    return engines.active_dir() / "llama-server"


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
                env=dict(os.environ, LD_LIBRARY_PATH=str(server_binary().parent)),
            )
            _SUPPORTS[key] = flag in (out.stdout + out.stderr)
        except Exception:
            # Unable to ask is not permission to assume. The flag is an
            # optimisation; the start is not.
            _SUPPORTS[key] = False
    return _SUPPORTS[key]


#: What every KV cache this engine allocates is quantised to.
#:
#: One constant for both caches, because there are two and the bug this fixes
#: was that they disagreed. ``--cache-type-k/v`` sizes the main cache;
#: ``--spec-draft-type-k/v`` sizes the draft model's, which llama.cpp defaults
#: to ``f16`` and does *not* inherit from the main one. So for as long as only
#: the first pair was passed, the model's own draft context was held at full
#: precision beside a main cache at half -- and that gap was the whole of what
#: multi-token prediction cost in memory.
#:
#: q8_0 halves the cost per token against f16 for close to no quality loss, and
#: is what makes long context affordable on 24 GB. q4_0 would halve it again
#: but degrades long-context reasoning, so it is deliberately not offered.
CACHE_TYPE = "q8_0"


def spec_flags(profile: speculation.Profile, model_path: str) -> list[str]:
    """The speculation arguments for this profile against this checkpoint.

    Two things can silently remove a flag here, and both have to be checked on
    an install that works:

    * **The binary.** The engine build is pinned but not upgraded in place --
      ``install-engine`` skips replacement when a binary is already there --
      so a binary older than ``--spec-type`` survives an ``lllm3090`` upgrade
      and would exit on an argument it has never heard of.
    * **The checkpoint.** Only the file on disk decides whether the MTP head is
      there, and llama.cpp refuses to start with ``--spec-type draft-mtp``
      against one that lacks it. See :mod:`lllm3090.gguf`.

    Dropping ``draft-mtp`` from a checkpoint that has no head is right for the
    default profile and wrong for a named one, which is why a named profile is
    refused earlier, in :func:`start`, rather than quietly reshaped here.
    """
    if not supports("--spec-type"):
        return []
    types = tuple(
        t for t in profile.spec_types
        if t != "draft-mtp" or gguf.has_mtp(model_path)
    )
    if not types:
        return []
    flags = ["--spec-type", ",".join(types)]
    if profile.draft_n_max is not None and supports("--spec-draft-n-max"):
        flags += ["--spec-draft-n-max", str(profile.draft_n_max)]
    # A drafter that is a *model* keeps its own KV cache, and it is not sized
    # by --cache-type-k/v: llama.cpp defaults it to f16 whatever the main cache
    # is set to. Quantising it gives back 2.45 of the 4.80 KiB/token that MTP
    # costs on the dense 27B -- about 412 MiB at a 168k window -- and moves the
    # ceiling from 200704 tokens to 208896.
    #
    # It cannot change what the model says. Every draft is verified by the full
    # model whatever cache produced it, so a coarser draft can only be accepted
    # less often, and measured either side it is not: 43.7 -> 44.1 tok/s at a
    # short prompt and 32.3 -> 33.3 at 62k. (The acceptance figures moved too,
    # by less than this instrument swings on content alone, so the right
    # reading is "no penalty" rather than "faster".)
    #
    # Only for a draft model. The n-gram modes draft from the prompt and have
    # no context to quantise, so passing this for them would set a flag that
    # describes nothing.
    if any(t.startswith("draft-") for t in types) and supports(
        "--spec-draft-type-k"
    ) and supports("--spec-draft-type-v"):
        flags += [
            "--spec-draft-type-k", CACHE_TYPE,
            "--spec-draft-type-v", CACHE_TYPE,
        ]
    return flags
#: Levels ``llama-server --reasoning-effort`` accepts, less ``default`` --
#: which means "pass nothing", and is spelled here by leaving ``--effort`` off.
#:
#: The engine takes the level and hands it to the chat template. What the
#: *template* does with it is the template's business: Qwen3.8-27B's implements
#: ``low``, ``medium`` and ``xhigh``, folds ``high`` into ``xhigh``, and raises
#: on everything else. So this list is what the engine will accept, not what
#: any given model will render. :func:`template_refuses_effort` is what closes
#: that gap.
REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")


def template_refuses_effort(effort: str, timeout: float = 5.0) -> str | None:
    """The chat template's complaint about ``effort``, or ``None`` if it renders.

    A template that does not implement a level raises while rendering the
    prompt, which is a 500 on *every* request. The engine starts, loads the
    weights, and answers ``/health`` regardless -- so without this the start
    reports success and the failure arrives later, per request, as a Jinja
    traceback that does not obviously point back at a flag typed minutes ago.

    ``/apply-template`` renders a prompt and generates nothing, so asking costs
    a round trip. Anything other than a rendering error -- a build without the
    endpoint, a connection that fails -- returns ``None``: not being able to
    check is not evidence of a problem.
    """
    body = json.dumps({
        "reasoning_effort": effort,
        "messages": [{"role": "user", "content": "ping"}],
    }).encode()
    req = urllib.request.Request(
        f"{config.ENGINE_URL}/apply-template",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return None
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("error", {}).get("message", "")
        except Exception:
            return None
        # The engine returns the whole Jinja traceback, source line included.
        # The last line carries the template's own message, which is the half
        # written for a human.
        lines = [line for line in str(detail).splitlines() if line.strip()]
        return lines[-1] if lines else "the chat template rejected it"
    except Exception:
        return None


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
    spec: speculation.Profile | None = None,
    effort: str | None = None,
) -> tuple[bool, str]:
    """Launch llama-server and block until it answers.

    ``wait`` is how many seconds to poll for readiness; 0 returns as soon as the
    process is launched.

    ``spec`` is what the engine guesses ahead with; ``None`` is
    :data:`lllm3090.speculation.DEFAULT`, which is what every start got before
    there was a choice. Whether a profile is allowed on the installed backend
    is settled by :func:`lllm3090.speculation.resolve` before we get here --
    this only has to refuse the one thing that cannot be known until the
    checkpoint is in hand.

    ``ctx`` is the whole KV pool and ``parallel`` is how many conversations
    share it, so each slot gets ``ctx // parallel`` tokens. What to pass is
    decided by :func:`lllm3090.catalog.plan`, which fills one conversation to
    the model's ceiling before opening a second; an agent that wants room for a
    subagent has to ask for it.

    Both caches are quantised to :data:`CACHE_TYPE` -- the main one and the
    draft model's, which is a separate cache with a separate flag and does not
    inherit the setting. See :func:`spec_flags`.

    ``effort`` is a reasoning level from :data:`REASONING_EFFORTS`, given to the
    chat template for the life of the process. It is the only way to reach a
    model's thinking-length control from a harness that has no idea it exists:
    Claude Code's ``/effort`` travels as ``output_config.effort``, a field
    llama.cpp does not implement and does not reject, so it is dropped in
    silence and the template's own default -- ``xhigh`` on Qwen3.8-27B -- stands.
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
    if effort and not supports("--reasoning-effort"):
        # Refused rather than dropped. An optimisation this project adds by
        # itself (see --spec-type below) is right to fall away silently on an
        # older binary; a level the user typed is not, and dropping it would
        # reproduce exactly the silence that makes this flag necessary.
        return False, (
            "this llama-server is too old for --reasoning-effort; "
            "replace it with 'lllm3090 install-engine --force'"
        )
    if mmproj:
        if not Path(mmproj).exists():
            return False, f"projector not found: {mmproj}"
        # The projector is a GGUF too, and a wrong one fails the same way.
        bad = not_a_gguf(Path(mmproj))
        if bad:
            return False, bad
    spec = spec or speculation.DEFAULT
    if spec.needs_mtp and not gguf.has_mtp(model_path):
        # A named profile is a request, and this one is a measurement of a
        # configuration that includes the MTP head. Serving the rest of it
        # under the same name would be a different engine wearing the number.
        return False, (
            f"--profile {spec.name} needs a multi-token prediction head and "
            f"{name} does not carry one. Its figures were measured with "
            "draft-mtp in the mix; without it they describe nothing."
        )

    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LD_LIBRARY_PATH=str(server_binary().parent))
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
                "--cache-type-k", CACHE_TYPE,
                "--cache-type-v", CACHE_TYPE,
                "--jinja",
                # What the engine guesses ahead with. The default is the MTP
                # head alone at llama.cpp's own draft width, which is what has
                # always shipped; anything else is a measured profile the user
                # asked for by name. Which settings win is a property of the
                # backend rather than of the drafter -- see
                # lllm3090.speculation for the numbers and why.
                *spec_flags(spec, model_path),
                # Vision: the projector is a separate GGUF that turns image
                # input into embeddings the model can attend to.
                *(["--mmproj", mmproj] if mmproj else []),
                # Server-wide, because there is nowhere else to put it: the
                # level is a launch argument, not a per-request one, on the
                # endpoint an agent talks to. See REASONING_EFFORTS.
                *(["--reasoning-effort", effort] if effort else []),
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
            complaint = template_refuses_effort(effort) if effort else None
            if complaint:
                # A loaded engine that fails every request is worse than none:
                # it answers /health, so anything watching it reports it up.
                stop()
                return False, (
                    f"{name} does not accept --effort {effort}: {complaint}"
                )
            note = f", effort {effort}" if effort else ""
            return True, (
                f"engine ready: {name} -- {per} tokens x {parallel} slots "
                f"(pool {ctx}{note})"
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
