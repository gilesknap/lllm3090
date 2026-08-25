#!/usr/bin/env python3
"""FreeToken control panel — local-only start/stop/model-select + live engine log.

Binds 127.0.0.1 by design: these endpoints start processes. Reach it remotely
with an SSH tunnel:  ssh -L 8080:127.0.0.1:8080 giles@ws03
"""
import asyncio, json, os, pathlib, re, shutil, signal, subprocess, time, urllib.request
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

MODELS_DIR = pathlib.Path("/home/giles/models")
START_SH   = MODELS_DIR / "ft-engine-start.sh"
ENV_SH     = MODELS_DIR / "ft-env.sh"
LOG_PATH   = MODELS_DIR / "ft-state" / "logs" / "serve-1919.log"
DAEMON     = "http://127.0.0.1:1900"
ENGINE     = "http://127.0.0.1:1919"
HERE       = pathlib.Path(__file__).parent
# llama.cpp (GGUF) engine — shares port 1919 with FreeToken; only one runs at a time.
LLAMA_DIR  = pathlib.Path("/home/giles/opt/llama.cpp/llama-b10628")
LLAMA_LOG  = MODELS_DIR / "llama-server.log"
LLAMA_PID  = MODELS_DIR / "ft-state" / "llama-server.pid"
LLAMA_CTX  = 131072

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

app = FastAPI(title="FreeToken control")
_busy = asyncio.Lock()
_last: dict = {"action": None, "ok": None, "detail": ""}


def clean(line: str) -> str:
    # strip ANSI colour and collapse \r progress-bar redraws to the final frame
    line = ANSI.sub("", line)
    if "\r" in line:
        line = line.split("\r")[-1]
    return line.rstrip("\n")


def _get(url: str, timeout: float = 3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def llama_pid():
    """PID of a live llama-server, else None (stale pidfiles are ignored)."""
    try:
        pid = int(LLAMA_PID.read_text().strip())
    except Exception:
        return None
    return pid if pathlib.Path(f"/proc/{pid}").exists() else None


def active_log():
    """Whichever engine's log is current. _tail_stream sees the inode change and
    emits its 'restarted' marker, so switching engines re-anchors the stream."""
    return LLAMA_LOG if llama_pid() else LOG_PATH


def discover_models():
    """FreeToken checkpoints (a dir with config.json) and GGUF files, in one list.
    A dir holding both is served by FreeToken — it is the higher-fidelity path."""
    out = []
    for p in sorted(MODELS_DIR.iterdir()):
        if not p.is_dir() or p.name in {"ft-state", "control", "logs"}:
            continue
        if (p / "config.json").exists():
            size = sum(f.stat().st_size for f in p.glob("*.safetensors"))
            out.append({"name": p.name, "path": str(p),
                        "gb": round(size / 1e9, 1), "kind": "freetoken"})
            continue
        ggufs = sorted(p.glob("*.gguf"))
        if ggufs:
            size = sum(f.stat().st_size for f in ggufs)
            out.append({"name": p.name, "path": str(ggufs[0]),
                        "gb": round(size / 1e9, 1), "kind": "gguf"})
    return out


def _stop_llama(timeout: int = 30):
    """SIGTERM the llama-server and wait for the VRAM to actually come back."""
    pid = llama_pid()
    if pid is None:
        LLAMA_PID.unlink(missing_ok=True)
        return True, "no llama-server running"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        LLAMA_PID.unlink(missing_ok=True)
        return True, "already gone"
    for _ in range(timeout * 2):
        if not pathlib.Path(f"/proc/{pid}").exists():
            LLAMA_PID.unlink(missing_ok=True)
            return True, f"stopped llama-server (pid {pid})"
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    LLAMA_PID.unlink(missing_ok=True)
    return True, f"killed llama-server (pid {pid})"


def _start_llama(gguf_path: str, name: str, ctx: int):
    """Launch llama-server on 1919. Blocks until /health answers, like
    ft-engine-start.sh does for FreeToken, so the caller learns of a real failure."""
    env = dict(os.environ, LD_LIBRARY_PATH=str(LLAMA_DIR))
    with LLAMA_LOG.open("wb") as log:
        proc = subprocess.Popen(
            [str(LLAMA_DIR / "llama-server"),
             "--model", gguf_path, "--alias", name,
             "--host", "127.0.0.1", "--port", "1919",
             "--n-gpu-layers", "999", "--ctx-size", str(ctx),
             "-fa", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
             "--jinja"],
            stdout=log, stderr=subprocess.STDOUT, env=env,
            start_new_session=True,
        )
    LLAMA_PID.write_text(str(proc.pid))
    for _ in range(240):                      # up to ~4 min for a cold load
        if proc.poll() is not None:
            LLAMA_PID.unlink(missing_ok=True)
            return False, f"llama-server exited with {proc.returncode}"
        if _get(f"{ENGINE}/health", timeout=2):
            return True, f"llama-server ready: {name} @ {ctx} ctx"
        time.sleep(1)
    return False, "llama-server did not become ready in time"


def _sh(cmd: list[str], timeout: int):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout or "")[-1500:] + (r.stderr or "")[-1500:]
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"


@app.get("/api/status")
def status():
    eng = _get(f"{DAEMON}/engine/status")
    health = _get(f"{DAEMON}/health")
    served = _get(f"{ENGINE}/v1/models", timeout=2)
    lpid = llama_pid()
    model_id = ctx = None
    if served and served.get("data"):
        model_id = served["data"][0].get("id")
        ctx = served["data"][0].get("context_length")
    elif served and served.get("models"):          # llama.cpp shape
        model_id = served["models"][0].get("name")
    if ctx is None and lpid:
        ctx = LLAMA_CTX                            # llama.cpp does not report it
    vram = None
    if shutil.which("nvidia-smi"):
        ok, out = _sh(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                       "--format=csv,noheader,nounits"], 10)
        if ok and out.strip():
            u, t = (x.strip() for x in out.strip().splitlines()[0].split(","))
            vram = {"used_mb": int(u), "total_mb": int(t)}
    if lpid:
        eng = {"running": True, "starting": False, "stopping": False, "adopted": False,
               "pid": lpid, "port": 1919, "model": model_id or "llama-server",
               "uptimeS": eng.get("uptimeS", 0) if isinstance(eng, dict) else 0}
    return {
        "daemon_up": health is not None,
        "engine_kind": "gguf" if lpid else "freetoken",
        "engine": eng or {},
        "answering": model_id is not None,
        "model_id": model_id,
        "context_length": ctx,
        "vram": vram,
        "busy": _busy.locked(),
        "last": _last,
        "models": discover_models(),
    }


@app.get("/api/logs")
def logs(tail: int = 200):
    LOG = active_log()
    if not LOG.exists():
        return {"lines": []}
    with LOG.open("r", errors="replace") as f:
        buf = f.readlines()[-tail:]
    return {"lines": [clean(l) for l in buf if clean(l).strip()]}


async def _tail_stream():
    """Follow the engine log across truncation and replacement.

    Line-boundary safe: the initial seek is advanced to the next newline, and any
    incomplete trailing write is held back until its newline arrives — otherwise a
    fragment gets emitted as if it were a whole line.
    """
    pos, ino, pending = 0, None, ""
    LOG = active_log()
    if LOG.exists():
        st = LOG.stat()
        ino = st.st_ino
        start = max(0, st.st_size - 4000)
        with LOG.open("r", errors="replace") as f:
            f.seek(start)
            if start:
                f.readline()          # discard the partial line we landed inside
            pos = f.tell()
    yield "retry: 3000\n\n"
    while True:
        try:
            LOG = active_log()
            if not LOG.exists():
                await asyncio.sleep(1); continue
            st = LOG.stat()
            if ino is None or st.st_ino != ino or st.st_size < pos:
                ino, pos, pending = st.st_ino, 0, ""   # rotated or truncated
                yield "event: rotated\ndata: --- engine log restarted ---\n\n"
            if st.st_size > pos:
                with LOG.open("r", errors="replace") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
                data = pending + chunk
                if data.endswith("\n"):
                    lines, pending = data.splitlines(), ""
                else:
                    parts = data.split("\n")
                    lines, pending = parts[:-1], parts[-1]
                for raw in lines:
                    line = clean(raw)
                    if line.strip():
                        yield f"data: {json.dumps(line)}\n\n"
            else:
                yield ": keepalive\n\n"
            await asyncio.sleep(0.7)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield f"data: {json.dumps('[tail error] ' + str(e))}\n\n"
            await asyncio.sleep(2)


@app.get("/api/logstream")
async def logstream():
    return StreamingResponse(_tail_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"})


async def _stop_all():
    """Free the GPU whichever engine holds it. Both are stopped unconditionally:
    a stale FreeToken process and a live llama-server can otherwise fight over 1919."""
    lok, lout = await asyncio.to_thread(_stop_llama)
    fok, fout = await asyncio.to_thread(
        _sh, ["/bin/bash", "-lc", f". {ENV_SH}; ft daemon stop --timeout 40"], 120)
    return (lok and fok), f"{lout}; {fout.strip()[-300:]}"


@app.post("/api/stop")
async def stop():
    if _busy.locked():
        return JSONResponse({"error": "busy"}, status_code=409)
    async with _busy:
        ok, out = await _stop_all()
        _last.update(action="stop", ok=ok, detail=out.strip()[-400:])
    return {"ok": ok, "detail": _last["detail"]}


@app.post("/api/start")
async def start(model: str, ctx: int = LLAMA_CTX):
    entry = next((m for m in discover_models() if m["name"] == model), None)
    if entry is None:
        return JSONResponse({"error": f"unknown model {model!r}"}, status_code=400)
    if _busy.locked():
        return JSONResponse({"error": "busy"}, status_code=409)
    async with _busy:
        await _stop_all()                      # one GPU: never two engines at once
        if entry["kind"] == "gguf":
            ok, out = await asyncio.to_thread(
                _start_llama, entry["path"], entry["name"], ctx)
        else:
            ok, out = await asyncio.to_thread(
                _sh, ["/bin/bash", str(START_SH), entry["path"]], 900)
        _last.update(action=f"start {model}", ok=ok, detail=out.strip()[-400:])
    return {"ok": ok, "detail": _last["detail"]}


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index.html").read_text()
