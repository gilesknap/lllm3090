"""The control panel: a loopback-only web UI for starting, stopping and
downloading models.

Binds 127.0.0.1 by design. These endpoints start processes and write to disk
with no authentication, so reach it remotely with an SSH tunnel
(``ssh -L 8080:127.0.0.1:8080 host``) rather than by binding a LAN address.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
from contextlib import asynccontextmanager
from importlib import resources

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import catalog, config, downloads, engine, hardware
from ._version import __version__

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Pick up anything a previous panel process left half-downloaded."""
    try:
        resumed = downloads.resume_interrupted(catalog.catalog_for_panel())
        if resumed:
            print(f"resuming interrupted downloads: {', '.join(resumed)}", flush=True)
    except Exception as e:
        print(f"could not check for interrupted downloads: {e}", flush=True)
    yield


app = FastAPI(title="lllm3090", version=__version__, lifespan=lifespan)
_busy = asyncio.Lock()
_last: dict = {"action": None, "ok": None, "detail": ""}


def clean(line: str) -> str:
    """Strip ANSI colour and collapse a progress-bar redraw to its last frame."""
    line = ANSI.sub("", line)
    if "\r" in line:
        line = line.split("\r")[-1]
    return line.rstrip("\n")


def vram() -> dict | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
        used, total = (int(x.strip()) for x in out.split(","))
        return {"used_mb": used, "total_mb": total}
    except Exception:
        return None


@app.get("/api/status")
def status():
    profile = hardware.detect()
    return {
        "version": __version__,
        "card": {
            "name": profile.name,
            "vram_gb": round(profile.vram_mib / 1024),
            "measured": profile.measured,
            "reference": hardware.reference().name,
        },
        "engine": engine.status(),
        "vram": vram(),
        "busy": _busy.locked(),
        "last": _last,
        "installed": catalog.installed(),
        "catalog": catalog.catalog_for_panel(),
        "downloads": downloads.all_downloads(),
        "models_dir": str(config.MODELS_DIR),
    }


@app.post("/api/start")
async def start(model: str, ctx: int | None = None, parallel: int | None = None):
    entry = next((m for m in catalog.installed() if m["name"] == model), None)
    if entry is None:
        return JSONResponse({"error": f"{model!r} is not installed"}, status_code=400)
    if _busy.locked():
        return JSONResponse({"error": "busy"}, status_code=409)

    # Size the pool from what actually fits rather than from a stored default,
    # and leave room for the concurrent conversations an agent needs.
    parallel = parallel or config.DEFAULT_PARALLEL
    known = next((m for m in catalog.load_catalog() if m.name == model), None)
    if ctx is None:
        if known is not None:
            p = catalog.plan(known, parallel)
            ctx, parallel = p.pool, p.parallel
        else:
            # An unknown GGUF: no KV figure to plan with, so be conservative.
            ctx = 32768 * parallel

    async with _busy:
        await asyncio.to_thread(engine.stop)
        # wait=0: launch and return. A load takes minutes; blocking here would
        # freeze the panel and hang systemd's stop until it SIGKILLs us.
        ok, detail = await asyncio.to_thread(
            engine.start, entry["path"], model, ctx, parallel, 0,
            known.chat_template if known else None,
        )
        _last.update(action=f"start {model}", ok=ok, detail=detail[-400:])
    return {"ok": ok, "detail": _last["detail"]}


@app.post("/api/stop")
async def stop():
    if _busy.locked():
        return JSONResponse({"error": "busy"}, status_code=409)
    async with _busy:
        ok, detail = await asyncio.to_thread(engine.stop)
        _last.update(action="stop", ok=ok, detail=detail)
    return {"ok": ok, "detail": detail}


@app.post("/api/download/{model_id}")
def download(model_id: str):
    entry = next((m for m in catalog.catalog_for_panel() if m["id"] == model_id), None)
    if entry is None:
        return JSONResponse({"error": f"unknown model {model_id!r}"}, status_code=404)
    if not entry["fits"]:
        return JSONResponse(
            {"error": f"{entry['name']} does not fit this card"}, status_code=400
        )
    return downloads.start(entry).as_dict()


@app.post("/api/download/{model_id}/cancel")
def cancel_download(model_id: str):
    return {"cancelled": downloads.cancel(model_id)}


@app.get("/api/logs")
def logs(tail: int = 200):
    log = config.ENGINE_LOG
    if not log.exists():
        return {"lines": []}
    with log.open("r", errors="replace") as f:
        buf = f.readlines()[-tail:]
    return {"lines": [clean(line) for line in buf if clean(line).strip()]}


async def _tail_stream():
    """Follow the engine log across truncation and replacement.

    Line-boundary safe: the initial seek is advanced past the partial line it
    lands inside, and an incomplete trailing write is held back until its
    newline arrives -- otherwise a fragment is emitted as though it were a line.
    """
    log = config.ENGINE_LOG
    pos, ino, pending = 0, None, ""
    if log.exists():
        st = log.stat()
        ino = st.st_ino
        start_at = max(0, st.st_size - 4000)
        with log.open("r", errors="replace") as f:
            f.seek(start_at)
            if start_at:
                f.readline()
            pos = f.tell()
    yield "retry: 3000\n\n"
    while True:
        try:
            if not log.exists():
                await asyncio.sleep(1)
                continue
            st = log.stat()
            if ino is None or st.st_ino != ino or st.st_size < pos:
                ino, pos, pending = st.st_ino, 0, ""
                yield "event: rotated\ndata: --- engine restarted ---\n\n"
            if st.st_size > pos:
                with log.open("r", errors="replace") as f:
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
    return StreamingResponse(
        _tail_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return resources.files("lllm3090.static").joinpath("index.html").read_text()
