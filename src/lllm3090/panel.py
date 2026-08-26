"""The control panel: a loopback-only web UI for starting, stopping and
downloading models.

Binds 127.0.0.1 by design. These endpoints start processes and write to disk
with no authentication, so reach it remotely with an SSH tunnel
(``ssh -L 8080:127.0.0.1:8080 host``) rather than by binding a LAN address.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from importlib import resources

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import catalog, config, downloads, engine, state
from ._version import __version__


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


@app.get("/api/status")
def status():
    """The machine, plus the two things only this process can answer for."""
    return {**state.snapshot(), "busy": _busy.locked(), "last": _last}


@app.post("/api/start")
async def start(model: str, ctx: int | None = None, parallel: int | None = None):
    entry = next((m for m in catalog.installed() if m["name"] == model), None)
    if entry is None:
        return JSONResponse({"error": f"{model!r} is not installed"}, status_code=400)
    if parallel is not None and parallel < 1:
        # A query string is not a form the page controls, so this is checked
        # here rather than left to the ValueError catalog.launch_plan raises.
        return JSONResponse(
            {"error": "parallel must be at least 1"}, status_code=400
        )
    if _busy.locked():
        return JSONResponse({"error": "busy"}, status_code=409)

    # Size the pool from what actually fits rather than from a stored default,
    # and leave room for the concurrent conversations an agent needs.
    known = next((m for m in catalog.load_catalog() if m.name == model), None)
    if ctx is None:
        p = catalog.launch_plan(model, parallel)
        ctx, parallel = p.pool, p.parallel
    else:
        parallel = parallel or config.DEFAULT_PARALLEL

    async with _busy:
        await asyncio.to_thread(engine.stop)
        # Measured with the outgoing engine already gone, so its VRAM is not
        # charged against its replacement. The panel is as able as the console
        # to start a plan the card cannot serve, so it makes the same check.
        warning = await asyncio.to_thread(catalog.free_vram_warning, known, ctx)
        # wait=0: launch and return. A load takes minutes; blocking here would
        # freeze the panel and hang systemd's stop until it SIGKILLs us.
        ok, detail = await asyncio.to_thread(
            engine.start, entry["path"], model, ctx, parallel, 0,
            known.chat_template if known else None, entry.get("mmproj"),
        )
        if warning:
            # Ahead of the detail rather than after it: the engine reports a
            # successful launch either way, and that is the line this warning
            # exists to qualify.
            detail = f"{warning}\n{detail}"
        _last.update(action=f"start {model}", ok=ok, detail=detail[-400:])
    return {"ok": ok, "detail": _last["detail"], "warning": warning}


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
    return {"lines": engine.tail(tail)}


@app.post("/api/logs/clear")
def clear_logs():
    """Empty the engine log, leaving the engine writing to it alone.

    Truncated rather than rotated or deleted. The engine holds this file open
    for the life of the process, so a rename would send its output to a file
    nothing is reading, and an unlink would send it nowhere at all.
    :func:`lllm3090.engine.start` opens the log ``O_APPEND`` for the same
    reason -- the next line the engine writes lands at the beginning of the
    emptied file rather than at the offset it had reached.

    The tailer notices the file has shrunk and emits ``rotated``, which is
    already what tells the browser to clear the pane it is showing.
    """
    if config.ENGINE_LOG.exists():
        os.truncate(config.ENGINE_LOG, 0)
    return {"cleared": True}


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
        # newline="": universal newlines would turn a progress redraw's \r into
        # \n, splitting one line into a frame per update. engine.clean collapses
        # the redraw, but only while the frames are still on the same line, and
        # engine.tail reads the same file the same way.
        with log.open("r", errors="replace", newline="") as f:
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
                with log.open("r", errors="replace", newline="") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
                data = pending + chunk
                # Split on "\n" alone: splitlines() breaks on \r as well, which
                # is exactly the character this stream has to keep hold of.
                parts = data.split("\n")
                lines, pending = parts[:-1], parts[-1]
                for raw in lines:
                    line = engine.clean(raw)
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
