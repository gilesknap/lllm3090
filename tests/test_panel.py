"""The panel's HTTP surface.

The only API this project has, and it was the only module with no tests at
all. Nothing here needs a GPU or an engine: `engine.status`, `engine.start`
and `engine.stop` are the boundary, and everything on this side of it -- the
catalogue read, the refusals, the busy gate, the log tail -- is what these
tests hold still.

The busy gate is faked rather than acquired. `_busy` lives on the app's event
loop and the test client drives that loop from another thread, so a lock taken
here would not be the lock the handler sees; what matters is the branch, and
`locked()` is what selects it.
"""

from __future__ import annotations

import asyncio
import struct
from typing import Self

import pytest
from fastapi.testclient import TestClient

from lllm3090 import catalog, config, downloads, engine, panel, state


class Busy:
    """Stands in for `_busy` when the test is about the gate, not the work."""

    def __init__(self, locked: bool):
        self._locked = locked

    def locked(self) -> bool:
        return self._locked

    async def __aenter__(self) -> Self:
        self._locked = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._locked = False


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A panel whose machine is this tmp_path: no models, no engine, no card."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "ENGINE_LOG", tmp_path / "engine.log")
    monkeypatch.setattr(config, "ENGINE_PID", tmp_path / "engine.pid")
    monkeypatch.setattr(engine, "status", lambda: {
        "running": False, "pid": None, "port": config.ENGINE_PORT,
        "answering": False, "model": None,
    })
    monkeypatch.setattr(panel, "_busy", Busy(False))
    return TestClient(panel.app)


# --- status ------------------------------------------------------------------


def test_status_describes_the_machine_with_nothing_on_it(client, tmp_path):
    """The shape both front ends read. It must be answerable on a bare box."""
    s = client.get("/api/status").json()
    assert s["version"]
    assert s["engine"]["running"] is False
    assert s["installed"] == []
    assert s["downloads"] == []
    assert s["catalog"], "the catalogue is bundled; it does not depend on the disk"
    assert s["busy"] is False
    assert s["endpoint"] == config.ENGINE_URL
    assert set(s["card"]) == {
        "name", "vram_gb", "measured", "present", "desktop", "reference"
    }
    # Free space is answered even though the models directory does not exist
    # yet: on a fresh install it is the first download that creates it.
    assert not (tmp_path / "models").exists()
    assert s["disk"]["free_gb"] > 0
    assert s["disk"]["total_gb"] >= s["disk"]["free_gb"]


def test_a_disk_that_cannot_be_measured_is_reported_as_unknown(client, monkeypatch):
    """None rather than zero: a machine that will not say is not a full disk."""
    def refuse(_path):
        raise OSError("no such device")

    monkeypatch.setattr(state.shutil, "disk_usage", refuse)
    assert client.get("/api/status").json()["disk"] is None


def test_a_gguf_the_catalogue_has_never_seen_is_still_offered(client, tmp_path):
    """A hand-placed checkpoint is startable, so it has to carry a window."""
    d = tmp_path / "models" / "MyOwn-7B"
    d.mkdir(parents=True)
    (d / "myown.gguf").write_bytes(b"GGUF" + b"\x00" * 32)
    row = next(m for m in client.get("/api/status").json()["installed"]
               if m["name"] == "MyOwn-7B")
    assert row["kind"] == "gguf"
    assert row["max_ctx"] > 0 and row["parallel"] >= 1


# --- start and stop ----------------------------------------------------------


def test_starting_something_that_is_not_installed_is_a_400(client):
    r = client.post("/api/start", params={"model": "Nothing-70B"})
    assert r.status_code == 400
    assert "not installed" in r.json()["error"]


def test_a_nonsense_slot_count_is_refused_before_anything_stops(
    client, tmp_path, monkeypatch
):
    """The query string is not a form the page controls."""
    monkeypatch.setattr(catalog, "installed", lambda *a, **k: [
        {"name": "M", "path": str(tmp_path / "m.gguf"), "mmproj": None, "gb": 1.0},
    ])
    stopped = []
    monkeypatch.setattr(engine, "stop", lambda *a, **k: stopped.append(1))
    r = client.post("/api/start", params={"model": "M", "parallel": 0})
    assert r.status_code == 400
    assert "at least 1" in r.json()["error"]
    assert not stopped, "a bad request must not take down the running engine"


def test_a_start_while_busy_is_a_409(client, tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "installed", lambda *a, **k: [
        {"name": "M", "path": str(tmp_path / "m.gguf"), "mmproj": None, "gb": 1.0},
    ])
    monkeypatch.setattr(panel, "_busy", Busy(True))
    assert client.post("/api/start", params={"model": "M"}).status_code == 409


def test_a_stop_while_busy_is_a_409(client, monkeypatch):
    monkeypatch.setattr(panel, "_busy", Busy(True))
    assert client.post("/api/stop").status_code == 409


def test_a_start_stops_the_old_engine_first_and_reports_the_new_one(
    client, tmp_path, monkeypatch
):
    """Order matters: the outgoing engine's VRAM must be back before the
    free-VRAM check measures the card for its replacement."""
    order: list[str] = []
    monkeypatch.setattr(catalog, "installed", lambda *a, **k: [
        {"name": "M", "path": str(tmp_path / "m.gguf"), "mmproj": None, "gb": 1.0},
    ])
    monkeypatch.setattr(engine, "stop",
                        lambda *a, **k: (order.append("stop"), (True, ""))[1])
    monkeypatch.setattr(catalog, "free_vram_warning",
                        lambda *a, **k: (order.append("measure"), None)[1])
    monkeypatch.setattr(engine, "start", lambda *a, **k: (
        order.append("start"), (True, "starting M")
    )[1])
    body = client.post("/api/start", params={"model": "M", "ctx": 4096}).json()
    assert order == ["stop", "measure", "start"]
    assert body["ok"] is True
    assert body["detail"] == "starting M"
    assert body["warning"] is None


def test_a_vram_warning_is_put_ahead_of_the_engine_s_own_answer(
    client, tmp_path, monkeypatch
):
    """The engine reports a successful launch either way; the warning is what
    qualifies it, so it cannot be appended below and scrolled off."""
    monkeypatch.setattr(catalog, "installed", lambda *a, **k: [
        {"name": "M", "path": str(tmp_path / "m.gguf"), "mmproj": None, "gb": 1.0},
    ])
    monkeypatch.setattr(engine, "stop", lambda *a, **k: (True, ""))
    monkeypatch.setattr(catalog, "free_vram_warning", lambda *a, **k: "Warning: tight")
    monkeypatch.setattr(engine, "start", lambda *a, **k: (True, "starting M"))
    body = client.post("/api/start", params={"model": "M", "ctx": 4096}).json()
    assert body["warning"] == "Warning: tight"
    assert body["detail"].startswith("Warning: tight\n")
    assert client.get("/api/status").json()["last"]["action"] == "start M"


def test_a_stop_records_what_it_did(client, monkeypatch):
    monkeypatch.setattr(engine, "stop", lambda *a, **k: (True, "stopped engine"))
    assert client.post("/api/stop").json() == {"ok": True, "detail": "stopped engine"}
    assert client.get("/api/status").json()["last"]["action"] == "stop"


# --- downloads ---------------------------------------------------------------


def test_downloading_a_model_the_catalogue_does_not_have_is_a_404(client):
    r = client.post("/api/download/no-such-model")
    assert r.status_code == 404
    assert "unknown model" in r.json()["error"]


def test_a_model_that_does_not_fit_this_card_is_refused_before_the_bytes(
    client, monkeypatch
):
    """21 GB is a long way to get before finding out it was never going to run."""
    entry = {"id": "huge", "name": "Huge-400B", "fits": False,
             "repo": "r", "file": "f.gguf"}
    monkeypatch.setattr(catalog, "catalog_for_panel", lambda *a, **k: [entry])
    started = []
    monkeypatch.setattr(downloads, "start", lambda e: started.append(e))
    r = client.post("/api/download/huge")
    assert r.status_code == 400
    assert "does not fit" in r.json()["error"]
    assert not started


def test_a_download_that_fits_is_started_and_reported(client, monkeypatch):
    """The response is the same row the panel already draws for a live
    download, so the button does not have to wait for the next poll to
    show anything."""
    entry = {"id": "ok", "name": "Fits-8B", "fits": True,
             "repo": "r", "file": "f.gguf"}
    monkeypatch.setattr(catalog, "catalog_for_panel", lambda *a, **k: [entry])
    # The thread is real; what it would fetch is not. This is the panel's
    # side of the boundary -- downloads.py has its own tests over a socket.
    monkeypatch.setattr(downloads, "_run", lambda dl: None)
    monkeypatch.setattr(downloads, "_downloads", {})
    body = client.post("/api/download/ok").json()
    assert body["id"] == "ok"
    assert body["name"] == "Fits-8B"
    assert set(body) >= {"state", "percent", "done_gb", "total_gb", "rate_mib_s"}


def test_cancelling_a_download_that_is_not_running_says_so(client):
    """Not a 404: cancel is what the button sends, and a download that has
    already finished or was never started is not an error to have asked."""
    assert client.post("/api/download/no-such-model/cancel").json() == {
        "cancelled": False
    }


# --- logs --------------------------------------------------------------------


def test_logs_with_no_log_file_are_empty_not_an_error(client):
    assert client.get("/api/logs").json() == {"lines": []}


def test_logs_are_cleaned_and_counted_from_the_end(client, tmp_path):
    config.ENGINE_LOG.write_text(
        "\x1b[32mfirst\x1b[0m\nsecond\n\nload: 10%\rload: 100%\nlast\n"
    )
    lines = client.get("/api/logs", params={"tail": 3}).json()["lines"]
    assert lines == ["second", "load: 100%", "last"]


def test_asking_for_no_log_lines_returns_none_of_them(client):
    """`readable[-0:]` is the whole log, which is the opposite of the request."""
    config.ENGINE_LOG.write_text("a\nb\n")
    assert client.get("/api/logs", params={"tail": 0}).json() == {"lines": []}


def test_clearing_the_log_empties_it_without_removing_it(client):
    """Truncated rather than unlinked: the engine has this file open, and a
    file it is writing into that nothing can read is worse than a full one."""
    config.ENGINE_LOG.write_text("a great deal of engine output\n" * 50)
    assert client.post("/api/logs/clear").json() == {"cleared": True}
    assert config.ENGINE_LOG.exists()
    assert config.ENGINE_LOG.read_text() == ""
    assert client.get("/api/logs").json() == {"lines": []}


def test_clearing_a_log_that_is_not_there_is_not_an_error(client):
    assert client.post("/api/logs/clear").json() == {"cleared": True}


def test_a_cleared_log_reaches_the_browser_as_a_rotation(client, monkeypatch):
    """Which is what the page already listens for to empty the pane it shows."""
    config.ENGINE_LOG.write_text("old output\n")
    monkeypatch.setattr(panel.asyncio, "sleep", _no_sleep)

    async def collect() -> list[str]:
        gen = panel._tail_stream()
        await gen.asend(None)
        await gen.asend(None)                      # catch up to the end
        client.post("/api/logs/clear")
        seen = [await gen.asend(None)]
        await gen.aclose()
        return seen

    assert "event: rotated" in asyncio.run(collect())[0]


def test_the_log_stream_holds_back_a_line_that_has_no_newline_yet(
    client, tmp_path, monkeypatch
):
    """A partial write is a fragment, not a line. Emitting it produces two
    half-lines in the browser where the file has one."""
    config.ENGINE_LOG.write_text("")
    monkeypatch.setattr(panel.asyncio, "sleep", _no_sleep)

    async def collect() -> list[str]:
        gen = panel._tail_stream()
        await gen.asend(None)                      # the retry: preamble
        config.ENGINE_LOG.write_text("whole line\npartial")
        seen = [await gen.asend(None) for _ in range(2)]
        config.ENGINE_LOG.write_text("whole line\npartial ends here\n")
        seen.append(await gen.asend(None))
        await gen.aclose()
        return seen

    events = asyncio.run(collect())
    assert 'data: "whole line"' in events[0]
    assert "partial" not in events[0]
    assert any("partial ends here" in e for e in events)


def test_the_log_stream_announces_a_restart_rather_than_replaying_the_file(
    client, monkeypatch
):
    """`lllm3090 start` truncates the log. Without this the browser would show
    the new engine's output appended to the old engine's, as one run."""
    config.ENGINE_LOG.write_text("old engine\n")
    monkeypatch.setattr(panel.asyncio, "sleep", _no_sleep)

    async def collect() -> list[str]:
        gen = panel._tail_stream()
        await gen.asend(None)                      # the retry: preamble
        await gen.asend(None)                      # catch up to the end
        config.ENGINE_LOG.write_text("new\n")     # a restart truncates it
        seen = [await gen.asend(None), await gen.asend(None)]
        await gen.aclose()
        return seen

    events = asyncio.run(collect())
    assert any("event: rotated" in e for e in events)
    assert any('data: "new"' in e for e in events)


async def _no_sleep(_seconds: float) -> None:
    """The tailer polls at 0.7s; the test has no reason to."""


# --- the page itself and the lifespan ----------------------------------------


def test_the_root_serves_the_panel_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "/api/logstream" in r.text, "the page must be the one wired to this API"


def test_a_broken_resume_check_does_not_stop_the_panel_coming_up(
    tmp_path, monkeypatch
):
    """The panel is how you fix a broken download. It has to start first."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")

    def explode(_entries):
        raise OSError("models dir is on a disk that is not there")

    monkeypatch.setattr(downloads, "resume_interrupted", explode)
    with TestClient(panel.app) as started:
        assert started.get("/api/logs").status_code == 200


# --- choosing the backend ----------------------------------------------------


@pytest.fixture
def engine_choice(client, tmp_path, monkeypatch):
    """The panel, plus the two engines this machine actually has."""
    from lllm3090 import engines
    managed = tmp_path / "llama.cpp"
    cuda = tmp_path / "engines" / f"{engines.LLAMA_BUILD}-cuda-sm86"
    for d in (managed, cuda):
        d.mkdir(parents=True)
        (d / "llama-server").write_text("#!/bin/sh\n")
    monkeypatch.setattr(config, "LLAMA_DIR", managed)
    monkeypatch.setattr(config, "ENGINES_DIR", tmp_path / "engines")
    monkeypatch.setattr(config, "ENGINE_CHOICE", tmp_path / "engine.json")
    monkeypatch.setattr(config, "LLAMA_DIR_FROM_ENV", False)
    monkeypatch.setattr(
        engines, "backend",
        lambda directory=None: "cuda" if "cuda" in str(
            directory or engines.active_dir()) else "vulkan",
    )
    return client, cuda


def test_the_panel_can_switch_backend(engine_choice):
    client, cuda = engine_choice
    r = client.post(f"/api/engine/select?engine_id={cuda.name}")
    assert r.status_code == 200, r.text
    assert r.json()["backend"] == "cuda"
    assert client.get("/api/status").json()["engines"]["active"] == cuda.name


def test_switching_does_not_restart_what_is_running(engine_choice, monkeypatch):
    """The two are separate decisions, and the second costs minutes of load
    time and the VRAM of whatever is answering right now."""
    client, cuda = engine_choice
    monkeypatch.setattr(engine, "status", lambda: {
        "running": True, "pid": 4242, "port": config.ENGINE_PORT,
        "answering": True, "model": "Qwen3.8-27B",
    })
    stopped: list[bool] = []
    monkeypatch.setattr(engine, "stop", lambda: stopped.append(True))
    body = client.post(f"/api/engine/select?engine_id={cuda.name}").json()
    assert not stopped, "a switch must not take down a working engine"
    assert body["restart_needed"] is True
    assert "restart" in body["detail"]


def test_an_environment_override_refuses_rather_than_doing_nothing(
    engine_choice, monkeypatch
):
    """The page renders the control disabled; a request arriving anyway is a
    stale tab, and silently succeeding would teach the wrong thing."""
    client, cuda = engine_choice
    monkeypatch.setattr(config, "LLAMA_DIR_FROM_ENV", True)
    r = client.post(f"/api/engine/select?engine_id={cuda.name}")
    assert r.status_code == 409
    assert "LLLM3090_LLAMA_DIR" in r.json()["error"]


def test_an_unknown_engine_is_refused(engine_choice):
    client, _ = engine_choice
    assert client.post("/api/engine/select?engine_id=%2Fbin%2Fsh").status_code == 400


def test_status_says_which_profiles_each_backend_allows(engine_choice):
    """The switch changes more than a speed: `copy` is measured to win on CUDA
    and to lose on Vulkan, so the page can say what picking one buys."""
    client, _ = engine_choice
    options = client.get("/api/status").json()["engines"]["options"]
    by_backend = {o["backend"]: o["profiles"] for o in options}
    assert by_backend["vulkan"] == ["default"]
    assert "copy" in by_backend["cuda"]


# --- what the page is told about speculation and cache ------------------------


def test_an_uncatalogued_checkpoint_says_whether_it_speculates(client, tmp_path):
    """The catalogue's rows answer this from models.yaml where the file is
    absent; a hand-placed checkpoint has no declaration, so the file is the
    only source there is -- and the panel draws one merged list, so a row that
    left the badge off would be saying "no head" rather than "no answer"."""
    d = tmp_path / "models" / "Homemade-3B"
    d.mkdir(parents=True)
    name = b"blk.40.nextn.enorm.weight"
    (d / "m.gguf").write_bytes(
        b"GGUF" + struct.pack("<I", 3)      # magic and format version
        + struct.pack("<QQ", 1, 0)          # one tensor, no metadata
        + struct.pack("<Q", len(name)) + name
        + struct.pack("<I", 1) + struct.pack("<Q", 8)   # one dimension, of 8
        + struct.pack("<I", 0) + struct.pack("<Q", 0)   # ggml type, offset
    )
    row = next(m for m in client.get("/api/status").json()["installed"]
               if m["name"] == "Homemade-3B")
    assert row["mtp"] is True
    assert row["mtp_declared"] is None, "there is no catalogue entry to declare it"


def test_the_page_is_told_what_every_model_is_started_with(client):
    """Flash attention, the main KV cache and the draft model's own cache are
    identical for every entry, so they belong beside the engine rather than
    repeated on nine rows."""
    fixed = client.get("/api/status").json()["engines"]["fixed"]
    assert fixed["flash_attention"] is True
    assert fixed["cache_type"] == engine.CACHE_TYPE
    # No engine is installed under the test fixture, so the probe cannot say
    # the flag is there -- and reporting q8_0 anyway would describe memory the
    # card is not going to give back.
    assert fixed["draft_cache_type"] is None
