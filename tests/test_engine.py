"""The engine's lifecycle, with no engine.

`engine.py` spawns a process, polls it for minutes, kills it, and reads a
pidfile and an HTTP endpoint to say what is running. None of that could be
exercised without a GPU, so all of it went untested -- which is backwards: the
paths that only run when something has gone wrong are the ones nobody watches
by hand.

Everything here is the failure side of that: a missing binary, a file that is
not a model, a pidfile pointing at nothing, an engine that has already gone.
The one thing deliberately not faked is the socket in `healthy` -- a closed
port is a closed port, and asking a real one costs a millisecond.
"""

from __future__ import annotations

import signal

import pytest

from lllm3090 import config, engine


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """State that belongs to this test, not to the machine running it."""
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "ENGINE_LOG", tmp_path / "state" / "engine.log")
    monkeypatch.setattr(config, "ENGINE_PID", tmp_path / "state" / "engine.pid")
    (tmp_path / "state").mkdir()
    return tmp_path


@pytest.fixture
def installed_engine(state_dir, monkeypatch):
    """A llama-server that exists and a Popen that records instead of running."""
    llama = state_dir / "llama.cpp"
    llama.mkdir()
    (llama / "llama-server").write_text("#!/bin/sh\n")
    monkeypatch.setattr(config, "LLAMA_DIR", llama)

    launched: list[list[str]] = []

    class FakeProc:
        pid = 4242

        def poll(self):
            return None

    monkeypatch.setattr(
        engine.subprocess, "Popen",
        lambda argv, **kw: (launched.append(argv), FakeProc())[1],
    )
    return launched


# --- start: the refusals that happen before anything is launched -------------


def test_no_engine_installed_is_a_refusal_not_a_crash(state_dir, monkeypatch):
    monkeypatch.setattr(config, "LLAMA_DIR", state_dir / "nothing-here")
    ok, detail = engine.start(str(state_dir / "m.gguf"), "M", 4096, 1, 0)
    assert not ok
    assert "install-engine" in detail, "the message has to say how to fix it"


def test_a_missing_model_is_refused(installed_engine, state_dir):
    ok, detail = engine.start(str(state_dir / "gone.gguf"), "M", 4096, 1, 0)
    assert not ok
    assert "model file not found" in detail
    assert not installed_engine


def test_a_file_that_is_not_a_gguf_is_refused(installed_engine, state_dir):
    """A safetensors download, or an HTML error page saved under a .gguf name."""
    impostor = state_dir / "model.gguf"
    impostor.write_bytes(b"<!DOCTYPE html>\n")
    ok, detail = engine.start(str(impostor), "M", 4096, 1, 0)
    assert not ok
    assert "not a GGUF" in detail
    assert not installed_engine, "nothing should be launched, and no pidfile written"
    assert not config.ENGINE_PID.exists()


def test_a_truncated_file_is_not_mistaken_for_a_gguf(state_dir):
    """A part file renamed by hand: shorter than the magic it should carry."""
    stub = state_dir / "half.gguf"
    stub.write_bytes(b"GG")
    assert engine.not_a_gguf(stub)


def test_a_projector_that_is_not_a_gguf_is_refused(installed_engine, state_dir):
    model = state_dir / "model.gguf"
    model.write_bytes(b"GGUF")
    proj = state_dir / "mmproj.gguf"
    proj.write_bytes(b"not this either")
    ok, detail = engine.start(str(model), "M", 4096, 1, 0, None, str(proj))
    assert not ok
    assert "not a GGUF" in detail
    assert not installed_engine


def test_a_real_gguf_gets_launched(installed_engine, state_dir):
    """The magic check must not stand in the way of the ordinary case."""
    model = state_dir / "model.gguf"
    model.write_bytes(b"GGUF" + b"\x00" * 64)
    ok, detail = engine.start(str(model), "M", 8192, 2, 0)
    assert ok
    assert "4096 tokens x 2 slots" in detail
    assert config.ENGINE_PID.read_text() == "4242"


# --- stop: every way there is nothing to stop --------------------------------


def test_stopping_with_no_pidfile_is_success(state_dir):
    """`lllm3090 stop` on a stopped machine is a no-op, not an error."""
    ok, detail = engine.stop()
    assert ok
    assert "no engine running" in detail


def test_a_stale_pidfile_is_cleaned_up(state_dir, monkeypatch):
    """The engine died with the machine; its pidfile did not."""
    config.ENGINE_PID.write_text("999999")
    monkeypatch.setattr(engine, "alive", lambda target: False)
    ok, detail = engine.stop()
    assert ok
    assert "no engine running" in detail
    assert not config.ENGINE_PID.exists()


def test_a_process_that_exits_between_check_and_signal_is_not_an_error(
    state_dir, monkeypatch
):
    """The race `stop` is written to lose gracefully."""
    config.ENGINE_PID.write_text("4242")
    monkeypatch.setattr(engine, "alive", lambda target: True)

    def gone(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(engine.os, "kill", gone)
    ok, detail = engine.stop()
    assert ok
    assert "already gone" in detail
    assert not config.ENGINE_PID.exists()


def test_a_live_engine_is_signalled_and_the_pidfile_removed(state_dir, monkeypatch):
    config.ENGINE_PID.write_text("4242")
    signals: list[int] = []
    # Alive for the pidfile read, gone by the first poll: SIGTERM worked.
    answers = iter([True, False])
    monkeypatch.setattr(engine, "alive", lambda target: next(answers, False))
    monkeypatch.setattr(engine.os, "kill", lambda pid, sig: signals.append(sig))
    ok, detail = engine.stop()
    assert ok
    assert signals == [signal.SIGTERM], "SIGKILL is for an engine that ignored it"
    assert "stopped engine (pid 4242)" in detail
    assert not config.ENGINE_PID.exists()


def test_an_engine_that_ignores_sigterm_is_killed(state_dir, monkeypatch):
    config.ENGINE_PID.write_text("4242")
    signals: list[int] = []
    monkeypatch.setattr(engine, "alive", lambda target: True)
    monkeypatch.setattr(engine.os, "kill", lambda pid, sig: signals.append(sig))
    monkeypatch.setattr(engine.time, "sleep", lambda _: None)
    ok, detail = engine.stop(timeout=1)
    assert ok
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert "killed engine" in detail
    assert not config.ENGINE_PID.exists(), "the VRAM is back either way"


# --- what the front ends read ------------------------------------------------


def test_status_with_nothing_running_asks_the_engine_nothing(state_dir, monkeypatch):
    """`answering` and `model` are short-circuited: there is nobody to ask."""
    asked: list[str] = []
    monkeypatch.setattr(engine, "_get", lambda url, timeout=3.0: asked.append(url))
    s = engine.status()
    assert s == {
        "running": False, "pid": None, "port": config.ENGINE_PORT,
        "answering": False, "model": None,
    }
    assert not asked


def test_status_of_a_loading_engine_says_running_but_not_answering(
    state_dir, monkeypatch
):
    """The gap this whole module exists for: the pidfile is there, the weights
    are not up yet, and reporting that as ready is what makes a slow load look
    like a failure."""
    config.ENGINE_PID.write_text("4242")
    monkeypatch.setattr(engine, "alive", lambda target: True)
    monkeypatch.setattr(engine, "_get", lambda url, timeout=3.0: None)
    s = engine.status()
    assert s["running"] and s["pid"] == 4242
    assert not s["answering"]
    assert s["model"] is None


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"data": [{"id": "Qwen3-8B"}]}, "Qwen3-8B"),          # OpenAI shape
        ({"models": [{"name": "Qwen3-8B"}]}, "Qwen3-8B"),      # llama.cpp shape
        ({"data": [{"name": "Qwen3-8B"}]}, "Qwen3-8B"),        # id absent
        ({"data": []}, None),                                  # up, nothing loaded
        ({}, None),
        (None, None),                                          # not answering
    ],
)
def test_served_model_reads_both_response_shapes(monkeypatch, payload, expected):
    monkeypatch.setattr(engine, "_get", lambda url, timeout=3.0: payload)
    assert engine.served_model() == expected


def test_healthy_is_false_when_nothing_is_listening(monkeypatch):
    """A real socket to a closed port: refused, caught, reported as unhealthy."""
    monkeypatch.setattr(config, "ENGINE_URL", "http://127.0.0.1:1")
    assert engine.healthy() is False
