"""What `engine.start` refuses, and why, before it launches anything.

Every check here happens before the `Popen`, which is the point: the panel
starts the engine with `wait=0`, so anything not caught up front is reported
to the user as "starting" and only contradicted a second later in a log
nobody is reading.
"""

from __future__ import annotations

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
