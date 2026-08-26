"""What the engine is actually launched with.

`engine.start` is otherwise untested, and the vision flag is invisible from
every other angle: a missing `--mmproj` produces an engine that starts happily
and simply cannot see.
"""

from __future__ import annotations

import pytest

from lllm3090 import config, engine


@pytest.fixture
def fake_engine(tmp_path, monkeypatch):
    """A binary and a model that exist, and a Popen that records instead of running."""
    llama = tmp_path / "llama.cpp"
    llama.mkdir()
    (llama / "llama-server").write_text("#!/bin/sh\n")
    monkeypatch.setattr(config, "LLAMA_DIR", llama)
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "ENGINE_LOG", tmp_path / "state" / "engine.log")
    monkeypatch.setattr(config, "ENGINE_PID", tmp_path / "state" / "engine.pid")

    seen: list[list[str]] = []

    class FakeProc:
        pid = 4242

        def poll(self):
            return None

    def fake_popen(argv, **kw):
        seen.append(argv)
        return FakeProc()

    monkeypatch.setattr(engine.subprocess, "Popen", fake_popen)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    return seen, model, tmp_path


def test_a_projector_is_passed_to_the_engine(fake_engine):
    seen, model, tmp = fake_engine
    proj = tmp / "mmproj-F16.gguf"
    proj.write_bytes(b"GGUF")
    ok, _ = engine.start(str(model), "M", 4096, 2, 0, None, str(proj))
    assert ok
    argv = seen[0]
    assert "--mmproj" in argv
    assert argv[argv.index("--mmproj") + 1] == str(proj)


def test_no_projector_means_no_flag(fake_engine):
    """A text model must not be handed an empty --mmproj."""
    seen, model, _ = fake_engine
    engine.start(str(model), "M", 4096, 2, 0, None, None)
    assert "--mmproj" not in seen[0]


def test_a_missing_projector_is_refused_before_launch(fake_engine):
    """Better a clear message than an engine that starts and cannot see."""
    seen, model, tmp = fake_engine
    ok, detail = engine.start(str(model), "M", 4096, 2, 0, None, str(tmp / "nope.gguf"))
    assert not ok
    assert "projector not found" in detail
    assert not seen, "must not launch the engine at all"
