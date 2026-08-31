"""What the engine is actually launched with.

`engine.start` is otherwise untested, and the vision flag is invisible from
every other angle: a missing `--mmproj` produces an engine that starts happily
and simply cannot see.
"""

from __future__ import annotations

import inspect

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
        # `--help` is engine.supports() probing the binary's capabilities, not
        # a launch. Recording it would put it at seen[0] and shift every
        # assertion in this file by one.
        if "--help" not in argv:
            seen.append(argv)
        return FakeProc()

    monkeypatch.setattr(engine.subprocess, "Popen", fake_popen)
    # A stub shell script cannot answer --help, so the probe would say "no" and
    # suppress every optional flag. Tests that care about the probe set this
    # themselves; the rest get a binary that understands everything.
    monkeypatch.setattr(engine, "supports", lambda flag: True)
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


def test_mtp_is_enabled_from_the_file_not_the_catalogue(tmp_path, monkeypatch):
    """The flag is added only for a checkpoint that actually carries the head.

    llama.cpp refuses to start with ``--spec-type draft-mtp`` against a
    checkpoint without one, so a catalogue field claiming MTP would turn a
    working start into a failed one the moment a repo shipped a build with the
    head stripped. The file on disk is the only thing that knows.
    """
    from lllm3090 import engine, gguf

    plain = tmp_path / "plain.gguf"
    plain.write_bytes(b"GGUF" + b"\0" * 64)          # unparseable past the magic
    assert gguf.has_mtp(plain) is False

    missing = tmp_path / "not-here.gguf"
    assert gguf.has_mtp(missing) is False, "an unreadable file must not add a flag"

    argv = inspect.getsource(engine.start)
    assert "gguf.has_mtp(model_path)" in argv, (
        "the flag must be decided from the checkpoint, not from a catalogue field"
    )
