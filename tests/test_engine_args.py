"""What the engine is actually launched with.

`engine.start` is otherwise untested, and the vision flag is invisible from
every other angle: a missing `--mmproj` produces an engine that starts happily
and simply cannot see.
"""

from __future__ import annotations

import inspect

import pytest

from lllm3090 import config, engine, speculation


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

    argv = inspect.getsource(engine.spec_flags)
    assert "gguf.has_mtp(model_path)" in argv, (
        "the flag must be decided from the checkpoint, not from a catalogue field"
    )


# ---------------------------------------------------------------------------
# Speculation profiles
# ---------------------------------------------------------------------------


@pytest.fixture
def with_mtp(fake_engine, monkeypatch):
    """The fake engine, with a checkpoint that carries the head.

    The fixture's model is four bytes of magic and nothing else, which reads
    correctly as "no MTP". Every profile worth testing is about what happens
    when the head *is* there.
    """
    monkeypatch.setattr(engine.gguf, "has_mtp", lambda path: True)
    return fake_engine


def test_the_default_is_what_shipped_before_there_was_a_choice(with_mtp):
    """MTP alone, at llama.cpp's own draft width. Passing no width is a
    decision -- 3 is the right width on Vulkan -- not an omission."""
    seen, model, _ = with_mtp
    engine.start(str(model), "M", 4096, 2, 0)
    argv = seen[0]
    assert argv[argv.index("--spec-type") + 1] == "draft-mtp"
    assert "--spec-draft-n-max" not in argv


def test_the_copy_profile_asks_for_both_drafters_and_a_wider_draft(with_mtp):
    """The configuration that takes the dense 27B from 94.0 to 115.1 tok/s on
    copy-heavy work, and the only reason the flags are a table at all."""
    seen, model, _ = with_mtp
    engine.start(str(model), "M", 4096, 2, 0, spec=speculation.COPY)
    argv = seen[0]
    assert argv[argv.index("--spec-type") + 1] == "draft-mtp,ngram-cache"
    assert argv[argv.index("--spec-draft-n-max") + 1] == "7"


def test_a_checkpoint_with_no_head_gets_no_speculation_at_all(fake_engine):
    """The default degrades rather than refusing: llama.cpp exits at load when
    handed draft-mtp against a checkpoint that has no head, so the alternative
    to dropping the flag is an engine that does not start."""
    seen, model, _ = fake_engine
    engine.start(str(model), "M", 4096, 2, 0)
    assert "--spec-type" not in seen[0]


def test_a_named_profile_is_refused_rather_than_approximated(fake_engine):
    """Where the default degrades, a request does not.

    --profile copy is a measurement of draft-mtp,ngram-cache. Run against a
    checkpoint with no head it would be ngram-cache alone -- a different
    configuration, which on Vulkan measures 0.65x. Serving that under the same
    name is how a number stops meaning anything.
    """
    seen, model, _ = fake_engine
    ok, detail = engine.start(str(model), "M", 4096, 2, 0, spec=speculation.COPY)
    assert not ok
    assert "multi-token prediction head" in detail
    assert not seen, "must not launch the engine at all"


def test_an_engine_too_old_to_speculate_is_launched_without_it(with_mtp, monkeypatch):
    """The build is pinned but not upgraded in place -- install-engine skips a
    binary that is already there -- so a binary older than --spec-type survives
    an lllm3090 upgrade, and passing it one would make it exit at load."""
    monkeypatch.setattr(engine, "supports", lambda flag: False)
    seen, model, _ = with_mtp
    engine.start(str(model), "M", 4096, 2, 0, spec=speculation.COPY)
    assert "--spec-type" not in seen[0]
    assert "--spec-draft-n-max" not in seen[0]


def test_the_draft_model_gets_the_same_quantised_cache_as_the_main_one(with_mtp):
    """The draft model keeps its own KV cache and llama.cpp defaults it to f16
    whatever --cache-type-k/v says, so for as long as only the main pair was
    passed, MTP's context sat at full precision beside a main cache at half.
    That gap was the whole of what MTP cost in memory: 2.45 of 4.80 KiB/token
    on the dense 27B, about 412 MiB at a 168k window."""
    seen, model, _ = with_mtp
    engine.start(str(model), "M", 4096, 2, 0)
    argv = seen[0]
    for flag in ("--cache-type-k", "--cache-type-v",
                 "--spec-draft-type-k", "--spec-draft-type-v"):
        assert argv[argv.index(flag) + 1] == engine.CACHE_TYPE


def test_a_prompt_drafter_is_not_handed_a_cache_it_does_not_have(fake_engine):
    """The n-gram modes draft from the prompt, with no model and no context to
    quantise. Passing the flag there would set one that describes nothing."""
    seen, model, _ = fake_engine
    prompt_only = speculation.Profile(
        name="t", spec_types=("ngram-cache",), draft_n_max=None,
        backends=None, summary="",
    )
    engine.start(str(model), "M", 4096, 2, 0, spec=prompt_only)
    argv = seen[0]
    assert argv[argv.index("--spec-type") + 1] == "ngram-cache"
    assert "--spec-draft-type-k" not in argv


def test_no_speculation_means_no_draft_cache_flags(fake_engine):
    """A checkpoint with no head gets no draft context, so sizing one is
    describing something that was never allocated."""
    seen, model, _ = fake_engine
    engine.start(str(model), "M", 4096, 2, 0)
    assert "--spec-draft-type-k" not in seen[0]


def test_a_binary_too_old_for_the_draft_cache_flags_still_starts(
    with_mtp, monkeypatch
):
    """These arrived after --spec-type did, so supporting one is not supporting
    the other, and an unknown argument makes llama-server exit at load."""
    monkeypatch.setattr(
        engine, "supports", lambda flag: not flag.startswith("--spec-draft-type")
    )
    seen, model, _ = with_mtp
    engine.start(str(model), "M", 4096, 2, 0)
    argv = seen[0]
    assert argv[argv.index("--spec-type") + 1] == "draft-mtp"
    assert "--spec-draft-type-k" not in argv
    assert "--spec-draft-type-v" not in argv


def test_both_caches_are_set_from_one_constant():
    """They have to agree, and the bug was that they were written in two places
    and only one of them was written."""
    source = inspect.getsource(engine)
    assert source.count('"q8_0"') == 1, (
        "q8_0 belongs in CACHE_TYPE alone; a second literal is a second cache "
        "setting free to drift from the first, which is this exact bug"
    )


def test_a_width_the_binary_has_never_heard_of_is_not_passed(with_mtp, monkeypatch):
    """--spec-type and --spec-draft-n-max arrived in different releases, so
    supporting one is not supporting the other."""
    monkeypatch.setattr(engine, "supports", lambda flag: flag != "--spec-draft-n-max")
    seen, model, _ = with_mtp
    engine.start(str(model), "M", 4096, 2, 0, spec=speculation.COPY)
    argv = seen[0]
    assert argv[argv.index("--spec-type") + 1] == "draft-mtp,ngram-cache"
    assert "--spec-draft-n-max" not in argv
def test_an_effort_level_is_handed_to_the_engine(fake_engine):
    """The only route to a model's thinking control: Claude Code's own
    `/effort` travels as `output_config.effort`, which llama.cpp neither
    implements nor rejects, so it never reaches the template."""
    seen, model, _ = fake_engine
    ok, _ = engine.start(str(model), "M", 4096, 2, 0, None, None, effort="low")
    assert ok
    argv = seen[0]
    assert argv[argv.index("--reasoning-effort") + 1] == "low"


def test_no_effort_means_no_flag(fake_engine):
    """Omitting it leaves the template's own default in charge, which is what
    every start before this flag existed did."""
    seen, model, _ = fake_engine
    engine.start(str(model), "M", 4096, 2, 0, None, None, effort=None)
    assert "--reasoning-effort" not in seen[0]


def test_an_engine_too_old_for_the_flag_is_refused(fake_engine, monkeypatch):
    """The opposite of how --spec-type degrades, on purpose. A level the user
    typed must not be dropped in silence -- silence is the failure this flag
    exists to fix."""
    seen, model, _ = fake_engine
    monkeypatch.setattr(engine, "supports", lambda flag: flag != "--reasoning-effort")
    ok, detail = engine.start(str(model), "M", 4096, 2, 0, None, None, effort="low")
    assert not ok
    assert "--reasoning-effort" in detail
    assert not seen, "must not launch the engine at all"


def test_a_level_the_template_refuses_fails_the_start(fake_engine, monkeypatch):
    """A loaded engine that 500s on every request is worse than no engine: it
    answers /health, so everything watching it reports it up."""
    _, model, _ = fake_engine
    monkeypatch.setattr(engine, "healthy", lambda: True)
    monkeypatch.setattr(
        engine, "template_refuses_effort",
        lambda effort, **kw: f"Unexpected reasoning effort {effort}.",
    )
    stopped = []
    monkeypatch.setattr(engine, "stop", lambda *a, **kw: stopped.append(True))
    ok, detail = engine.start(
        str(model), "M", 4096, 2, 300, None, None, effort="minimal"
    )
    assert not ok
    assert "does not accept --effort minimal" in detail
    assert "Unexpected reasoning effort minimal." in detail
    assert stopped, "the engine it just started must not be left serving errors"


def test_a_level_the_template_accepts_starts_normally(fake_engine, monkeypatch):
    _, model, _ = fake_engine
    monkeypatch.setattr(engine, "healthy", lambda: True)
    monkeypatch.setattr(engine, "template_refuses_effort", lambda effort, **kw: None)
    ok, detail = engine.start(str(model), "M", 4096, 2, 300, None, None, effort="low")
    assert ok
    assert "effort low" in detail, "the level is worth seeing in the ready line"
