"""The fit arithmetic is what makes the panel's promises true or false."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lllm3090 import catalog, config


def test_every_catalogue_entry_is_well_formed():
    models = catalog.load_catalog()
    assert models, "catalogue is empty"
    for m in models:
        assert m.kind in {"dense", "moe"}, f"{m.id} has odd kind {m.kind}"
        assert m.kv_kib_per_token > 0, f"{m.id} has no KV cost"
        assert m.max_ctx > 0, f"{m.id} has no context ceiling"
        assert m.file.endswith(".gguf"), f"{m.id} does not name a GGUF"
        assert "/" in m.repo, f"{m.id} repo is not owner/name"


def test_every_catalogue_entry_fits_the_target_card():
    """The catalogue is curated. An entry that does not fit does not belong."""
    for m in catalog.load_catalog():
        assert catalog.fit(m).fits, f"{m.name} does not fit a 24 GB card"


def test_context_never_exceeds_the_rope_ceiling():
    for m in catalog.load_catalog():
        assert catalog.plan(m).per_session <= m.max_ctx


def test_q8_cache_buys_context():
    for m in catalog.load_catalog():
        f = catalog.fit(m)
        assert f.pool_q8 >= f.pool_f16


def test_plan_never_exceeds_vram_or_the_rope_ceiling():
    """The two limits that must both hold, for every model, at every slot count."""
    for m in catalog.load_catalog():
        for parallel in (1, 2, 4):
            p = catalog.plan(m, parallel)
            assert p.per_session <= m.max_ctx, f"{m.id} would extrapolate RoPE"
            assert p.pool <= catalog.fit(m).pool_q8, f"{m.id} pool exceeds VRAM"
            assert p.pool == p.per_session * p.parallel


def test_plan_leaves_room_for_a_subagent():
    """The default must admit a second conversation, not just a large first one.

    A pool sized for one session is what makes subagents serialise behind their
    parent and evict its cached prefix.
    """
    for m in catalog.load_catalog():
        p = catalog.plan(m)
        assert p.parallel >= 2, f"{m.id} leaves no room for a subagent"
        # Each slot must still hold an agent harness's system prompt (~40k)
        # plus room to work.
        assert p.per_session >= 32768, f"{m.id} gives only {p.per_session} per slot"


def test_small_models_get_their_whole_window():
    """Spare VRAM goes to concurrency, not to context the model cannot use."""
    small = next(m for m in catalog.load_catalog() if m.name == "Qwen3-8B")
    p = catalog.plan(small)
    assert p.per_session == small.max_ctx
    assert p.capped_by == "rope"


def test_rope_capped_models_are_given_the_free_slots():
    """If the architecture runs out before the card does, spend the rest on slots."""
    for m in catalog.load_catalog():
        p = catalog.plan(m)
        if p.capped_by != "rope":
            continue
        assert p.parallel > config.DEFAULT_PARALLEL, (
            f"{m.id} has spare cache it could serve another conversation with"
        )
        assert p.parallel <= config.MAX_AUTO_PARALLEL
        # Free means free: every slot still gets the whole window.
        assert p.per_session == m.max_ctx


def test_an_explicit_slot_count_is_honoured_exactly():
    """Auto-expansion must not override what the user asked for."""
    for m in catalog.load_catalog():
        for n in (1, 2, 3):
            assert catalog.plan(m, n).parallel == n


def test_headless_is_never_worse_than_desktop():
    for m in catalog.load_catalog():
        assert catalog.fit(m, desktop=False).pool_q8 >= catalog.fit(m).pool_q8


def test_a_model_larger_than_the_card_does_not_fit():
    huge = catalog.Model(
        id="huge", name="Huge", repo="x/y", file="z.gguf",
        size_gb=40.0, kind="dense", params="", kv_kib_per_token=64,
        max_ctx=8192,
    )
    f = catalog.fit(huge)
    assert not f.fits
    assert f.headline == "does not fit"


@pytest.mark.parametrize("desktop", [True, False])
def test_budget_is_positive_and_below_the_card(desktop: bool):
    from lllm3090 import hardware

    for profile in hardware.load_profiles():
        budget = profile.usable_vram_mib(desktop)
        assert 0 < budget < profile.vram_mib, profile.id


def test_the_agent_floor_separates_usable_models_from_unusable():
    """A window below the harness's own prompt cannot run it at all.

    Qwen3-8B is the case that matters: it is the model the tutorial sends people
    to first, and its 32k ceiling is under Claude Code's ~40k system prompt.
    """
    usable, unusable = [], []
    for m in catalog.load_catalog():
        window = catalog.plan(m).per_session
        (usable if window > config.AGENT_PROMPT_FLOOR else unusable).append(m.name)

    assert "Qwen3-8B" in unusable, "the 8B should be flagged as too small for agents"
    assert "Qwen3.8-27B" in usable
    assert "Qwen3.6-35B-A3B" in usable
    assert usable, "no model can run an agent harness -- the catalogue is broken"


def test_the_recommendation_agrees_with_the_measurements():
    """Exactly one model is recommended, and it is defensible on the numbers.

    The docs recommended the slowest model in the catalogue for a while, because
    the advice was written before anything was measured and nothing forced the
    two to agree. This is that forcing function.
    """
    models = catalog.load_catalog()
    recommended = [m for m in models if "recommended" in m.tags]
    assert len(recommended) == 1, f"expected one recommendation, got {recommended}"
    pick = recommended[0]

    assert pick.verified, "recommending a model whose speed was never measured"
    assert catalog.fit(pick).fits, "recommending a model that does not fit"
    assert catalog.plan(pick).per_session > config.AGENT_PROMPT_FLOOR, (
        "the recommended model cannot hold an agent harness's system prompt"
    )

    faster = [
        m.name for m in models
        if m.verified
        and catalog.fit(m).fits
        and (m.expected_tok_s or 0) > (pick.expected_tok_s or 0)
        and catalog.plan(m).per_session >= catalog.plan(pick).per_session
    ]
    # The exception the docstring allows, made enforceable: a faster model with
    # at least as much context may be passed over, but only if the recommended
    # entry's own note names it and says why -- in the catalogue, where the
    # panel shows it, rather than in a commit message nobody reads.
    unexplained = [n for n in faster if n not in pick.notes]
    assert not unexplained, (
        f"{unexplained} are faster than {pick.name} with no less context -- "
        f"either the recommendation moves, or {pick.name}'s notes name them "
        "and say why it stays"
    )


def test_a_projector_is_counted_against_the_budget():
    """Vision is not free: the projector sits in VRAM like any other weights.

    Leaving it out of the arithmetic would promise context the projector has
    already spent, which is the failure this whole module exists to prevent.
    """
    for m in catalog.load_catalog():
        if not m.vision:
            continue
        text_only = replace(m, mmproj=None, mmproj_gb=0.0)
        assert m.weights_mib > text_only.weights_mib, f"{m.id} projector is free?"
        assert catalog.plan(m).pool <= catalog.plan(text_only).pool, (
            f"{m.id} claims more context with a projector loaded than without"
        )


def test_a_vision_entry_holds_back_workspace_for_the_vision_tower():
    """A projector's file size is not what it costs to run.

    Measured on a 3090: Gemma-4-26B-A4B's 1.19 GB projector occupied 1376 MiB,
    and at a full KV pool the engine loaded, reported itself healthy, and failed
    *every* request with `vk::Device::allocateMemory: ErrorOutOfDeviceMemory`.
    Counting only the file promised context the card could not serve.
    """
    for m in catalog.load_catalog():
        if not m.vision:
            continue
        as_text = replace(m, mmproj=None, mmproj_gb=m.mmproj_gb)
        held_back = catalog.fit(as_text).pool_q8 - catalog.fit(m).pool_q8
        assert held_back > 0, (
            f"{m.id} reserves nothing extra for its vision tower"
        )


def test_a_vision_entry_names_a_projector_and_a_size():
    """One without the other silently under-counts VRAM or fails at launch."""
    for m in catalog.load_catalog():
        assert bool(m.mmproj) == (m.mmproj_gb > 0), (
            f"{m.id} has mmproj={m.mmproj!r} but mmproj_gb={m.mmproj_gb}"
        )
        if m.mmproj:
            assert m.mmproj.endswith(".gguf"), f"{m.id} projector is not a GGUF"


def test_a_projector_on_disk_is_never_served_as_the_model(tmp_path):
    """--model pointed at a projector starts an engine that answers nothing.

    Sorting matters: 'Gemma-4-26B...gguf' sorts before 'mmproj-F16.gguf' but a
    lower-cased model name would not, and the first GGUF in the directory is
    what gets served.
    """
    d = tmp_path / "some-model"
    d.mkdir()
    (d / "mmproj-F16.gguf").write_bytes(b"proj")
    (d / "zz-weights-Q4_K_XL.gguf").write_bytes(b"weights")
    entry = catalog.installed(models_dir=tmp_path)[0]
    assert entry["path"].endswith("zz-weights-Q4_K_XL.gguf")
    assert entry["mmproj"].endswith("mmproj-F16.gguf")


def test_a_directory_holding_only_a_projector_is_not_a_model(tmp_path):
    d = tmp_path / "orphan"
    d.mkdir()
    (d / "mmproj-F16.gguf").write_bytes(b"proj")
    assert catalog.installed(models_dir=tmp_path) == []


def test_headless_never_offers_less_than_a_desktop():
    """Freeing the compositor's VRAM can only add cache, never remove it."""
    for m in catalog.load_catalog():
        d = catalog.plan(m, desktop=True)
        h = catalog.plan(m, desktop=False)
        assert h.pool >= d.pool, f"{m.name} claims less with more VRAM free"
        assert h.per_session >= d.per_session


def test_an_unknown_session_is_assumed_to_be_a_desktop(monkeypatch):
    """Guessing headless would hand out context the card does not have.

    Both failure modes are silent, but they are not symmetric: over-reserving
    costs context, while under-reserving produces an engine that loads, reports
    itself healthy and fails every request.
    """
    from lllm3090 import hardware

    monkeypatch.setattr(hardware.shutil, "which", lambda _: None)
    assert hardware.graphical() is True

    def boom(*a, **k):
        raise OSError("systemctl exploded")

    monkeypatch.setattr(hardware.shutil, "which", lambda _: "/bin/systemctl")
    monkeypatch.setattr(hardware.subprocess, "run", boom)
    assert hardware.graphical() is True


def test_vram_needed_counts_the_projector_and_the_q8_cache():
    m = next(x for x in catalog.load_catalog() if x.vision)
    bare = catalog.vram_needed_mib(m, 0)
    assert bare == m.weights_mib, "no cache should cost only the weights"
    # A q8 cache is half the f16 figure the catalogue stores.
    grown = catalog.vram_needed_mib(m, 2048) - bare
    assert grown == 2048 * (m.kv_kib_per_token / 2) / 1024


# --- hardware profiles ------------------------------------------------------
# The point of profiles is that the arithmetic travels to cards nobody here
# owns. These are the tests that can be written without one.


@pytest.mark.parametrize(
    "profile_id",
    ["rtx-3090", "rtx-4090", "rtx-5090", "rtx-pro-6000-blackwell"],
)
def test_a_card_of_24gb_or_more_fits_the_whole_catalogue(profile_id):
    """The catalogue is curated for a 3090, so nothing in it may fail on one.

    Counting entries would make this a census that needs editing every time a
    model is added. The invariant is that curation means what it says.
    """
    from lllm3090 import hardware

    profile = next(p for p in hardware.load_profiles() if p.id == profile_id)
    missing = [
        m.name for m in catalog.load_catalog()
        if not catalog.fit(m, profile=profile).fits
    ]
    assert not missing, f"{profile_id} cannot fit curated entries: {missing}"


def test_a_16gb_card_is_told_the_truth_about_what_does_not_fit():
    """The 5080 has less memory than a card from 2020, and must say so.

    This is the case the profile work exists for: before it, the panel would
    have offered a 15.4 GB model on a 16 GB card and let someone find out after
    the download.
    """
    from lllm3090 import hardware

    small = next(p for p in hardware.load_profiles() if p.id == "rtx-5080")
    models = catalog.load_catalog()
    fits = [m.name for m in models if catalog.fit(m, profile=small).fits]
    assert fits, "a 16 GB card should still run something"
    assert len(fits) < len(models), "16 GB cannot hold everything curated for 24"
    assert "Qwen3.8-27B" not in fits, "15.4 GB of weights leaves no usable cache"
    assert "gpt-oss-20b" in fits, "the small models are the point of this card"


def test_a_smaller_card_never_promises_more_context():
    from lllm3090 import hardware

    profiles = {p.id: p for p in hardware.load_profiles()}
    small, big = profiles["rtx-5080"], profiles["rtx-5090"]
    for m in catalog.load_catalog():
        s = catalog.plan(m, profile=small)
        b = catalog.plan(m, profile=big)
        assert s.pool <= b.pool, f"{m.name} claims more on the smaller card"


def test_exactly_one_profile_carries_the_measurements():
    from lllm3090 import hardware

    measured = [p for p in hardware.load_profiles() if p.measured]
    assert len(measured) == 1, "speeds belong to one card, not several"
    assert measured[0].id == config.REFERENCE_PROFILE


def test_a_card_sharing_a_size_is_not_mistaken_for_the_measured_one(monkeypatch):
    """A 3090 Ti is 24 GiB at compute 8.6 and is still not the reference card.

    Matching on capacity alone would hand it the rtx-3090 profile and print its
    speeds as measurements taken on the card in front of you.
    """
    from lllm3090 import hardware

    monkeypatch.setattr(hardware, "_smi", lambda q: {
        "name": "NVIDIA GeForce RTX 3090 Ti", "compute_cap": "8.6",
        "memory.total": "24576",
    }.get(q))
    p = hardware.detect()
    assert p.id != config.REFERENCE_PROFILE
    assert p.detected and not p.measured
    assert p.vram_mib == 24576


def test_no_gpu_does_not_borrow_the_reference_card_s_measurements(monkeypatch):
    """With no GPU, capacity may be borrowed to read the catalogue; speed may not."""
    from lllm3090 import hardware

    monkeypatch.setattr(hardware, "_smi", lambda q: None)
    p = hardware.detect()
    assert not p.present, "a machine with no GPU must not report one"
    assert not p.measured, "no card here, so no figure was measured on it"
    assert p.vram_mib == hardware.reference().vram_mib
    # The catalogue is still inspectable, which is the whole point of the fallback.
    assert catalog.load_catalog()


def test_an_unknown_card_is_honest_rather_than_absent(monkeypatch):
    """An unrecognised GPU must still compute fit, and must not claim speed."""
    from lllm3090 import hardware

    monkeypatch.setattr(hardware, "_smi", lambda q: {
        "name": "NVIDIA GeForce RTX 9090", "compute_cap": "13.0",
        "memory.total": "49152",
    }.get(q))
    p = hardware.detect()
    assert p.detected and not p.measured
    assert p.vram_mib == 49152
    # Fit still works, because capacity is all the arithmetic needs.
    assert all(catalog.fit(m, profile=p).fits for m in catalog.load_catalog())
