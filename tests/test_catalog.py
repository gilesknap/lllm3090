"""The fit arithmetic is what makes the panel's promises true or false."""

from __future__ import annotations

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


# --- hardware profiles ------------------------------------------------------
# The point of profiles is that the arithmetic travels to cards nobody here
# owns. These are the tests that can be written without one.


@pytest.mark.parametrize(
    "profile_id, expect_fits",
    [
        ("rtx-3090", 5),
        ("rtx-4090", 5),   # same capacity as the 3090
        ("rtx-5090", 5),   # more
        ("rtx-5080", 2),   # 16 GB: only the two small models
        ("rtx-pro-6000-blackwell", 5),
    ],
)
def test_what_fits_follows_the_card(profile_id, expect_fits):
    from lllm3090 import hardware

    profile = next(p for p in hardware.load_profiles() if p.id == profile_id)
    fits = [m for m in catalog.load_catalog() if catalog.fit(m, profile=profile).fits]
    assert len(fits) == expect_fits, (
        f"{profile_id}: {[m.name for m in fits]}"
    )


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
