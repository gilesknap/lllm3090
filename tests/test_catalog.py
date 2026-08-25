"""The fit arithmetic is what makes the panel's promises true or false."""

from __future__ import annotations

import pytest

from llm3090 import catalog, config


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
    budget = config.usable_vram_mib(desktop)
    assert 0 < budget < config.TARGET_VRAM_MIB
