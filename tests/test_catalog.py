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
    budget = config.usable_vram_mib(desktop)
    assert 0 < budget < config.TARGET_VRAM_MIB


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
