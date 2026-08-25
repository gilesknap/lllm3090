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
        assert m.default_ctx <= m.max_ctx, f"{m.id} defaults past its RoPE ceiling"
        assert m.file.endswith(".gguf"), f"{m.id} does not name a GGUF"
        assert "/" in m.repo, f"{m.id} repo is not owner/name"


def test_every_catalogue_entry_fits_the_target_card():
    """The catalogue is curated. An entry that does not fit does not belong."""
    for m in catalog.load_catalog():
        assert catalog.fit(m).fits, f"{m.name} does not fit a 24 GB card"


def test_context_never_exceeds_the_rope_ceiling():
    for m in catalog.load_catalog():
        f = catalog.fit(m)
        assert f.max_ctx_q8 <= m.max_ctx
        assert f.max_ctx_f16 <= m.max_ctx


def test_q8_cache_buys_context_or_hits_the_ceiling():
    for m in catalog.load_catalog():
        f = catalog.fit(m)
        # Halving the per-token cost doubles context, unless the architecture's
        # own ceiling caps it first.
        assert f.max_ctx_q8 >= f.max_ctx_f16


def test_headless_is_never_worse_than_desktop():
    for m in catalog.load_catalog():
        assert catalog.fit(m, desktop=False).max_ctx_q8 >= catalog.fit(m).max_ctx_q8


def test_a_model_larger_than_the_card_does_not_fit():
    huge = catalog.Model(
        id="huge", name="Huge", repo="x/y", file="z.gguf",
        size_gb=40.0, kind="dense", params="", kv_kib_per_token=64,
        max_ctx=8192, default_ctx=8192,
    )
    f = catalog.fit(huge)
    assert not f.fits
    assert f.headline == "does not fit"


@pytest.mark.parametrize("desktop", [True, False])
def test_budget_is_positive_and_below_the_card(desktop: bool):
    budget = config.usable_vram_mib(desktop)
    assert 0 < budget < config.TARGET_VRAM_MIB
