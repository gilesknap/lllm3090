"""The sweep's arithmetic, and the promises its output makes.

Everything here runs offline. The configs in ``tests/data/model_configs.json``
are the real ones, trimmed to the fields the module reads, so a change to the
derivation is checked against what HuggingFace actually serves rather than
against a hand-written idealisation of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lllm3090 import catalog, config, hardware, sweep

CONFIGS = json.loads((Path(__file__).parent / "data/model_configs.json").read_text())


def test_the_derivation_reproduces_every_catalogue_figure():
    """The sweep must agree with the hand-checked catalogue, exactly.

    This is the test that licenses trusting the sweep on a model nobody has
    checked. Every entry in ``models.yaml`` had its KV cost worked out by hand
    from the architecture; if the same arithmetic run from ``config.json``
    reproduces all of them, it can be believed about a model that is not in the
    list yet. If it cannot, nothing downstream of it means anything.

    Gemma is the case that makes it worth having. Its full-attention layers
    carry their own head count and width -- ``num_global_key_value_heads`` and
    ``global_head_dim`` -- and reading the sliding layers' figures instead is
    wrong by 2x on the 26B and 4x on the 12B, in the direction that promises
    context the card cannot serve.
    """
    known = {m.name: m for m in catalog.load_catalog()}
    checked = 0
    for name, cfg in CONFIGS.items():
        assert name in known, f"fixture {name} is no longer in the catalogue"
        derived = sweep.kv_kib_per_token(cfg)
        assert derived == known[name].kv_kib_per_token, (
            f"{name}: derived {derived} KiB/token, catalogue says "
            f"{known[name].kv_kib_per_token}"
        )
        checked += 1
    assert checked >= 7, "the fixture set has shrunk; it should cover the catalogue"


def test_every_catalogue_entry_has_a_fixture():
    """A model added without a fixture is a model the sweep was never checked on."""
    names = {m.name for m in catalog.load_catalog()}
    # Two catalogue entries are the same checkpoint at different bit-rates, so
    # they share one config and one fixture.
    names.discard("Qwen3.6-35B-A3B-Q4KS")
    assert names <= set(CONFIGS), f"no config fixture for {names - set(CONFIGS)}"


def test_linear_and_sliding_layers_cost_nothing_per_token():
    """The whole reason a generic VRAM calculator gets this wrong.

    Only full attention grows with the conversation. Counting every layer would
    make Qwen3.8-27B four times more expensive than it is, and Gemma six.
    """
    qwen = CONFIGS["Qwen3.8-27B"]["text_config"]
    assert sweep._full_attention_layers(qwen) == 16
    assert qwen["num_hidden_layers"] == 64

    gemma = CONFIGS["Gemma-4-26B-A4B"]["text_config"]
    assert sweep._full_attention_layers(gemma) == 5
    assert gemma["num_hidden_layers"] == 30


def test_a_config_that_cannot_be_priced_is_refused_rather_than_guessed():
    """A wrong KV cost does not fail loudly, so it must not be produced at all.

    It yields a plan the card cannot honour, an engine that loads and reports
    itself healthy, and a failure at the first request. Skipping is cheap;
    guessing costs a download and a debugging session.
    """
    with pytest.raises(sweep.Unsupported):
        sweep.kv_kib_per_token({"kv_lora_rank": 512, "num_hidden_layers": 61})
    with pytest.raises(sweep.Unsupported):
        sweep.kv_kib_per_token({"num_hidden_layers": 32})
    with pytest.raises(sweep.Unsupported):
        sweep.max_ctx({"num_hidden_layers": 32})


def _candidate(**over):
    base = {
        "repo": "someone/Thing-GGUF", "file": "Thing-UD-IQ4_XS.gguf",
        "name": "Thing", "size_gb": 12.0, "kv_kib_per_token": 20,
        "max_ctx": 262144, "kind": "moe",
        "params": "40 layers, 10 full-attention + 30 other",
    }
    return sweep.Candidate(**{**base, **over})


def test_emitted_yaml_loads_back_as_a_catalogue_entry():
    """The output is meant to be pasted, so pasting it has to work.

    An entry that parses is an entry the panel can already price, which turns
    "paste and hope" into "paste and the rest of this suite checks it".
    """
    priced = sweep.price(_candidate(), hardware.reference())
    text = sweep.to_yaml([priced])
    entries = yaml.safe_load("models:\n" + text)["models"]
    assert len(entries) == 1
    model = catalog.Model(**entries[0])
    assert model.kv_kib_per_token == 20
    assert model.max_ctx == 262144


def test_a_swept_entry_never_claims_a_speed():
    """Speed is a measurement on one card and this tool never takes one.

    The roofline that would derive one is calibrated between 20% and 35% of
    roof for a resident MoE -- wider than the error bar shipping a number would
    imply. Emitting one would be the exact mistake the catalogue's
    "measured, not estimated" rule exists to prevent.
    """
    priced = sweep.price(_candidate(), hardware.reference())
    entry = yaml.safe_load("models:\n" + sweep.to_yaml([priced]))["models"][0]
    assert entry["verified"] is False
    assert "expected_tok_s" not in entry
    assert catalog.Model(**entry).expected_tok_s is None


def test_a_vision_candidate_carries_its_projector_into_the_entry():
    """The projector costs VRAM, so an entry that omits it over-promises."""
    priced = sweep.price(
        _candidate(mmproj="mmproj-F16.gguf", mmproj_gb=1.1), hardware.reference()
    )
    entry = yaml.safe_load("models:\n" + sweep.to_yaml([priced]))["models"][0]
    model = catalog.Model(**entry)
    assert model.vision
    assert model.weights_mib > catalog.Model(**{**entry, "mmproj_gb": 0.0}).weights_mib


# ---------------------------------------------------------------------------
# min_vram_mib, and the three states it feeds
# ---------------------------------------------------------------------------


def _card(vram_mib: int, capability: str = "8.6") -> hardware.Profile:
    return hardware.Profile(
        id="test", name="test card", compute_capability=capability,
        vram_mib=vram_mib, bandwidth_gbs=0,
    )


@pytest.mark.parametrize("model", catalog.load_catalog(), ids=lambda m: m.id)
def test_min_vram_is_exactly_the_card_that_clears_the_agent_floor(model):
    """The derived figure must be the real threshold, not an approximation.

    It is the inverse of ``fit()``, so it is only worth anything if a card of
    that size is agent-ready and a slightly smaller one is not. Anything looser
    and the panel's "about N GB would clear it" is advice that does not hold.
    """
    need = catalog.min_vram_mib(model, desktop=True)
    if need is None:
        # Bounded by RoPE rather than by memory: no card lifts it, and the
        # claim to check is that an enormous one still does not.
        assert not catalog.plan(model, desktop=True, profile=_card(196608)).agent_ready
        return

    enough = catalog.plan(model, desktop=True, profile=_card(int(need))).agent_ready
    assert enough, f"{model.name}: {need} MiB was supposed to be enough"

    short = catalog.plan(
        model, desktop=True, profile=_card(int(need) - 1024)
    ).agent_ready
    assert not short, f"{model.name}: {need} MiB is not the threshold, it is padding"


def test_fitting_and_being_usable_are_reported_as_different_things():
    """The distinction the catalogue was built to make, enforced in the UI data.

    Two of the models most often recommended for a 24 GB card fit it and leave
    a window an agent cannot use. A front end with a single "fits" flag has to
    render those as successes, which is how someone spends 20 GB of disk before
    finding out.
    """
    tight = next(
        m for m in catalog.load_catalog() if m.name == "Qwen3.6-35B-A3B-Q4KS"
    )

    def state_on(vram_mib: int) -> tuple[str, str]:
        card = _card(vram_mib)
        return catalog.status(
            tight, catalog.plan(tight, desktop=True, profile=card),
            catalog.fit(tight, True, card), card,
        )

    # It loads on a 24 GB card with room to spare, and still cannot hold an
    # agent's system prompt at the two slots the plan hands out by default.
    assert catalog.fit(tight, True, _card(24576)).fits, "the premise is that it fits"
    state, note = state_on(24576)
    assert state == catalog.STATUS_TIGHT
    assert "40k" in note and "GB would clear it" in note

    # The same entry on a 5090 is simply fine, which is the whole point of
    # computing this per card rather than declaring it once for a 3090.
    assert state_on(32768)[0] == catalog.STATUS_OK


def test_a_model_no_card_can_rescue_does_not_send_you_shopping():
    """Qwen3-8B's ceiling is the architecture's, and a bigger card cannot move it."""
    eight_b = next(m for m in catalog.load_catalog() if m.name == "Qwen3-8B")
    assert eight_b.max_ctx <= config.AGENT_PROMPT_FLOOR
    assert eight_b.min_vram_gb is None

    huge = _card(98304)
    state, note = catalog.status(
        eight_b, catalog.plan(eight_b, desktop=True, profile=huge),
        catalog.fit(eight_b, True, huge), huge,
    )
    assert state == catalog.STATUS_TIGHT
    assert "no card lifts it" in note
    assert "GB would clear it" not in note


def test_a_capability_floor_is_honoured_and_unknowns_are_not_held_against_a_card():
    """The one hardware requirement that cannot be derived from a file size."""
    blackwell_only = catalog.Model(
        id="b", name="B", repo="x/y", file="y.gguf", size_gb=8.0, kind="dense",
        params="", kv_kib_per_token=20, max_ctx=262144,
        min_compute_capability="12.0",
    )
    assert not catalog.capability_ok(blackwell_only, _card(24576, "8.6"))
    assert catalog.capability_ok(blackwell_only, _card(32768, "12.0"))
    # A card that will not say cannot be proven inadequate, and hiding half the
    # catalogue on that basis is worse than letting a download fail.
    assert catalog.capability_ok(blackwell_only, _card(24576, "unknown"))
    # And the common case: nothing declared, nothing objected to.
    for m in catalog.load_catalog():
        assert catalog.capability_ok(m, _card(24576, "8.6"))


def test_capability_outranks_capacity_in_the_reported_state():
    """A card that cannot run it at all should not be told to buy more memory."""
    model = catalog.Model(
        id="b", name="B", repo="x/y", file="y.gguf", size_gb=8.0, kind="dense",
        params="", kv_kib_per_token=20, max_ctx=262144,
        min_compute_capability="12.0",
    )
    card = _card(24576, "8.6")
    state, note = catalog.status(
        model, catalog.plan(model, desktop=True, profile=card),
        catalog.fit(model, True, card), card,
    )
    assert state == catalog.STATUS_CAPABILITY
    assert "12.0" in note and "8.6" in note


def test_notes_do_not_quote_a_context_figure():
    """Notes are judgement; context is arithmetic that recomputes per card.

    Two entries carried absolute windows -- "212k per conversation", "61k
    against 212k" -- that were right when written and wrong once the driver
    reserve and the KV overhead factor landed. The row then said one thing in
    its context column and another in its notes, with nothing to say which was
    current. A note cannot be recomputed, so it must not carry a number that
    needs recomputing.

    Slot counts are the same claim wearing different clothes -- "four slots at
    its full 128k window" is as card-dependent as a window is, and ``plan()``
    hands out a different number on a 16 GB card than on a 96 GB one. The first
    version of this guard checked only windows and let three of those through.
    """
    import re

    banned = (
        # A window: "212k per conversation", "61k against 212k".
        r"\b\d{2,3}k\s+(per conversation|against)",
        # A slot count, in digits or words: "four slots", "three concurrent
        # conversations", "two slots at its full window".
        (
            r"\b(\d+|one|two|three|four|five|six)\s+"
            r"(slots?|concurrent\s+conversations?)\b"
        ),
    )
    for m in catalog.load_catalog():
        for pattern in banned:
            assert not re.search(pattern, m.notes, re.IGNORECASE), (
                f"{m.name}: notes assert a figure the panel computes per card "
                f"and this note cannot follow -- matched /{pattern}/"
            )


def test_the_advised_card_size_rounds_up():
    """A threshold rounded to nearest is advice to buy the wrong card.

    ``min_vram_gb`` is the smallest card that works, not an estimate of one. A
    requirement of 24.4 GB rendered as "about 24 GB" sends someone to a 24 GB
    card that cannot run it -- and unlike every other error here, that one
    costs money rather than a download.
    """
    assert catalog._advise_gb(24.4) == 25
    assert catalog._advise_gb(24.0) == 24
    assert catalog._advise_gb(23.1) == 24


def test_the_emitted_note_names_the_card_it_was_priced_against():
    """With --gpu, the plan is a claim about the overridden card, not this one.

    Attaching a correct figure to the wrong hardware is worse than omitting it:
    the reader has no way to see that the two halves of the sentence disagree.
    """
    five_thousand_ninety = next(
        p for p in hardware.load_profiles() if p.id == "rtx-5090"
    )
    text = sweep.to_yaml(
        [sweep.price(_candidate(), five_thousand_ninety)], five_thousand_ninety
    )
    assert "RTX 5090" in text
    assert "RTX 3090" not in text
