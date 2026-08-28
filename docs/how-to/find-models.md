# Find models that fit your card

The catalogue is small, hand-checked, and always behind. `lllm3090 sweep` is
how it gets widened: it asks HuggingFace what GGUF checkpoints exist, works out
what each one would cost on *your* card, and tells you which are worth spending
disk on.

It downloads no weights. A candidate costs one `config.json`.

```bash
lllm3090 sweep
```

```text
Pricing against NVIDIA GeForce RTX 3090 (24 GB), desktop session held back
Surveying the top 100 GGUF repositories...

CANDIDATE                             SIZE   KIND     KV  CONTEXT
Qwen3.5-9B                            6.0G  dense    32K  256k x 2 (pool 512k, limited by rope)
...

Priced and rejected (7):
  Rejected entries fit the card. They are rejected because one conversation
  gets less than the 40k an agent harness spends on its system prompt before
  your first word.

  Qwen3-Coder-30B-A3B-Instruct    fits, but leaves 35k per conversation ...
  deepseek-v4                     too big for this card -- needs about 153 GB
```

## The rejections are the point

A model that fits and a model you can use are different claims, and the gap is
wide. `Qwen3-Coder-30B-A3B-Instruct` loads on a 24 GB card with room to spare
and leaves 35k per conversation — less than Claude Code spends on its own
system prompt and tool definitions before you type anything.

That is why the sweep prints what it rejected and why. A survey that showed
only its successes would lose that finding every time it ran, and the models it
loses it about are the ones most often recommended for this card.

## Pricing for a card you do not have

```bash
lllm3090 sweep --gpu rtx-5090
```

Any profile in `profiles.yaml` works: `rtx-3090`, `rtx-4090`, `rtx-5090`,
`rtx-5080`, `rtx-pro-6000-blackwell`. Useful for answering "would this be worth
it on the card I am thinking of buying" without owning one, and for checking
what a friend on different hardware would see.

Speeds are never priced this way. Fit is arithmetic over capacity and holds on
any card; tokens per second is a measurement on one card and is not
extrapolated to another. See [](../explanations/other-cards.md).

## Adding what you find to the catalogue

```bash
lllm3090 sweep --yaml
```

emits catalogue entries for the survivors, ready to paste into
`src/lllm3090/data/models.yaml`. Everything derivable is filled in — size from
the API, KV cost from the model's own `config.json`, the RoPE ceiling, whether
it is sparse or dense, the projector if it ships one.

Three things are deliberately **not** filled in:

- **`expected_tok_s` is absent and `verified` is false.** Speed is a
  measurement. Download the model and run `lllm3090 bench <model>`, then set
  both by hand.
- **`notes` is a `TODO`.** Notes answer "why would someone pick this over the
  entry above it", which no arithmetic produces. A plausible-sounding sentence
  generated here would read exactly like the hand-written ones around it.
- **`tags` is `[TODO]`.** Same reason.

Check `max_ctx` before you trust it. It comes from `max_position_embeddings`,
which some models report as a YaRN-extended ceiling rather than the window the
base model holds coherently — `Qwen3-8B` claims 40960 and is curated here at
32768.

## What it skips, and why it skips loudly

Run with `--skipped` to see what could not be priced:

```bash
lllm3090 sweep --skipped
```

A repo is skipped when its KV cost cannot be derived: no `config.json` in the
repo **and** no `base_model` tag to borrow one from (a GGUF conversion often
keeps no config of its own, so the Hub's record of what it was converted from
is followed), a multi-head latent attention model whose cache layout depends on
the engine build rather than the config, or no ~4-bit single-file quant.

Skipping is deliberate, and it is the safe direction. A wrong KV figure does
not fail loudly: it produces a plan the card cannot honour, an engine that
loads and reports itself healthy, and a failure at the first request. That is
the failure this project exists to prevent, so a candidate the arithmetic
cannot reach is dropped rather than guessed at.

## Trusting the arithmetic

The derivation is checked against the shipped catalogue. Every entry in
`models.yaml` had its KV cost worked out by hand from the architecture, and
`tests/test_sweep.py` asserts that the same arithmetic run from each model's
`config.json` reproduces all of them exactly — 144, 64, 24, 20, 20, 16 and 13
KiB per token.

That is what licenses believing it about a model nobody has checked. It also
catches the trap that makes this harder than it looks: Gemma-4 gives its
full-attention layers a head count and width of their own
(`num_global_key_value_heads`, `global_head_dim`), and reading the sliding
layers' figures instead is wrong by 2x on the 26B and 4x on the 12B — in the
direction that promises context the card cannot serve.
