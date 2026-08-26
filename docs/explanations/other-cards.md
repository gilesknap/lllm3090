# Running on a card that is not a 3090

This project was built and measured on one GPU, and the honest position is that
the numbers it reports are of two different kinds.

## Arithmetic travels; measurements do not

**Whether a model fits, and what context it leaves**, is arithmetic over
capacity: weights, minus reserves, divided by KV cost per token. Give it the
memory a card actually has and it is correct there too. Nothing about it is
3090-specific.

**Tokens per second** is a measurement. It is true of the card it was taken on
and says nothing about another. Scaling it by a ratio of memory bandwidths
produces a prediction — and the predictions in this repository were wrong by
27–50% the last time that was tried, because the formula being used assumed
experts crossing PCIe when they were resident in VRAM. See
`.claude/skills/moe-throughput-model` for the correction.

So `lllm3090` computes fit for whatever card it finds, and refuses to restate
someone else's measurement as though it were yours.

## What you get on each card

`profiles.yaml` ships capacity and compute capability for the cards below. Fit
and context are computed for all of them; only the 3090 has measured speeds.

| card | VRAM | models that fit |
|---|---|---|
| RTX 3090 | 24 GB | all eight |
| RTX 4090 | 24 GB | all eight — identical capacity, so identical context |
| RTX 5090 | 32 GB | all eight, with more context |
| **RTX 5080** | **16 GB** | **three** — `gpt-oss-20b`, `Qwen3-8B`, `Gemma-4-12B-QAT` |
| RTX PRO 6000 Blackwell | 96 GB | all eight, most of them capped by RoPE |

The 5080 is worth dwelling on: it has **less memory than a card from 2020**, so
`Qwen3.8-27B` at 15.4 GB leaves no room for a usable cache and the two 35B
variants do not load at all. The panel says so rather than letting you find out
after a 17 GB download.

These counts are the output of `catalog.fit()`, not a hand-maintained list, so
`lllm3090 models` on the card in front of you is the authority — but they date
from a catalogue of eight entries and will drift as it grows.

## An unrecognised card

Not an error. `lllm3090` builds a profile from what `nvidia-smi` reports, so fit
and context are computed correctly from real memory, and marks it as unmeasured
so no speed is claimed for it.

A card is recognised by **name**, not by size. An RTX 3090 Ti carries the same
24 GB at the same compute capability as the 3090 every figure here was measured
on, and is still a different card — so it gets a profile of its own rather than
inheriting measurements taken on something else.

## No card at all

In CI or a container with no GPU, the catalogue is still readable: capacity is
borrowed from the RTX 3090 so the fit arithmetic has something to work with.
The profile is marked as absent, and the CLI and panel both say plainly that
nothing shown describes a card in that machine.

## Contributing numbers for your card

```bash
lllm3090 bench Qwen3.8-27B
```

That runs the bundled `llama-bench` and prints a profile block to paste into an
issue. It is the only way speeds for a card get into the catalogue: nobody here
owns a 5090, and inventing its numbers would be worse than leaving them blank.
