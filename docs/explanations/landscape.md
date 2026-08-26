# What else is out there, and what this does that they do not

The obvious question about a project like this is why it exists at all, when
running a local model is a solved problem with several popular answers. Most of
it *is* solved, and by better-resourced software than this. One part is not.

## The serving layer is commodity

Starting an engine, stopping it, swapping models, downloading weights, showing a
log in a browser — every one of those is available elsewhere, and the gap closed
almost entirely during 2025.

`llama-server` itself gained **router mode** in December 2025: it discovers
models in its own cache, loads one on first request, unloads the least recently
used when it hits `--models-max`, and switches models from a dropdown in its
built-in web UI.

[`llama-swap`](https://github.com/mostlygeek/llama-swap) is the mature form of
the same idea — a proxy that maps model IDs to launch commands, with a web UI
carrying token metrics, manual load and unload, idle timeouts, request filters
and streaming logs. It is backend-agnostic, so it fronts vLLM and TabbyAPI as
readily as llama.cpp.

Above those sit the batteries-included stacks: Ollama, LM Studio, Jan, LocalAI,
KoboldCpp, RamaLama, and GPUStack for more than one machine. All of them
download, serve, swap and chat from one install.

**None of this is a reason to use `lllm3090` instead.** Whether to adopt
`llama-swap` here rather than maintain a process supervisor is an open question,
recorded in [issue #35](https://github.com/gilesknap/lllm3090/issues/35).

The Anthropic Messages API is not a differentiator either. `llama.cpp` serves
`/v1/messages` natively, and Ollama and LM Studio both added it in early 2026,
so pointing Claude Code at any of them works. `lllm3090 claude` sets three
environment variables for one subprocess; it is a convenience, not a capability.

## What nothing else computes

VRAM calculators exist, as web pages. They take parameter count times
bits-per-weight for the weights, and apply a single KV formula on top.

That formula assumes every layer is full attention, and it is the assumption
that breaks. As [](what-fits.md) sets out, linear-attention layers hold no
per-token KV at all, and sliding-window layers are provisioned as a fraction of
the token budget rather than at the window size. The consequence is that cache
cost does not track parameter count — `Qwen3-8B` costs **seven times more per
token** than `Qwen3.6-35B-A3B`, which is four times its size.

A generic formula gets both of those wrong, in opposite directions. So the
catalogue does the arithmetic per entry, from the architecture, and the numbers
are then checked against the card described in [](one-gpu.md).

Two further things follow from that and have no equivalent anywhere:

- **The budget is measured, not assumed.** A desktop session is holding VRAM
  that a text console is not, and the planner uses whichever target you are
  actually in.
- **Throughput figures are measured on this card**, not estimated from a
  bandwidth roofline. Where a catalogue entry claims a tok/s number, it was
  observed.

## What the popular choice does instead

Ollama selects a context length from a coarse VRAM tier — roughly 4K below
24 GiB — and truncates any input longer than that **silently**. No error, no
warning; the tokens simply never reach the model.

That surfaces much later, as a model that forgets things it was told or a
retrieval pipeline returning answers that are wrong in ways nobody can trace
back. Recovering the context the card can actually hold means setting `num_ctx`
yourself, which means already knowing the arithmetic.

On one 24 GB card doing agentic work, that is the exact failure this project
exists to prevent — and the reason the catalogue, not the panel, is the point.

## When you should use something else

Honestly, and in order of how likely you are to hit it:

- **You do not have a 3090.** Every figure here is computed for 24 GB at compute
  capability 8.6, and would be wrong on your card — see [](other-cards.md).
  Ollama or LM Studio will get you running today.
- **You want more than one model resident**, or unloaded automatically when
  idle. `llama-swap` does both; this does neither.
- **You have more than one GPU, or more than one machine.** GPUStack is built
  for that and this is explicitly not.
- **You want images, speech, or embeddings from the same server.** LocalAI
  covers far more model formats and modalities.

What is left, and what this is for: one 3090, one model at a time, and every
number about it checked before you spend 20 GB of disk finding out.
