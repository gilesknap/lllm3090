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

## Why not `llama.cpp` and an assistant to choose the flags

That is the sharper form of the question, and it deserves a straight answer,
because `llama-server` **is** the engine here. Nothing in this project extends
it: `engine.start` assembles `--model --alias --host --port --n-gpu-layers 999
--ctx-size --parallel --cache-type-k/v --jinja` and a handful of conditionals,
and llama.cpp does the rest. Nor does the objection above transfer — with
`llama-server` you set `--ctx-size` yourself, so the number is one you chose and
nothing is discarded silently behind it. The failure moves from a truncation you
cannot see to a number that either refuses to load or leaves half the card
unspent.

For one model, on a card you already know, the answer is largely yes. Ask, keep
the resulting command in a shell script, and most of the value is yours. What
does not survive is the *next* model, and this is why.

**The per-token cost is a fact about a checkpoint, not about its name.** It
falls out of how many of the model's layers keep a KV cache at all, which is
readable from the GGUF header and from nowhere else. Asked without that file in
hand, the plausible answer is parameter count times a generic KV formula — the
one [](what-fits.md) shows to be wrong in both directions at once, and which
gives no sign of being wrong. Here `kv_kib_per_token` is read from the header,
recorded once, and checked by a test against the real file on disk.

**The budget half is a stack of constants that cannot be inferred at all.**
`DRIVER_RESERVE_MIB`, `WORKSPACE_RESERVE_MIB`, `DESKTOP_RESERVE_MIB`,
`ALLOCATOR_OVERHEAD`, `MTP_LAYER_OVERHEAD`, `BACKEND_KV_FACTOR` — every one of
them exists because a plan that ignored it produced an engine that would not
start, or a window 39k tokens smaller than the card could hold. They are not
reasoning; they are residue. An assistant can be *told* them, which means
reading them out of this repository.

**A flag's verdict can invert between backends.** `ngram-cache` drafting is
1.41x on CUDA copy-heavy work and 0.65x on Vulkan prose; a draft width of 7 wins
copying on CUDA and costs MTP 22% on Vulkan. So "switch on speculative decoding,
it is faster" is both true and false on this machine depending on which engine
is installed, and the default install is the one it is false on. That is why
`speculation.py` is a table of measured verdicts that refuses itself on the
backends where it was not measured, rather than a flag.

**And speed is not a choice at all.** `expected_tok_s: 55, verified: true` is an
observation. No amount of argument-picking produces one, and an estimate printed
in the same column would read exactly like a measurement.

None of that makes the catalogue clever. It makes it *cached*: `lllm3090 sweep`
is precisely "derive an entry from the model's own `config.json`, fill in
everything that follows from it, and invent nothing else" — the
machine-checkable form of asking. What you get over asking each time is not
capability, it is trust: the arithmetic was done once, against the file, with a
test holding it there, so it does not quietly drift on the day you are least
likely to check it.

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
