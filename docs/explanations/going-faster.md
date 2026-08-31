# Speed levers, tried and untried

The dense 27B is the slowest model in the catalogue, and
[structurally so](dense-vs-moe.md) — it reads every one of its parameters per
token. That is not the whole story: multi-token prediction has already taken it
from 34.9 to 56.6 tok/s, and several further levers have been looked at. This
page records all of them — what is already in the engine, what was measured and
set aside, what is waiting on a build, and what is parked and why.

## Two clocks, not one

Every lever below moves exactly one of these, and confusing them wastes a day:

- **Decode** — tokens after the first. Memory-bound: bandwidth divided by bytes
  read per token. Speculative decoding attacks this and nothing else.
- **Prefill** — turning the prompt into a KV cache before the first token.
  Compute-bound, and the dense 27B's worst number: about two minutes at 80k
  under Vulkan, against 21 s for the sparse 35B at 40k. Backend and matrix-core
  support attack this; speculation does not touch it.

A change that doubles decode leaves the first-turn-on-a-big-repo problem exactly
where it was.

## Already in the engine

Not levers to pull, but the baseline everything else is measured against — see
`lllm3090.engine.start`:

- **Multi-token prediction**, when the checkpoint has the head for it.
  `--spec-type draft-mtp` is added on evidence rather than on a catalogue claim:
  `lllm3090.gguf.has_mtp` reads the tensor names out of the file, because
  llama.cpp refuses to start with that flag against a checkpoint lacking the
  head. Measured on this card, on the pinned build: **34.9 → 56.6 tok/s
  (1.62×)** on the dense 27B, 130.5 → 171.8 (1.32×) on the sparse 35B. The
  table in the next section re-measures the dense model on a newer engine and
  lands in the same place.
- **All layers on the GPU** (`--n-gpu-layers 999`). Nothing in the catalogue is
  allowed to need offload; a model that does not fit is not offered.
- **Flash attention on** (`-fa on`), not left at `auto`.
- **q8_0 KV cache**, both K and V. This halves KV against f16 and is what puts
  the dense 27B's window in reach at all. Going further to q4_0 saves another
  72% of KV but costs roughly 37% of decode at long context — the
  dequantisation is on the critical path — so it is not taken.
- **Whole windows before second slots.** `lllm3090.catalog.plan` fills one
  window to the architecture's ceiling and grants another slot only where it
  costs nothing. The dense 27B is therefore a single 172k session, capped by
  VRAM. See [](../how-to/context-and-slots.md).

Note that `expected_tok_s` in the catalogue is still the pre-MTP 35 for this
model. That figure is a floor now, not a prediction.

## Measured and set aside: prompt-lookup drafting

MTP is one of several modes behind llama.cpp's `--spec-type`. `ngram-cache` was
swept against it on the dense 27B twice — once on a build that turned out to be
broken, and again on **b10715**, which contains the fix. It loses on every
workload tried.

| | prose | code-edit | long-copy |
|---|---|---|---|
| baseline | 34.1 | 33.8 | 32.8 |
| `draft-mtp` | 54.8 (1.61×) | 55.7 (1.65×) | **59.7 (1.82×)** |
| `ngram-cache` | 22.2 (0.65×) | 26.2 (0.78×) | 28.9 (0.88×) |
| `draft-mtp,ngram-cache` | 46.2 (1.35×) | 50.0 (1.48×) | 52.7 (1.61×) |

Both objections that kept the earlier result from being a verdict have now been
answered, and neither rescued it:

1. **The workload was wrong, and fixing it was not enough.** n-gram drafting can
   only pay when the output repeats a long stretch of the input, and the first
   measurement used a seven-line edit. The `long-copy` prompt — 369 lines copied
   back with one identifier renamed — lifts acceptance from **0%** to **58%**,
   so the objection was sound. It is still 0.88×. Copy-heavy work is where
   prompt-lookup does least badly here, not where it wins.
2. **The build was broken, and it had been flattering ngram.** The natural
   assumption was that a bug producing invalid acceptance numbers was costing
   ngram a fair hearing. It was doing the opposite: on the pre-fix build prose
   read 0.87× at 47% acceptance, and on b10715 the same sweep reads **0.65× at
   24%**. The target had been accepting drafts it never chose. **MTP's numbers
   did not move** (1.62× → 1.61×), because its drafts are short and mostly
   right, so there were few spurious accepts to lose.

Stacking remains worse than MTP alone, and the acceptance column says why:
adding ngram to MTP takes long-copy from 100% acceptance down to 87%. The weak
drafts displace good ones rather than adding to them.

**The pinned build still predates the fix.** `cli.LLAMA_BUILD` is `b10628`
(25 August); [PR #27812](https://github.com/ggml-org/llama.cpp/pull/27812)
merged on the 28th. Anything measured on the installed engine reproduces the
invalid acceptance numbers, so a sweep that matters needs `SWEEP_LLAMA_DIR`
pointed at a newer build until the pin moves.

## The live candidate: DFlash2

[DFlash2](https://inco.ai/blog/dflash2/) is a block-diffusion drafter published
for this exact target model. It predicts a whole block of tokens per forward
pass and keeps the top candidates at every position, which is why its acceptance
length is 5.13–5.39 tokens per verification step where a conventional 0.6B
drafter manages about 2. Output is unchanged.

Reported, on the 27B:

| hardware | baseline | DFlash2 | |
|---|---|---|---|
| RTX PRO 6000, CUDA, LiveCodeBench | 67.97 tok/s | 153.91 tok/s | 2.26× |
| same, 36k synthetic | | | 3.55× |
| Apple M5 Pro, Q4_K_M | 10.42 tok/s | | 1.85× |

Read those multipliers carefully. **They are against no speculation at all, and
this engine already runs MTP.** The number that decides it is DFlash2 against
56.6 tok/s, not against 34.9 — and the published claim on that narrower
comparison is correspondingly narrower: DFlash2 beats MTP by 0.52 tokens of
acceptance length on this model. Nothing has been measured on a 3090 under
Vulkan either way.

**What it costs.** The drafter is a second GGUF —
[1.1 GB at Q4_K_M](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF),
2.0 GB at Q8_0 — and the two drafters are alternatives, not additions. At
32 KiB/token of q8_0 KV, 1.1 GB is roughly 36k tokens, taking the dense 27B's
session from about 172k to about 136k before the drafter's own cache. Whether
that trade is worth it is a per-model judgement the catalogue would have to
carry.

**What it needs.** `--spec-type` has listed `draft-dflash` since DFlash v1
landed in July, so the flag being present proves nothing. DFlash2 — the local
convolution and candidate selector — is
[PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342), first released in
build **b10658**; work was still landing at b10715. Invocation:

```
--spec-type draft-dflash --spec-draft-model DRAFTER.gguf --spec-draft-n-max 7
```

llama.cpp defaults that width to 3 and the published figures use 7.
[Issue #27544](https://github.com/ggml-org/llama.cpp/issues/27544) reports
speculation throughput collapsing on Vulkan with `-np > 1` and a draft width
above 8 — confirmed on AMD and Intel, explicitly not on NVIDIA CUDA, untested on
NVIDIA Vulkan. The dense 27B now runs one slot, so this only bites a caller who
asks for more.

## CUDA instead of Vulkan

The prefill lever, and the one with the largest recorded gap. Community
scoreboards put CUDA about 36% ahead of Vulkan on prompt processing; this
project's own measurements put Vulkan 3–4× behind. The 3090 entries in
llama.cpp's CUDA scoreboard sit at pp512 ≈ 5560 with flash attention.

[Why this is scoped to one GPU](one-gpu.md) records the reason for Vulkan:
prebuilt CUDA binaries for Linux do not exist, and building needs a toolchain
the distribution did not ship. Half of that is still true and half has expired:

- **Still true.** As of b10715 llama.cpp publishes CUDA archives for Windows
  only. Linux gets CPU, Vulkan, ROCm, SYCL and OpenVINO. There is no download.
- **No longer true.** CUDA 13.3 lists Ubuntu 26.04 LTS as a supported
  distribution and GCC 6.x–15.x as supported host compilers. The reference
  machine runs Ubuntu 26.04 with GCC 15.2. The compiler objection has lapsed,
  and building from source is the ordinary route on Linux — it is the only route
  llama.cpp's own build documentation describes.

One prediction worth writing down before it is tested, because it is falsifiable
and cheap to check once a CUDA build exists: **the drafter verdicts above may
not survive the backend change, and ngram's should move furthest.** Verifying k
drafted tokens is a batched forward pass — the same work prefill does — and
prefill is precisely where Vulkan is 3–4× behind here. Prompt-lookup drafts long
runs where MTP drafts one or two, so it is the config most exposed to weak
batched compute, and its break-even acceptance should fall further than MTP's.
Issue #27544, where speculation collapses on Vulkan above a draft width of 8 but
not on CUDA, points the same way. Nothing has been measured on CUDA to support
this.

So the obstacle is no longer *can it build* but *what the project promises*.
`install-engine` currently downloads a tarball and is done; a CUDA path means
either a multi-gigabyte toolkit and a local compile on every install, or
shipping binaries built here — a different project with a different support
burden. Worth measuring the gain before deciding, since the gain is the whole
argument.

## Worth one check: coopmat2

Before any of that, confirm the current Vulkan build is using
`VK_KHR_cooperative_matrix` version 2 rather than falling back to coopmat1.
The difference was 2.2× against 4.4× on prefill in the original benchmarks, it
is driver-gated on NVIDIA, and it is free if it is merely switched off. The
device line in the engine log at model load says which is in use;
`--list-devices` does not.

## Parked

Neither of these is a bad idea; both are blocked on something outside this
project.

**TurboQuant KV cache.** Sub-3-bit KV via randomised Hadamard transforms —
3.25 bits/value at ~4.9× compression against f16, with quality loss reported as
negligible, and the dequantisation penalty that rules out q4_0 here designed out
rather than tolerated. For a model at 64 KiB/token this would be transformative.
It is **not merged**: it exists in several independent forks, CUDA and Metal are
validated, Vulkan is still in development. Watch
[discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969).

**vLLM or SGLang with AWQ-Marlin.** The headline is real — Marlin kernels at
741 tok/s against GGUF's 93 in one comparison — but it is a *concurrency*
number, measured at ten or more simultaneous users, and it comes with a higher
time-to-first-token under load. This engine serves one person. DFlash2 also runs
in vLLM, SGLang and TensorRT-LLM, so it is not an argument for switching either.
Revisit only if the workload becomes genuinely concurrent.

## How any of this gets decided

Not by reading the tables above. Every figure in
[the catalogue](../reference/catalogue.md) marked `verified: true` was measured
on this card, and speed claims here are held to the same standard — which means
a sweep, with a warm-up discarded after every server start, seven timed samples
per prompt per config, a unique prefix and `cache_prompt=false` on every
generation, and the served model checked either side of the run. Three samples
once made MTP read as no gain at all when it was 1.32×.

That instrument is `dev/spec-sweep.py`. It is a development tool rather than
part of the installed package — it kills and restarts servers on its own port,
which is the opposite of what the engine lifecycle promises — but it launches
with the same flags `lllm3090.engine.start` uses, and that correspondence is the
point. Its `baseline` config is no speculation at all, so MTP appears as a
config rather than as the floor. Point `SWEEP_DRAFT` at a drafter GGUF to
include DFlash2:

```
SWEEP_DRAFT=~/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  dev/spec-sweep.py ~/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_S.gguf
```

`SWEEP_LLAMA_DIR` points it at a different engine, which is how a CUDA build
gets compared against a Vulkan one without editing the script between runs. Take
both from the same upstream build when doing that, or the comparison is a
version comparison as well as a backend one. Every run prints the build number
and the device line it found, because two tables of tokens per second that do
not say what produced them cannot be compared later — which is precisely how the
earlier numbers stopped being usable.
