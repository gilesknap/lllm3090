# Speed levers, measured

Every lever tried on the dense 27B, with the number it produced. This page is
the scoreboard; [](what-makes-it-fast.md) explains what the levers *are* and why
they work, and this page assumes it.

All of it on one RTX 3090, Qwen3.8-27B at Q4_K_S, llama.cpp `b10715`. Three
workloads throughout — **prose**, a **code-edit**, and **long-copy** (369 lines
copied back with one identifier renamed). Every ratio is against its own run's
`baseline` config, which is no speculation at all.

**Most levers were measured twice, on Vulkan and on CUDA, and several verdicts
differ by backend.** Vulkan is what installs. A reader who acts on a CUDA number
while running the shipped build gets the Vulkan one.

## Every lever

| lever | moves | on Vulkan (shipped) | on CUDA (opt-in) | verdict |
|---|---|---|---|---|
| Multi-token prediction | decode | **1.61–1.82×** | **1.96–2.22×** | **on by default** — worth more on CUDA |
| q8_0 KV cache | capacity | 2× the conversation | same | **on by default** |
| q8_0 *draft* cache | capacity | +2.45 KiB/token back | not measured | **on by default** — free |
| Flash attention | both | not separately measured | not separately measured | **on by default** |
| CUDA instead of Vulkan | both | — | 1.31× prefill, 1.28× decode bare, **~1.6×** as shipped | opt-in — costs a toolkit and a per-card binary |
| DFlash2 drafter | decode | 0.90–1.04× *vs MTP* | 0.92–1.01× *vs MTP* | rejected on **both** — a wash, for 36k tokens of context |
| Draft width 7 | decode | 0.78–0.98× *vs width 3* | 0.83–1.22× *vs width 3* | rejected on Vulkan; on CUDA, copy-heavy only |
| Prompt lookup (ngram) | decode | 0.65–0.88× | 0.92–1.41× | rejected on Vulkan; **wins copying on CUDA** |
| ngram stacked on MTP | decode | 0.84–0.90× *vs MTP* | 0.86–1.10× *vs MTP* | rejected on Vulkan; **fastest copying config on CUDA** |
| q4_0 KV cache | capacity | 4× the conversation, ~0.63× decode | not measured | not taken — never measured here |
| coopmat2 | prefill | unknown | n/a — a Vulkan concept | undeterminable from this build |
| TurboQuant KV | capacity | claimed ~2.5× vs q8_0 | not merged | parked — not merged anywhere |
| vLLM / AWQ-Marlin | throughput | 8× at ten concurrent users | a different engine | out of scope — this engine serves one |

**Nothing has displaced multi-token prediction on the backend that ships.** On
CUDA that stops being true: two of the three rejections come back, MTP itself
grows from 1.6× to 2.0×, and the fastest copy-heavy configuration is one this
page had written off.

## What to run

**On the engine that installs today, change nothing.**

```bash
lllm3090 start Qwen3.8-27B
```

MTP on from the checkpoint's own tensors, q8_0 cache, flash attention on, every
layer on the GPU. Nothing here beats it on Vulkan — it is already at the card's
memory roofline. The three tempting additions are all regressions: `ngram-cache`
0.65×, `--spec-draft-n-max 7` 0.78×, DFlash2 a wash costing ~36k tokens.

**For the 1.6×, the engine has to change, not the flags.** Build one with
`lllm3090 build-cuda`, then:

```bash
LLLM3090_LLAMA_DIR=~/.local/share/lllm3090/engines/b10715-cuda-sm86 \
  lllm3090 start Qwen3.8-27B
```

That reaches the engine only when the **CLI** starts it. The panel starts the
engine in its own process and reads the systemd unit's environment, so a model
started from the browser still gets Vulkan.

**For copy-heavy sessions on CUDA** — refactoring, renaming, reformatting:

```bash
lllm3090 start Qwen3.8-27B --profile copy
```

`--spec-type draft-mtp,ngram-cache --spec-draft-n-max 7`. It costs ~14% on prose
and code editing, so it is a per-session choice. **Refused on Vulkan**, where
the same additions are 0.84–0.90× against MTP alone.

## The fastest this model goes

Decode has a memory roofline: ~16 GB of weights per token at 936 GB/s is
**roughly 58 tok/s**, and nothing reads memory faster than memory goes.
Speculation is the one lever that beats it, because an accepted draft rides
along on a read that already happened.

| | prose | code-edit | long-copy | peak vs roofline |
|---|---|---|---|---|
| memory roofline, no speculation | ~58 | ~58 | ~58 | 1.00× |
| Vulkan, no speculation | 34.1 | 33.8 | 32.8 | 0.59× |
| **Vulkan + MTP — what installs today** | **54.8** | **55.7** | **59.7** | 1.03× |
| CUDA, no speculation | 43.2 | 43.6 | 42.4 | 0.75× |
| **CUDA + MTP** | **84.9** | **89.0** | **94.0** | 1.62× |
| CUDA + MTP + ngram, width 7 | 72.9 | 76.0 | **115.1** | **1.98×** |
| CUDA vs Vulkan, *same flags* | 1.55× | 1.60× | 1.57× | |
| CUDA vs Vulkan, *best config either side* | 1.55× | 1.60× | **1.93×** | |

**115.1 tok/s is the best measured result on this card**, at just under twice
what it can do without guessing. Two rows exceeding the roofline is not an
anomaly — it is the definition of speculative decoding working.

:::{warning}
**Every figure above is a short-prompt figure.** Fill the 168k window and
CUDA + MTP holds about **38.7 tok/s**. See the depth table below.
:::

### What depth costs

One server at 172032, prompts from 4k to 158k, three samples each, on CUDA.

| depth | prefill | baseline | + MTP | MTP is worth |
|---|---|---|---|---|
| 3.7k | 1314/s | 42.2 | 66.0 | 1.56× |
| 7.4k | 1298/s | 40.6 | 80.4 | 1.98× |
| 14.7k | 1267/s | 39.4 | 63.8 | 1.62× |
| 33k | 1145/s | 33.2 | 58.7 | 1.77× |
| 62k | 974/s | 25.6 | 57.2 | 2.23× |
| 95k | 847/s | 19.4 | 46.3 | 2.39× |
| 125k | 777/s | 17.8 | 44.3 | 2.49× |
| **158k** | **695/s** | **13.6** | **38.7** | **2.85×** |

- **Speculation is worth more at depth, not less** — 1.6× shallow to 2.85× at a
  full window. A deeper cache makes every forward pass costlier, and verifying
  k drafts amortises one costly pass across k tokens.
- **It halves the depth penalty.** Bare decode collapses to 0.32× across the
  window (42.2 → 13.6); with MTP the same span is 0.55× (~70 → 38.7).
- **Prefill halves and speculation does nothing for it**: 1314 → 695 without
  drafting, 1179 → 664 with. A wider prompt is still a wider prompt.

:::{note}
Each run re-measured its shallowest depth last, as a control — a card slowing as
it heats reads exactly like a cache slowing as it fills. Controls came back at
69.9 against 66.0 (MTP, *faster* at 79 °C) and 41.2 against 42.2 (baseline, 2.4%
down), so the decline is depth. The `+ MTP` column carries content noise the
baseline does not: acceptance swings 61–84% across these rows with no trend in
depth, which is why 7.4k reads 80.4.
:::

## What is already in the engine

The baseline everything else is measured against — see `lllm3090.engine.start`.

| | flag | why |
|---|---|---|
| Multi-token prediction | `--spec-type draft-mtp` | Added from the checkpoint's tensors, never a catalogue field: llama.cpp refuses to start with it against a checkpoint lacking the head. 34.9 → 56.6 tok/s (1.62×) on the dense 27B, 130.5 → 171.8 (1.32×) on the sparse 35B. |
| q8_0 KV cache | `--cache-type-k/v q8_0` | Halves KV against f16; what puts this model's window in reach at all. |
| q8_0 draft cache | `--spec-draft-type-k/v q8_0` | A separate cache with separate flags that does **not** inherit from the pair above. See below. |
| All layers on GPU | `--n-gpu-layers 999` | Nothing in the catalogue is allowed to need offload. |
| Flash attention | `-fa on` | Not left at `auto`. |
| Whole windows first | — | `catalog.plan` fills one conversation to the ceiling before opening a second. See [](../how-to/context-and-slots.md). |

**q4_0 KV** would save another 72% of cache but costs roughly 37% of decode at
long context — dequantisation is on the critical path — so it is not taken. That
figure is a community number this project has never measured.

## Prompt lookup: rejected on Vulkan, a win on CUDA

The one verdict on this page that flips with the backend. Same commit, same
prompts, both engines; acceptance in brackets where measured.

| | prose | code-edit | long-copy |
|---|---|---|---|
| **Vulkan** baseline | 34.1 | 33.8 | 32.8 |
| `draft-mtp` | 54.8 (1.61×) | 55.7 (1.65×) | **59.7 (1.82×)** |
| `ngram-cache` | 22.2 (0.65×) | 26.2 (0.78×) | 28.9 (0.88×) |
| `draft-mtp,ngram-cache` | 46.2 (1.35×) | 50.0 (1.48×) | 52.7 (1.61×) |
| **CUDA** baseline | 43.2 | 43.6 | 42.4 |
| `draft-mtp` | 84.9 (1.96×, 85%) | 89.0 (2.04×, 86%) | 94.0 (2.22×, 100%) |
| `ngram-cache` | 40.2 (0.93×, 36%) | 39.9 (0.92×, 1%) | **59.6 (1.41×**, 57%) |
| `draft-mtp,ngram-cache` | 72.5 (1.68×, 70%) | 76.7 (1.76×, 77%) | **103.5 (2.44×**, 86%) |

- **Prompt lookup wins the workload it was built for**: 0.88× on Vulkan,
  **1.41×** on CUDA, at essentially the same acceptance (58% against 57%). The
  drafts are as good as they always were; the backend stopped charging full
  price to check them.
- **Stacking beats MTP alone on copying** on CUDA — 2.44× against 2.22×. The
  displacement is still visible (acceptance 100% → 86%, as on Vulkan), but the
  extra drafts are cheap enough that winning some beats keeping acceptance
  perfect.
- **Prose and code editing stay losses** on both. Prompt lookup did not become a
  good idea; it became a good idea *for copying*.

Two objections were answered on Vulkan and neither rescued it. The workload was
wrong — `long-copy` lifts acceptance from **0% to 58%** — and it is still 0.88×.
And the build was broken in a way that had been *flattering* ngram: on `b10628`
prose read 0.87× at 47% acceptance, and on `b10715` — which carries [PR
#27812](https://github.com/ggml-org/llama.cpp/pull/27812) — the same sweep reads
**0.65× at 24%**. The target had been accepting drafts it never chose. **MTP's
numbers did not move** (1.62× → 1.61×), because its drafts are short and mostly
right. That is why the pin moved: the engine everyone installed was the one
producing invalid acceptance numbers, so anything measured on it had to be
discounted.

:::{note}
Two CUDA cells should not be leaned on. `draft-mtp,ngram-cache` on code-edit
spanned 70.3–92.5 tok/s across seven samples — a 32% spread — so its 1.76× is
soft. And `ngram-cache` on prose reads 36% acceptance where Vulkan read 24%;
acceptance is a property of the drafts rather than the backend, so that gap is
unexplained. Nothing above rests on either.
:::

## DFlash2: rejected on both

[DFlash2](https://inco.ai/blog/dflash2/) is a block-diffusion drafter published
for this exact model, with large published figures: 2.26× on an RTX PRO 6000,
3.55× on a 36k synthetic sweep, 1.85× on an M5 Pro. Width 3, against each run's
own baseline.

| | prose | code-edit | long-copy |
|---|---|---|---|
| **Vulkan** baseline | 32.8 | 32.4 | 31.6 |
| `draft-mtp` | **52.7** (1.61×, 85%) | 52.0 (1.61×, 84%) | **57.0** (1.80×, 100%) |
| `dflash` | 51.0 (1.55×, 80%) | **54.0** (1.67×, 85%) | 51.3 (1.62×, 86%) |
| **CUDA** baseline | 43.2 | 43.6 | 42.4 |
| `draft-mtp` | **84.9** (1.96×, 85%) | 89.0 (2.04×, 86%) | **94.0** (2.22×, 100%) |
| `dflash` | 83.8 (1.94×, 80%) | **90.3** (2.07×, 88%) | 86.1 (2.03×, 85%) |

A win of 4% on code editing and losses of 3% and 10% on Vulkan; 1%, 1% and 8% on
CUDA. Same shape, slightly flatter — for **1.1 GB of VRAM** that MTP costs
nothing for. At 32 KiB/token that is ~36k tokens, taking this model's session
from ~172k to ~136k. Paying a fifth of the context for a wash is not a trade.

The published multipliers are not wrong, they answer a different question: they
are all against **no speculation at all**, and this engine has run MTP since the
head was detected automatically. Against that floor DFlash2's own claim was
always narrow — 0.52 tokens of acceptance length. It was never losing to the
backend; it was losing to MTP, which is free, and which a better backend speeds
up too. On CUDA it would edge code editing to 91.7 tok/s, and that is left out
of the headline because 1.1 GB of drafter is not free.

## Draft width belongs to the backend, not the drafter

The first run of this comparison swept DFlash2 at its published width of 7
against MTP at llama.cpp's default of 3 — two variables, reading as a verdict on
the drafter. On Vulkan, at width 7:

| Vulkan, width 7 | prose | code-edit | long-copy |
|---|---|---|---|
| `draft-mtp` | 41.3 (63%) | 49.4 (71%) | 55.7 (97%) |
| `dflash` | 43.9 (67%) | 48.2 (70%) | 35.7 (52%) |

Widening 3 → 7 costs MTP 22% on prose and DFlash2 14%, with acceptance falling
85% → 63% and 80% → 67%. So DFlash2's own README invocation
(`--spec-draft-n-max 7`) is actively harmful here. [Issue
#27544](https://github.com/ggml-org/llama.cpp/issues/27544) reports the same
shape on AMD and Intel above width 8, explicitly *not* on CUDA — so the effect
is wider than the issue describes.

On CUDA the penalty is a third of the size. Widening 3 → 7:

| width 3 → 7, CUDA | prose | code-edit | long-copy |
|---|---|---|---|
| `draft-mtp` | **0.92–0.93×** | 0.83–0.87× | **1.13–1.22×** |
| `dflash` | 1.01× | 1.02–1.06× | 0.83–0.90× |
| `draft-mtp,ngram-cache` | 1.00–1.01× | 0.99–1.04× | **1.11–1.19×** |

**Acceptance falls by the same amount on both backends and the speed does not.**
MTP on prose goes 85% → 64% on CUDA against 85% → 63% on Vulkan — the drafter
throws away just as many drafts. Vulkan charges 22% of decode for that; CUDA
charges 8%. Same wasted drafts, a third of the bill. Acceptance belongs to the
drafter, the price of a wide verify belongs to the backend, and here they
separate cleanly.

**Where acceptance barely falls, width 7 wins.** On `long-copy` MTP holds 96% at
width 7 and CUDA turns that into **13–22%** — 106.5 tok/s against 94.0, and
115.1 stacked with ngram. Vulkan, at 97% acceptance on the same workload, still
lost.

So: **take the default 3 on Vulkan** (the engine passes no width, so it already
does). **On CUDA, 3 for prose and code editing, 7 for copy-heavy work** — which
is a per-workload knob rather than a default, and is why it is `--profile copy`
rather than something switched on automatically.

:::{note}
Each cell above is a span because the two defensible ways to compute it
disagree: against the baseline measured in the same run, or raw tok/s of one run
over the other. Reading per-request rates out of the server logs shows why — the
width-3 run *ramps up* off idle while the width-7 run, started a minute later,
**declines 9% over ten minutes** of sustained load. Same binary, same flags. So
that run's `long-copy` baseline of 39.5 is the trough of its own decline and
flatters every ratio taken against it. Nothing changes sign under either
reading, but the copy row is known to about ±5% and no better.
:::

## CUDA against Vulkan

Both engines from llama.cpp commit `662a0b0` — the same commit as the pinned
Vulkan build, so this is a backend comparison and nothing else — with CUDA 13.3
compiled for `sm_86` alone, derived from the card rather than typed.

| cold prefill, empty cache | Vulkan | CUDA | |
|---|---|---|---|
| 10k tokens | 10.5 s | 8.1 s | 1.30× |
| 40k tokens | 48.5 s | 37.1 s | 1.31× |
| 80k tokens | **118.2 s** | **90.0 s** | 1.31× |

| `llama-bench` | Vulkan | CUDA | |
|---|---|---|---|
| pp512 | 1026.9 | 1217.4 | 1.19× |
| pp4096 | 1014.0 | 1343.8 | 1.33× |
| tg64 | 32.9 | 42.2 | **1.28×** |

- **The 3–4× prefill claim this page used to make was wrong.** It is 1.3×. What
  was right is the number that made it plausible: 80k really does take two
  minutes — but that is what this card costs to prefill 27B dense, not a tax the
  backend charges. The fix for a slow first turn is a shorter prompt or a warm
  cache, not a backend.
- **Decode was the surprise.** Community scoreboards put CUDA ~10% ahead on
  generation; here it is **28%**.
- **Vulkan does not batch.** 1026.9 → 1014.0 from pp512 to pp4096, where CUDA
  goes 1217.4 → 1343.8. That single line is the mechanism behind every
  draft-width and ngram result above: verifying k drafts *is* a batched forward
  pass, so a backend that gains nothing from a wider batch punishes wider drafts.
- **The effects multiply rather than add**, which is why the headline is 1.6× and
  not 1.3×. MTP is worth 1.61–1.82× on Vulkan and 1.96–2.22× on CUDA at
  acceptance rates matching within a point (85/86/100% against 85/84/100%).
  Identical drafts, identical accepts, more speed.

### What CUDA costs

**In context.** Loading the dense 27B at rising context until the server dies,
desktop session running, `--parallel 1`, q8_0 KV both ways:

| ctx | CUDA | Vulkan |
|---|---|---|
| 40960 | 18300 MiB | 17914 MiB |
| 172032 | **23902 MiB** | 23009 MiB |
| 176128 | **dies** | — |
| 188416 | dies | 23645 MiB |
| 200704 | dies | **24121 MiB** |
| 204800 | dies | dies |

**CUDA's ceiling is ~28k tokens lower**, from two costs: **~230 MiB more fixed
overhead** (roughly the 24125-against-24822 MiB of reported capacity), and
**~10% more per token of KV** — the resident slope is 43.8 KiB/token against
39.8. The second compounds with depth and is the larger at a full window.

**In everything else:**

- **A 4–6 GB toolkit and a local compile.** llama.cpp publishes CUDA archives
  for Windows only; Linux gets CPU, Vulkan, ROCm, SYCL and OpenVINO.
- **Ubuntu's own CUDA cannot build it.** 26.04 ships 13.1, which declares
  `rsqrt`/`rsqrtf` without an exception specifier while glibc 2.43 declares them
  `noexcept(true)`; `nvcc` then refuses anything including `<math.h>`. 13.3
  tests for glibc ≥ 2.42 and matches it. It has to come from NVIDIA's
  `ubuntu2604` repository.
- **A binary tied to one card.** `sm_86-real` will not run on another
  architecture at all, so the directory is named `b10715-cuda-sm86`.
- **A weaker identity.** A downloaded build is a tag and a digest recorded in
  this repository, reviewed in a diff and outside the control of whoever serves
  the bytes — the only check that can notice a tag whose asset was replaced. A
  compiled one reports `build 1, commit 662a0b0`, because a shallow clone has no
  tag history: only the commit is real, and nobody attests to the binary. That
  is why `lllm3090 build-cuda` never switches to what it builds.

So the question is not whether CUDA is faster — it is — but whether **1.6× on an
ordinary turn and up to 1.9× on refactoring work** is worth asking a user for a
toolkit and a per-card binary, in a project whose install is one verified
download. That is a decision about the install promise, and the measurement does
not make it. It is a harder question than the one asked before the drafters were
re-measured (*is 1.3× worth a toolkit?*), which was weighing a comparison of two
backends running no speculation while the engine has never run that way.

### What the planner does about it

`config.KV_OVERHEAD_FACTOR` was one constant of `1.12`, calibrated on
Gemma-4-26B-A4B — a Vulkan engine running a model with no MTP head. It gave both
backends the same window, and on CUDA that window was *exactly* the measured
ceiling: 172032 loads with 674 MiB free. Not a failure, but no margin at all, on
a machine whose desktop was observed moving 100 MiB in an afternoon.

Resident KiB/token against each model's nominal q8 figure:

| | nominal | Vulkan, no MTP | Vulkan, MTP | CUDA, no MTP | CUDA, MTP |
|---|---|---|---|---|---|
| Qwen3.8-27B | 32 | 35.00 (1.094×) | 39.80 (1.244×) | 39.00 (1.219×) | 43.77 (1.368×) |
| Qwen3.6-35B-A3B-MTP | 10 | 12.15 (1.216×) | 14.62 (1.462×) | 13.63 (1.363×) | 16.60 (1.660×) |

**One constant cannot express that — but not for the reason those figures
suggest.** On one of these models "KiB per token" is not a well-defined quantity
at all. Every figure above is a slope between two context sizes, which only
means something if resident VRAM is linear in context. With mid-points added,
Vulkan, no speculation:

| segment | Qwen3.8-27B | Qwen3.6-35B-A3B-MTP |
|---|---|---|
| low | 34.92 | **8.99** |
| mid | 35.08 | 11.57 |
| high | — | **16.78** |
| end to end | 35.00 | **12.15** |

The dense model is linear to within 0.5%. The hybrid MoE's marginal cost
**nearly doubles across its range**, so its end-to-end 12.15 is an average over a
curve that depends entirely on which two points produced it — an error worth
~560 MiB against a desktop that moves ~100 MiB. **Only the backend multiplier
generalises**: CUDA costs 1.100–1.136× across both models and both speculation
settings, because it is a ratio of two measurements taken the same way and the
curvature divides out.

Most of the rest is arithmetic. Both models are hybrids
(`full_attention_interval = 4`, so one layer in four keeps a cache) and
`block_count` **includes the MTP head**, which is why it reads 65 for a 64-layer
model:

| from the GGUF header | full-attn layers | nominal f16 | `models.yaml` |
|---|---|---|---|
| Qwen3.8-27B | 16 | 64.0 | 64 ✓ |
| Qwen3.6-35B-A3B-MTP | 10 | 20.0 | 20 ✓ |

`kv_heads × (key_length + value_length) × 2 bytes × full-attention layers`
reproduces the catalogue exactly, and the MTP term stops being a mystery: **the
head is one more full-attention layer**, predicting 4.00 and 2.00 KiB/token
against 4.80 and 2.46 measured — a consistent ×1.20 and ×1.23.

So the constant is now the terms it was covering:

| term | value | from |
|---|---|---|
| `Q8_0_RATIO` | 0.53125 | arithmetic: 34 bytes per 32 values, **not half of f16** |
| MTP's cache | one full-attention layer, ×1.25 | the header, and ×1.20/×1.23 measured |
| `BACKEND_KV_FACTOR["cuda"]` | 1.136 | top of the measured 1.100–1.136 range |
| `BACKEND_FIXED_MIB["cuda"]` | 230 | flat, so off the budget not the slope |
| `ALLOCATOR_OVERHEAD` | 1.12 ÷ 2 ÷ 0.53125 | the Gemma measurement, re-attributed |

**That last row is the one to read twice.** Correcting `Q8_0_RATIO` must *not*
make everything 6% more conservative, because 1.12 was measured *through* the
same error — Gemma's resident 11.2 KiB/token over a nominal of 10 that should
have been 10.625. The measurement was right and the attribution was not. Stack
the correction instead of re-attributing and Gemma is priced 6% above what it was
observed to cost.

The result has a narrow blast radius: **every model with no MTP head is priced
exactly as before**, and the only entries that move are the two that were never
paying for their draft cache. Qwen3.8-27B goes 168k → 156k on Vulkan and 132k on
CUDA; the 35B-A3B-MTP does not move, being capped by RoPE rather than VRAM.

What this does **not** do is fit the hybrid's curve. `fit` still divides a budget
by a per-token cost, which is right for a linear model and wrong for a curved
one, so the budget is kept conservative rather than tuned to a slope.

:::{note}
This section has been rewritten twice by measurement. It first claimed a tidy
three-term model derived from one model; the second model broke two of its three
terms. It then claimed the overhead factor was model-dependent; adding mid-points
showed that figure was an average over a curve. Both are recorded rather than
quietly fixed, because the pattern is the point — **a slope between two
measurements looks like a constant until you measure a third.**
:::

### The draft cache was never quantised

The draft model keeps its own cache with its own flags — `--spec-draft-type-k`
and `--spec-draft-type-v` — which do **not** inherit from `--cache-type-k/v`.
The engine set only the second pair, so MTP's context ran at its `f16` default
beside a main cache at `q8_0`. That was the whole of the MTP term.

| dense 27B, Vulkan, MTP on | resident KiB/token | loads at | dies at |
|---|---|---|---|
| draft cache at its default | 40.18 | 200704 | 204800 |
| draft cache `q8_0` | **37.73** | **208896** | 212992 |

**2.45 KiB/token back against an MTP term of 4.80** — almost exactly half, which
is what f16 → 8 bits should give and is the evidence for what the default really
was. About **412 MiB at a 168k window**, and 4k–12k more tokens of ceiling.

It costs nothing:

| | default | `q8_0` | |
|---|---|---|---|
| ~4k prompt | 43.7 tok/s, 65% | 44.1, 66% | 1.007× |
| ~62k prompt | 32.3 tok/s, 64% | **33.3, 71%** | **1.031×** |

**Read the acceptance column as "no penalty" rather than as a gain** — seven
points is inside what content alone swings on this instrument, which spans
61–84% on one unchanged config in the depth table above. It cannot change output
either way, because every draft is verified whatever cache produced it.

The engine now passes both, from the same constant as the main cache, so they
cannot drift apart again. This is a lever on the **shipped Vulkan engine** and
has nothing to do with CUDA.

## What the CUDA re-measurement settled

This page had committed to a falsifiable prediction — that verdicts measured on
Vulkan should not be treated as settled for CUDA — so it is obliged to say how
that did, not to quote it only if it landed. Nine config-runs, both widths, same
commit, same prompts, same protocol.

| the page predicted | outcome |
|---|---|
| ngram's rejection is not settled for CUDA | **right** — 0.88× → 1.41× on copying, a flipped verdict |
| stacking's rejection is not settled | **right** — now the fastest copying config, 2.44× against 2.22× |
| the width penalty belongs to the backend | **right** — same acceptance collapse, a third of the speed cost |
| DFlash2's rejection is not settled | **wrong** — 0.92–1.01× on CUDA against 0.90–1.04× on Vulkan. Unchanged. |
| (unstated) MTP's own multiplier would carry over | **wrong** — 1.61–1.82× on Vulkan, 1.96–2.22× on CUDA |

The DFlash2 miss is the useful one: sound reasoning that predicted the wrong
result. What it missed is that DFlash2 was never losing to the backend, it was
losing to MTP — which is free, and which a better backend speeds up too. **A
lever measured against a moving comparator does not gain on it just because both
get faster.**

The unstated assumption is the more expensive error, because it was doing work in
a decision. "1.3× is probably not worth a toolkit" was reasoning about a number
the engine has never run at.

## Unanswered and parked

**coopmat2.** Whether this build uses `VK_KHR_cooperative_matrix` version 2 or
falls back to coopmat1 was worth 2.2× against 4.4× on prefill in the original
benchmarks, and is free if it is merely switched off. Unmeasurable here:
`--list-devices` names the device and not the extension, and b10715's server log
prints no `ggml_vulkan:` banner at all — 30 KB of load log with no mention of the
GPU, the backend or the matrix cores. `vulkaninfo` is not installed.

**TurboQuant KV.** Sub-3-bit KV via randomised Hadamard transforms — 3.25
bits/value at ~4.9× compression against f16, negligible reported quality loss,
and the dequantisation penalty that rules out q4_0 designed out rather than
tolerated. For a model at 64 KiB/token this would be transformative. **Not
merged**: several independent forks, CUDA and Metal validated, Vulkan still in
development. Watch [discussion
#20969](https://github.com/ggml-org/llama.cpp/discussions/20969).

**vLLM or SGLang with AWQ-Marlin.** The headline is real — Marlin kernels at 741
tok/s against GGUF's 93 — but it is a *concurrency* number at ten or more
simultaneous users, with a higher time-to-first-token under load. This engine
serves one person. DFlash2 also runs in vLLM, SGLang and TensorRT-LLM, so it is
not an argument for switching either.

## How these numbers were taken

Every speed claim here is held to the same standard as a catalogue entry marked
`verified: true`: a sweep, a warm-up discarded after every server start, **seven
timed samples** per prompt per config, a unique prefix and `cache_prompt=false`
on every generation, and the served model checked either side of the run. Three
samples once made MTP read as no gain at all when it was 1.32×.

The instrument is `dev/spec-sweep.py` — a development tool rather than part of
the package, since it starts and stops a server per config. It launches with the
same flags `lllm3090.engine.start` uses, which is the point. It runs on its own
port, stops only servers it started and only by pid, and refuses to run while the
engine is up, since two engines on one card measures the contention.

```bash
SWEEP_DRAFT=~/models/.drafters/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  dev/spec-sweep.py ~/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_S.gguf

lllm3090 fetch-engine --build b10800      # measure a candidate beside the install
SWEEP_BUILD=b10800 dev/spec-sweep.py MODEL.gguf
```

Four things that have each cost a measurement here:

- **Keep `baseline` in a restricted sweep.** The config list exists so a second
  width need not re-run everything, and the baseline looks like a known
  quantity. It is not: across two CUDA runs 35 minutes apart on an idle card it
  moved 0.5% on prose and **6.8% on `long-copy`**.
- **A baseline in the same run does not cancel drift *through* it.** The
  baseline config runs first, and the card it measures is not the card the later
  configs get — one run gained 3% as clocks ramped, another lost 9% over ten
  minutes, both the identical invocation. Randomising config order, or
  re-measuring `baseline` at the end as well, would bound it. Neither is
  implemented, so **do not start a sweep on a card that has just finished one**.
- **Within a backend, 4–7% is the noise floor** and the same size as several
  effects being judged. Ratios taken inside one run are much tighter — seven
  samples of the width-3 `long-copy` baseline spanned 42.36 to 42.46 — which is
  why the tables never compare across runs when they can avoid it.
- **Keep a drafter out of the models directory.** `catalog.installed` takes the
  alphabetically first non-projector GGUF in each directory, so a drafter beside
  the weights it drafts for sorts ahead of them and quietly becomes the model the
  panel starts. `~/models/.drafters/` is skipped as a dot-directory, which is a
  convention and not a guarantee.

The baselines every ratio on this page divides by, shown directly rather than
left implied:

| baseline, tok/s | prose | code-edit | long-copy |
|---|---|---|---|
| Vulkan, ngram run | 34.1 | 33.8 | 32.8 |
| Vulkan, DFlash2 run | 32.8 | 32.4 | 31.6 |
| CUDA, width-3 run | 43.2 | 43.6 | 42.4 |
| CUDA, width-7 run | 43.4 | 41.6 | 39.5 |

**Across backends the number is solid.** All twelve pairings of a CUDA run
against a Vulkan one span **1.20–1.35×, mean 1.285×** — against `llama-bench`'s
independent 1.283× on tg64. Two harnesses, three workloads, one answer. That
agreement is what licenses the cross-run absolute comparisons above, including
the 1.55–1.60× headline; without it they would be two unrelated tables.
