# Speed levers, tried and untried

The dense 27B is the slowest model in the catalogue, and
[structurally so](dense-vs-moe.md) — it reads every one of its parameters per
token. That is not the whole story: multi-token prediction has already taken it
from 34.9 to 56.6 tok/s, and several further levers have been looked at. This
page records all of them — what is already in the engine, what was measured and
set aside, what is waiting on a build, and what is parked and why.

Two of them have since been measured and neither displaced MTP, so the short
version is that the engine's defaults are already the best combination tried on
this card. The value in what follows is mostly in why, and in the one variable —
draft width — that turned out to matter more than the choice of drafter.

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
  head. Measured on this card, on b10628: **34.9 → 56.6 tok/s (1.62×)** on the
  dense 27B, 130.5 → 171.8 (1.32×) on the sparse 35B. The tables below
  re-measure the dense model on b10715 and land in the same place, which is one
  of the reasons the pin was allowed to move.
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

**The pin moved because of this.** It was `b10628` (25 August), three days older
than [PR #27812](https://github.com/ggml-org/llama.cpp/pull/27812) — so the
engine everyone installed was the one that produced invalid acceptance numbers,
and any measurement taken on it had to be discounted. `engines.LLAMA_BUILD` is
now `b10715`, which is also what the sweep defaults to, so a sweep and the
served engine are the same build again.

## Measured and not adopted: DFlash2

[DFlash2](https://inco.ai/blog/dflash2/) is a block-diffusion drafter published
for this exact target model. It predicts a whole block of tokens per forward
pass and keeps the top candidates at every position, and its published figures
are large: 2.26× on an RTX PRO 6000 under CUDA, 3.55× on a 36k synthetic sweep,
1.85× on an M5 Pro. Output is unchanged.

It does not beat the MTP head this engine already uses. Measured here on
b10715, at llama.cpp's default draft width of 3, against a baseline of
32.8 / 32.4 / 31.6 tok/s:

| | prose | code-edit | long-copy |
|---|---|---|---|
| `draft-mtp` | **52.7** (1.61×, 85%) | 52.0 (1.61×, 84%) | **57.0** (1.80×, 100%) |
| `dflash` | 51.0 (1.55×, 80%) | **54.0** (1.67×, 85%) | 51.3 (1.62×, 86%) |

A win of 4% on code editing, a loss of 3% on prose and 10% on copying — for
1.1 GB of VRAM that MTP costs nothing for. At 32 KiB/token of q8_0 KV that is
roughly 36k tokens, taking the dense 27B's session from about 172k to about
136k. Paying a fifth of the context for a wash is not a trade worth making, so
the catalogue does not carry it.

The published multipliers are not wrong, they are answering a different
question: they are all against **no speculation at all**, and this engine has
run MTP since the head was detected automatically. Against that floor, DFlash2's
own claim was always narrow — 0.52 tokens of acceptance length — and on this
backend it does not survive.

### Draft width matters more than which drafter

The first run of this comparison was wrong, in a way worth recording. It swept
DFlash2 at its published width of 7 against MTP at llama.cpp's default of 3,
because `--spec-draft-n-max` was set per-config rather than per-sweep. That is
two variables, and it reads as a verdict on the drafter:

| at width 7 | prose | code-edit | long-copy |
|---|---|---|---|
| `draft-mtp` | 41.3 (63%) | 49.4 (71%) | 55.7 (97%) |
| `dflash` | 43.9 (67%) | 48.2 (70%) | 35.7 (52%) |

Widening from 3 to 7 costs MTP 22% on prose and DFlash2 14%, and acceptance
falls for both — 85% → 63% and 80% → 67%. **The penalty is the backend's, not
the drafter's**, which is why the width is now applied to every config by the
sweep rather than chosen by each. It also means the invocation in DFlash2's own
README is actively harmful here:

```
--spec-type draft-dflash --spec-draft-model DRAFTER.gguf --spec-draft-n-max 7
```

Take the default 3 on Vulkan. The engine passes no width at all and therefore
already gets it.

The one place the drafter really does differ is copying: at width 7 DFlash2's
acceptance on `long-copy` collapses to 52% while MTP holds 97%. Block drafting
is worst at exactly the workload where the next tokens are least in doubt.

[Issue #27544](https://github.com/ggml-org/llama.cpp/issues/27544) reports the
same shape on AMD and Intel above width 8, explicitly not on CUDA. These numbers
are below that threshold and on NVIDIA, so the effect is wider than the issue
describes.

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
and cheap to check once a CUDA build exists: **every drafter verdict above may
change, and the wide-draft ones should change most.** Verifying k drafted tokens
is a batched forward pass — the same work prefill does — and prefill is
precisely where Vulkan is 3–4× behind here.

That is no longer only an argument. Widening the draft from 3 to 7 costs both
drafters throughput *and* acceptance on this backend, which is the signature of
verification being expensive rather than of drafts being poor, and issue #27544
reports the same shape on AMD and Intel but explicitly not on CUDA. If the
penalty is the backend's, a CUDA build should flatten it — which would make
wider drafts affordable, and would give prompt-lookup, whose whole method is
drafting long runs, its only plausible route back. Nothing has been measured on
CUDA to support this.

So the obstacle is no longer *can it build* but *what the project promises*.
`install-engine` currently downloads a tarball and is done; a CUDA path means
either a multi-gigabyte toolkit and a local compile on every install, or
shipping binaries built here — a different project with a different support
burden. Worth measuring the gain before deciding, since the gain is the whole
argument.

## Still unanswered: coopmat2

Whether this build uses `VK_KHR_cooperative_matrix` version 2 or falls back to
coopmat1 is worth knowing before any of that, because the difference was 2.2×
against 4.4× on prefill in the original benchmarks and it is free if it is
merely switched off.

It is also, so far, unmeasurable here. `--list-devices` names the device and not
the extension, and b10715's server log does not print a `ggml_vulkan:` banner at
all — 30 KB of load log on this machine contains no mention of the GPU, the
backend or the matrix cores. The sweep looks for that line and prints it when it
finds one, which on this build is never. `vulkaninfo` would at least say what
the driver exposes, and is not installed.

## Parked

Neither of these is a bad idea; both are blocked on something outside this
project.

**TurboQuant KV cache.** Sub-3-bit KV via randomised Hadamard transforms —
3.25 bits/value at ~4.9× compression against f16, with quality loss reported as
negligible, and with the dequantisation penalty that rules out q4_0 here
designed out rather than tolerated. For a model at 64 KiB/token this would be transformative.
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
part of the installed package — it starts and stops a server per config, which
is the opposite of what the engine lifecycle promises — but it launches with the
same flags `lllm3090.engine.start` uses, and that correspondence is the point.
It runs on its own port, stops only the servers it started and only by pid, and
refuses to run at all while the engine is up, since two of them on one card
measures the contention rather than the model. Its `baseline` config is no speculation at all, so MTP appears as a
config rather than as the floor. Point `SWEEP_DRAFT` at a drafter GGUF to
include DFlash2, and `SWEEP_NMAX` at a draft width, which applies to every
draft-model config so that a comparison of drafters is not also a comparison of
widths. The n-gram modes are sized by their own `--spec-ngram-*` knobs and keep
their defaults:

```
SWEEP_DRAFT=~/models/.drafters/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  dev/spec-sweep.py ~/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_S.gguf
```

Keep a drafter out of the models directory. `catalog.installed` takes the
alphabetically first non-projector GGUF in each directory, so a drafter dropped
beside the weights it drafts for would sort ahead of them and quietly become the
model the panel starts. `~/models/.drafters/` is skipped along with every other
dot-directory, which is a convention and not yet a guarantee.

It never sweeps the installed engine. Builds to measure are fetched beside it,
one directory per upstream tag:

```
lllm3090 fetch-engine --build b10800
SWEEP_BUILD=b10800 dev/spec-sweep.py MODEL.gguf
```

That separation is what makes an upgrade decidable at all: choosing between
b10628 and b10715 meant running both, which is impossible if measuring one
installs it over the other. It is how the pin moved. `SWEEP_LLAMA_DIR` remains as an escape hatch for a
build with no tag to name it by — a locally compiled CUDA engine, when that
happens. Every run prints the build number and the device line it found,
because two tables of tokens per second that do not say what produced them
cannot be compared later — which is precisely how the earlier numbers stopped
being usable.

## What a build has to prove

Every engine this project runs or measures is fetched by tag and verified, and
the two digests involved are not the same claim.

`engines.LLAMA_SHA256` is recorded in the repository: reviewed in a diff, and
outside the control of whoever serves the bytes. It is therefore the only check
that can notice a tag whose asset was replaced after it was pinned — the digest
GitHub publishes cannot, because it would move with the replacement.

The published digest earns its place elsewhere. It is available for *any* tag
before the download starts, which is what lets a candidate nobody has pinned yet
be verified exactly as strictly as the incumbent. So `fetch` checks both where
both exist, refuses when they disagree, and falls back to the recorded one alone
when the API is unreachable — it is rate limited unauthenticated, and an install
that fails because a stranger used the quota is a bad install. Without a
recorded digest that fallback is not available and the fetch fails instead: an
unverified build is not worth measuring against.

Promoting a candidate to the pin is then a single mechanical act — commit the
digest `fetch-engine` printed.
