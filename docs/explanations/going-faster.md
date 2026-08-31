# Speed levers, tried and untried

The dense 27B is the slowest model in the catalogue, and
[structurally so](dense-vs-moe.md) — it reads every one of its parameters per
token. That is not the whole story: multi-token prediction has already taken it
from 34.9 to 56.6 tok/s, and several further levers have been looked at. This
page records all of them — what is already in the engine, what was measured and
set aside, what it would cost to have, and what is parked and why.

Most of them have now been measured twice, on Vulkan and on CUDA, and **several
of the verdicts differ by backend**. A lever is therefore reported with the
backend it applies to, and the one that ships is Vulkan.

If you want the levers themselves explained rather than scored, start with
[](what-makes-it-fast.md); this page assumes them and reports numbers.

## Every lever, on Qwen3.8-27B

All of it measured on one RTX 3090 against the dense 27B at Q4_K_S, on llama.cpp
b10715. Ranges span three workloads — prose, a code edit, and copying 369 lines
back with one identifier renamed — and each ratio is against its own run's
baseline, so they compare with each other and not with absolute numbers from
elsewhere.

**Two effect columns, because there are two backends and they disagree.** The
shipped engine is Vulkan; the CUDA column is measured but not installed anywhere,
and a reader who acts on it while running the shipped build will get the Vulkan
number. Where the two differ, the verdict says which backend it applies to.

| lever | moves | on Vulkan (shipped) | on CUDA (not shipped) | verdict |
|---|---|---|---|---|
| [Multi-token prediction](#already-in-the-engine) | decode | **1.61–1.82×** | **1.96–2.22×** | **on by default** — and worth more on CUDA |
| [q8_0 KV cache](#already-in-the-engine) | capacity | 2× the conversation | same | **on by default** |
| [Flash attention](#already-in-the-engine) | both | not separately measured | not separately measured | **on by default** |
| [CUDA instead of Vulkan](#cuda-instead-of-vulkan-13-bare-16-as-shipped) | both | — | 1.31× prefill, 1.28× decode bare, **~1.6×** as shipped | not adopted — costs a toolkit and a per-card binary |
| [DFlash2 drafter](#measured-and-not-adopted-dflash2) | decode | 0.90–1.04× *against MTP* | 0.92–1.01× *against MTP* | rejected on **both** — a wash, for 36k tokens of context |
| [Draft width 7](#draft-width-matters-more-than-which-drafter) | decode | 0.78–0.98× *against width 3* | 0.87–1.22× *against width 3* | rejected on Vulkan; on CUDA, copy-heavy work only |
| [Prompt lookup (ngram)](#measured-and-set-aside-prompt-lookup-drafting) | decode | 0.65–0.88× | 0.92–1.41× | rejected on Vulkan; **wins copying on CUDA** |
| [ngram stacked on MTP](#measured-and-set-aside-prompt-lookup-drafting) | decode | 0.84–0.90× *against MTP* | 0.86–1.10× *against MTP* | rejected on Vulkan; **the fastest copying config on CUDA** |
| [q4_0 KV cache](#already-in-the-engine) | capacity | 4× the conversation, ~0.63× decode | not measured | not taken — never measured here |
| [coopmat2](#still-unanswered-coopmat2) | prefill | unknown | n/a — a Vulkan concept | cannot be determined from this build |
| [TurboQuant KV](#parked) | capacity | claimed ~2.5× against q8_0 | not merged | parked — not merged anywhere |
| [vLLM / AWQ-Marlin](#parked) | throughput | 8× at ten concurrent users | a different engine | out of scope — this engine serves one |

**Nothing has displaced multi-token prediction on the backend that ships.** The
engine's defaults are the best combination tried on Vulkan, and the interesting
content below is mostly *why* — particularly one variable, draft width, which
turned out to matter more than the choice of drafter and to be a property of the
backend rather than of any drafter at all.

On CUDA that last sentence stops being a caveat and becomes the finding: two of
the three levers rejected above come back, MTP itself grows from 1.6× to 2.0×,
and the fastest configuration for copy-heavy work is one this page had written
off. [What the re-measurement settled](#what-the-cuda-re-measurement-settled)
records which predictions that confirmed and which it killed.

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

### On CUDA it is a different lever

The same sweep, same build commit, same prompts, on the CUDA engine — and the
verdict does not survive:

| | prose | code-edit | long-copy |
|---|---|---|---|
| baseline | 43.2 | 43.6 | 42.4 |
| `draft-mtp` | 84.9 (1.96×, 85%) | 89.0 (2.04×, 86%) | 94.0 (2.22×, 100%) |
| `ngram-cache` | 40.2 (0.93×, 36%) | 39.9 (0.92×, 1%) | **59.6 (1.41×**, 57%) |
| `draft-mtp,ngram-cache` | 72.5 (1.68×, 70%) | 76.7 (1.76×, 77%) | **103.5 (2.44×**, 86%) |

Two things changed and one did not.

**Prompt lookup wins the workload it was built for.** 0.88× on Vulkan, **1.41×**
on CUDA — at essentially the same acceptance, 58% against 57%. The drafts are as
good and as numerous as they always were; what changed is that the backend no
longer charges full price to check them.

**Stacking beats MTP alone on copying.** `draft-mtp,ngram-cache` at 2.44× against
`draft-mtp`'s 2.22× — 103.5 tok/s against 94.0, the fastest number anywhere in
this page's width-3 tables. The displacement effect is still visible in the
acceptance column (100% → 86%, exactly as on Vulkan), so the *drafts* are still
being displaced. On CUDA the extra drafts are cheap enough that winning some of
them beats keeping acceptance perfect.

**Prose and code editing are still losses**, 0.93× and 0.92×. Prompt lookup did
not become a good idea; it became a good idea *for copying*, which is the regime
it was always advertised for and the one Vulkan would not let it have.

Two numbers in that table should not be leaned on. `draft-mtp,ngram-cache` on
code-edit spanned 70.3–92.5 tok/s across seven samples — a 32% spread, far wider
than anything else measured here, so its 1.76× is soft. And `ngram-cache` on
prose reads 36% acceptance where Vulkan read 24%; acceptance is a property of
the drafts rather than of the backend, so that gap is unexplained, and nothing
above rests on it.

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
own claim was always narrow — 0.52 tokens of acceptance length — and it does not
survive on either backend measured here.

### CUDA does not rescue it

DFlash2's published figures are CUDA figures, so of the three rejections on this
page it was the one most likely to be an artefact of Vulkan. It is not. Same
sweep on the CUDA engine, width 3, against a baseline of 43.2 / 43.6 / 42.4:

| | prose | code-edit | long-copy |
|---|---|---|---|
| `draft-mtp` | **84.9** (1.96×, 85%) | 89.0 (2.04×, 86%) | **94.0** (2.22×, 100%) |
| `dflash` | 83.8 (1.94×, 80%) | **90.3** (2.07×, 88%) | 86.1 (2.03×, 85%) |

A win of 1% on code editing, a loss of 1% on prose and 8% on copying — against
4%, 3% and 10% on Vulkan. The same shape, slightly flatter: still a wash, still
for 1.1 GB and about a fifth of the context. **The rejection stands on both
backends**, and it is the only one on this page that does.

That matters more than a confirmation usually would, because this page predicted
the opposite. See [what the re-measurement
settled](#what-the-cuda-re-measurement-settled).

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

#### And on CUDA the penalty is a third of the size

This is the cleanest test of the claim that the width penalty belongs to the
backend, because acceptance and speed can be read separately. Widening 3 → 7,
each width against its own run's baseline:

| width 3 → 7, on CUDA | prose | code-edit | long-copy |
|---|---|---|---|
| `draft-mtp` | **0.92×** | 0.87× | **1.22×** |
| `dflash` | 1.01× | 1.06× | 0.90× |
| `draft-mtp,ngram-cache` | 1.00× | 1.04× | **1.19×** |

The width-7 run carries its own `baseline` config so each ratio is against the
baseline measured alongside it, which the Vulkan tables above cannot offer — the
Vulkan width-7 sweep recorded no baseline, so its 0.78–0.98× is a ratio of raw
tok/s across two runs. That distinction is not pedantry here: the two CUDA runs'
baselines differ by 6.8% on `long-copy`, which is a third of the effect being
measured on that row.

**Acceptance falls by the same amount on both backends and the speed does not.**
MTP on prose goes 85% → 64% acceptance on CUDA, against 85% → 63% on Vulkan —
the drafter is throwing away just as many drafts. Vulkan charges 22% of decode
for that; CUDA charges 8%. Same wasted drafts, a third of the bill. Acceptance is
a property of the drafter and the price of a wide verify is a property of the
backend, and here they are separated cleanly.

**Where acceptance barely falls, width 7 becomes a win.** On `long-copy` MTP
holds 96% at width 7, and CUDA turns that into **1.22×** — 2.70× over baseline,
against 2.22× at width 3. Stacked with ngram it reaches **2.91×**, 115.1 tok/s,
the fastest configuration measured anywhere on this page. Vulkan, at 97%
acceptance on the same workload, still lost.

So the advice splits by backend rather than being a rule:

- **On Vulkan, take the default 3.** Unchanged, and the engine already does.
- **On CUDA, 3 is still right for prose and code editing** — MTP loses 8% and
  13% at width 7 — **and 7 is right for copy-heavy work**, worth 19–22%.

Which is a per-workload knob and not a default, so nothing about it would be
switched on automatically even if CUDA shipped. `--spec-draft-n-max` is the flag,
and a user who knows their session is mostly refactoring is the one who should
set it.

## CUDA instead of Vulkan: 1.3× bare, 1.6× as shipped

Built and measured. Both engines from llama.cpp commit `662a0b0` — the same
commit as the pinned Vulkan build, so this is a backend comparison and nothing
else — with CUDA 13.3 compiled for `sm_86` alone, derived from the card rather
than typed.

**Real cold prefill**, dense 27B, empty cache, `cache_prompt=false`:

| prompt | Vulkan | CUDA | |
|---|---|---|---|
| 10k tokens | 10.5 s | 8.1 s | 1.30× |
| 40k tokens | 48.5 s | 37.1 s | 1.31× |
| 80k tokens | **118.2 s** | **90.0 s** | 1.31× |

**`llama-bench`**, same models and flags:

| | Vulkan | CUDA | |
|---|---|---|---|
| pp512 | 1026.9 | 1217.4 | 1.19× |
| pp4096 | 1014.0 | 1343.8 | 1.33× |
| tg64 | 32.9 | 42.2 | **1.28×** |

### The 3–4× claim was wrong

This page and `installation.md` both said Vulkan prefill was 3–4× behind CUDA,
and both were repeating a figure nobody here had measured. It is **1.3×**.

What was right is the number that made it plausible: 80k really does take about
two minutes. But that is mostly what this card costs to prefill 27B dense, not a
tax the backend is charging. CUDA takes two minutes down to a minute and a half.
It does not take it to thirty seconds, and no backend change will — the fix for
a slow first turn is to send a shorter prompt or to keep the cache warm.

### Decode was the surprise

Community scoreboards put CUDA about 10% ahead on generation. Here it is **28%**
— 32.9 to 42.2 tok/s before any speculation. That is the larger of the two
effects in practice, because decode is what a conversation spends its time doing.

The sweep confirms that figure from a second instrument: its `baseline` config is
no speculation at all, and reads 34.1 / 33.8 / 32.8 tok/s on Vulkan against
43.2 / 43.6 / 42.4 on CUDA — 1.27 / 1.29 / 1.29×, where `llama-bench` said 1.28.
Two different harnesses, three workloads, the same number.

### The compounding was underestimated

This section used to predict what CUDA plus speculation would give: *"42.2 with
MTP's measured 1.6× on top would be around 67 tok/s, against the 53 the Vulkan
engine delivers today."* That was measured, and **it was 21% low**. The real
figure is **84.9 tok/s** on prose, and 94.0 on copying.

The arithmetic was fine; the assumption inside it was wrong. It multiplied
CUDA's bare decode by MTP's *Vulkan* multiplier, and MTP does not have one
multiplier — it is worth 1.61–1.82× on Vulkan and **1.96–2.22× on CUDA**, at
acceptance rates that match to within a point (85 / 86 / 100% against
85 / 84 / 100%). Identical drafts, identical accepts, more speed.

That is the batching argument arriving where it was predicted to. Verifying k
drafts is a batched forward pass; a backend that gains 10% from a wider batch
gains it on every verification step, which is most of what a speculating engine
does. So the two effects do not add, they multiply, and the honest headline for
a user is not the bare number:

| dense 27B, as each backend would actually serve it | prose | code-edit | long-copy |
|---|---|---|---|
| Vulkan + MTP — **what installs today** | 54.8 | 55.7 | 59.7 |
| CUDA + MTP, same flags | 84.9 | 89.0 | 94.0 |
| | 1.55× | 1.60× | 1.57× |
| CUDA, best config costing no VRAM | 84.9 | 89.0 | **115.1** |
| | 1.55× | 1.60× | **1.93×** |

The last row buys `long-copy` with `--spec-type draft-mtp,ngram-cache` and
`--spec-draft-n-max 7`, both of which are flags rather than downloads. DFlash2
would edge code editing to 91.7 tok/s, and it is left out because 1.1 GB of
drafter is not free — that is the [whole
argument](#measured-and-not-adopted-dflash2) against it, and it does not stop
being true because the backend changed.

**CUDA is worth 1.3× to a benchmark and about 1.6× to this engine**, because the
engine speculates and the benchmark did not. On copy-heavy work, with the ngram
stack and width 7 that only pay on CUDA, it approaches 2×.

### And Vulkan does not batch

The single most useful line in those tables is Vulkan going **1026.9 → 1014.0**
from pp512 to pp4096 while CUDA goes 1217.4 → 1343.8. Vulkan gets *nothing* from
a wider batch. CUDA gets 10%.

That is the mechanism behind [the draft-width finding](#draft-width-matters-more-than-which-drafter):
verifying k drafted tokens is a batched forward pass, so a backend that does not
reward wider batches will punish wider drafts. It also means every drafter
verdict on this page was measured on the backend least able to make drafting pay,
and none of them should be treated as settled for CUDA — including the two that
were rejected.

## What the CUDA re-measurement settled

The paragraph immediately above is a falsifiable prediction, and this page is
obliged to say how it did rather than to quote it only if it landed. It was
tested by re-running the whole sweep on the CUDA engine — nine config-runs,
both draft widths, the same commit, the same prompts, the same seven-samples
protocol. **It was mostly right, and specifically wrong once.**

| the page predicted | outcome |
|---|---|
| ngram's rejection is not settled for CUDA | **right** — 0.88× → 1.41× on copying, a flipped verdict |
| stacking's rejection is not settled | **right** — it is now the fastest copying config, 2.44× against MTP's 2.22× |
| the width penalty belongs to the backend | **right** — same acceptance collapse, a third of the speed cost; and width 7 wins copy-heavy work |
| DFlash2's rejection is not settled | **wrong** — 0.92–1.01× against MTP on CUDA, against 0.90–1.04× on Vulkan. Unchanged. |
| (unstated) MTP's own multiplier would carry over | **wrong** — 1.61–1.82× on Vulkan, 1.96–2.22× on CUDA |

The DFlash2 miss is the useful one. The argument for re-testing was that Vulkan
was the backend least able to make drafting pay and DFlash2's own figures were
CUDA figures — sound reasoning that predicted the wrong result. What it missed
is that DFlash2 was never losing to the backend; it was losing to **MTP**, which
is free, and which a better backend speeds up too. A lever measured against a
moving comparator does not gain on it just because both get faster.

The unstated assumption is the more expensive error, because it was doing work
in a decision. "1.3× is probably not worth a toolkit" was reasoning about the
bare number when the engine has never run bare. The figure that decision should
have been weighing is 1.6×.

None of this changes what installs. The shipped engine is Vulkan, every verdict
in the Vulkan column stands, and a user who switches ngram on today still gets
0.65×.

### What it costs to have

- **A 4–6 GB toolkit and a local compile.** As of b10715 llama.cpp publishes
  CUDA archives for Windows only; Linux gets CPU, Vulkan, ROCm, SYCL and
  OpenVINO. There is no download.
- **Ubuntu's own CUDA is too old to build this.** 26.04 ships 13.1 in
  multiverse, which declares `rsqrt`/`rsqrtf` without an exception specifier
  while glibc 2.43 declares them `noexcept(true)`; `nvcc` then refuses anything
  including `<math.h>`. CUDA 13.3 tests for glibc ≥ 2.42 and matches it. The
  toolkit has to come from NVIDIA's `ubuntu2604` repository, not the
  distribution.
- **A binary tied to one card.** Compiled for `sm_86-real`, it will not run on a
  different architecture at all. The directory is named `b10715-cuda-sm86` so
  that is visible rather than implied.
- **About 700 MiB of context.** CUDA reports 24125 MiB of VRAM where Vulkan
  reports 24822. `catalog.fit` computes against a fixed envelope and does not
  know which backend it is planning for, so a CUDA engine would need that
  envelope to become backend-dependent — or it would promise a window it cannot
  hold.
- **A weaker identity.** A downloaded build is a tag and a digest. A compiled one
  reports `build 1, commit 662a0b0`, because a shallow clone has no tag history:
  only the commit is real, and nobody attests to the binary. See [what a build has to
  prove](#what-a-build-has-to-prove).

So the question is no longer whether CUDA is faster — it is — but whether **1.6×
on an ordinary turn, and up to 1.9× on refactoring work**, is worth asking every
user for a toolkit, a compiler and a per-card binary, in a project whose install
is currently one verified download. That is a decision about what this project
promises, and the measurement does not make it.

It is a harder question than it was, and deliberately so. The version of it that
was asked before the drafters were re-measured — *is 1.3× worth a toolkit?* —
was weighing the wrong number, because it compared two backends running no
speculation while the engine has never run that way. Against the shipped
configuration it is 1.6×, and 1.9× where a session is mostly copying. Whether
that clears the bar is still a judgement about the install promise rather than
about the numbers, but it is now being made against the right ones.

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

**Restrict a sweep with the third argument, but keep `baseline` in it.** The
comma-separated config list exists so a second width does not have to re-run
everything, and it is tempting to drop the baseline as a known quantity. It is
not one: across the two CUDA runs here, taken 35 minutes apart on an otherwise
idle card, the baseline moved 0.5% on prose and **6.8% on `long-copy`**. A ratio
against the other run's baseline would have carried that drift into the answer,
on the row where the effect being measured was 19%.

```
dev/spec-sweep.py MODEL.gguf "" baseline,draft-mtp,mtp+ngram,dflash
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
