---
name: lllm3090-runbook
description: Operating and changing the lllm3090 stack — install and upgrade, the panel and engine lifecycle, and the failure modes that present as something other than what they are. Use when starting, stopping or switching models, when the engine will not come up or disappears, when the panel is "up" but broken, when a download stalls, when a change appears not to take effect, or before measuring anything on the box.
---

# Operating lllm3090

One GPU, one engine. Everything below assumes that invariant.

## Layout

```
~/.local/share/uv/tools/lllm3090/    the package (a uv tool venv)
~/.local/share/lllm3090/llama.cpp/   the pinned engine build
~/.local/state/lllm3090/engine.log   what the panel streams
~/.local/state/lllm3090/engine.pid   the running engine
~/models/<Name>/*.gguf               checkpoints, one directory each
```

Ports: **1919** engine (OpenAI *and* Anthropic APIs) · **8080** panel.

**`~/models` is a symlink to `/srv/models` on the NVMe** (moved 2026-08-31).
`/home` is a Crucial MX500 on SATA and `/` is a Lexar NQ790 on Gen4 NVMe;
`MODELS_DIR` is an env var (`LLLM3090_MODELS_DIR`) but the symlink means the
package needs to know nothing. The originals are still at `~/models.sata` until
they are deleted. Read speeds, `O_DIRECT`:

| | `/` (NVMe Gen4) | `/home` (SATA) |
|---|---|---|
| sequential | 4.4 GB/s | 0.53 GB/s |
| random 1 MiB | 5.15 GB/s | 0.37 GB/s |

**A cold load does not improve by the ratio of those numbers**, and predicting
that it would was wrong by 6x. Measured on the same 18.2 GB checkpoint, page
cache dropped per run with `posix_fadvise(DONTNEED)` -- which needs no root, and
without which the second run reads a warm cache and looks impossible:

| | seconds to ready |
|---|---|
| SATA, cold | 62 |
| NVMe, cold | 15-26 (three runs) |
| NVMe, warm cache | 9.8 |

The warm figure is the floor: dequantisation, the VRAM upload and graph build
cost ~10 s whatever the disk does. Subtracting it, the effective read rate
during a load is ~0.35 GB/s on SATA and ~1.8 GB/s on NVMe -- the SATA figure
matches its random-read measurement, and the NVMe one is well under its
sequential. **A load is not a sequential stream, so never size one from a
sequential benchmark.** The real gain from the move is about 3x, not 8x.

## Install and upgrade

```bash
uv tool install lllm3090       # or: uv tool upgrade lllm3090
lllm3090 setup                 # idempotent; also the repair command
```

`setup` checks the hardware, installs `libvulkan1` if missing, fetches the
pinned engine by checksum, and installs and **restarts** the panel unit.

**Run `setup` after every upgrade, not just the first install.** Upgrading
replaces the package under the running panel, which then reads the new data
files with the old classes: it keeps listening and every API call fails, and
because the process never exits `Restart=on-failure` does not rescue it.

## Building a CUDA engine

Worth **1.31x on prefill and 1.28x on decode** on the 3090, measured against the
Vulkan build from the same commit -- not the 3-4x this project claimed for a
year before anyone measured it. A cold 80k prompt goes 118.2 s -> 90.0 s, so two
minutes is mostly what the card costs to prefill a dense 27B and no backend
change makes it thirty seconds.

**But 1.28x is the bare number and the engine is never bare.** With the MTP head
the engine turns on automatically, the dense 27B runs **54.8 tok/s on Vulkan
against 84.9 on CUDA -- 1.55x**, because MTP is worth 1.6x on Vulkan and 2.0x on
CUDA and the two multiply. On copy-heavy work, with levers that only pay on CUDA,
it reaches 1.9x. Quote 1.6x, not 1.3x, when the question is what a user would
feel; quote 1.28x only when comparing backends with speculation off. Measuring a
backend with the product's features disabled understated the answer by a quarter
here, and that mistake was load-bearing in a decision not to adopt it.

**Ubuntu's own CUDA cannot build it.** 26.04 ships 13.1 in multiverse; glibc
2.43 declares `rsqrt`/`rsqrtf` `noexcept(true)` and CUDA 13.1 declares them
bare, so `nvcc` refuses anything including `<math.h>`. CUDA 13.3 tests for glibc
>= 2.42 and matches (`_NV_RSQRT_SPECIFIER`). Take the toolkit from NVIDIA's
`ubuntu2604` repo via their `cuda-keyring` package, not from the distribution.

```bash
git clone --depth 1 --branch <TAG> https://github.com/ggml-org/llama.cpp
PATH=/usr/local/cuda-13.3/bin:$PATH cmake -S llama.cpp -B build \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$(nvidia-smi \
    --query-gpu=compute_cap --format=csv,noheader | tr -d .)-real" \
  -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)" --target llama-server llama-bench
```

**Derive the architecture, never type it.** `hardware.Profile.compute_capability`
already carries it, from `nvidia-smi`; the profiles span 8.6, 8.9 and 12.0. A
`-real` build runs on that architecture *only* -- name the directory for it
(`engines/b10715-cuda-sm86`) so a card swap fails legibly.

Two things a downloaded build gives you that a compiled one does not: an
identity (it reports `build 1, commit <sha>`, because a shallow clone has no tag
history, and nobody attests to the binary), and **~700 MiB of VRAM** -- CUDA
reports 24125 MiB where Vulkan reports 24822, which `catalog.fit` does not know.
`LLAMA_CURL=OFF` costs nothing: llama.cpp's `-hf` fetcher is unused here.

## Handing a sudo script to Giles

`fs.protected_regular=2` and `/tmp` is sticky, so **root cannot overwrite a file
in `/tmp` owned by another user** -- `curl -O` as root fails with exit 23,
"client returned ERROR on write", after transferring the whole body. Download as
Giles, elevate only for what needs it. Under `sudo`, `$HOME` is root's, so
resolve a staged path through `$SUDO_USER`.

## The commands

```bash
lllm3090 doctor                    # six checks; exits non-zero and names the failure
lllm3090 models                    # catalogue: fits, installed, measured speed
lllm3090 start <Name> [--ctx N] [--parallel N]
lllm3090 stop                      # frees the VRAM; do this before gaming
lllm3090 status
lllm3090 claude                    # Claude Code against the local model
lllm3090 bench <Name>              # llama-bench, and a profile block to contribute
lllm3090 fetch-engine --build TAG  # a build to measure, beside the install not over it
```

The panel at `http://127.0.0.1:8080` does the same and streams the engine log.
It binds loopback **by design** — its endpoints start processes with no
authentication, so remote access is an SSH tunnel, never a LAN bind.

## Speculative decoding

The pinned build's `--spec-type` accepts `draft-simple, draft-eagle3,
draft-mtp, draft-dflash, draft-dspark` and five ngram variants. Only one of
them is worth having here.

**MTP is on automatically and needs nothing.** `engine.start` reads the
checkpoint's header (`lllm3090.gguf`) and adds `--spec-type draft-mtp` when the
`nextn` tensors are present. Measured on the 3090, paired, GPU idle either
side: Qwen3.8-27B **34.9 -> 56.6** tok/s (1.62x), Qwen3.6-35B-A3B-MTP **130.5
-> 171.8** (1.32x, 179.9 on code editing). It is decided from the file, never
from the catalogue: llama.cpp *refuses to start* with that flag against a
checkpoint lacking the head, and a metadata key is not proof -- a conversion can
announce `nextn_predict_layers` and ship no tensors. It gets *better* on
copy-heavy work, not worse: reproducing a 369-line file with one identifier
renamed, Qwen3.8-27B ran **32.8 -> 59.7** tok/s (**1.82x**) at **100%** draft
acceptance, because the next token is trivially predictable when the output is
copying a known input. MTP's figures are also the ones the Vulkan bug below did
*not* touch -- they are unchanged across the fix.

**ngram is a regression on Vulkan and a win on CUDA -- this is the one verdict
here that flips with the backend.** Everything in the next two paragraphs is
about the Vulkan build that ships. On the CUDA engine, same commit and prompts,
`ngram-cache` goes 0.88x -> **1.41x** on copy-heavy work and
`draft-mtp,ngram-cache` reaches **2.44x**, beating `draft-mtp` alone at 2.22x --
the fastest width-3 configuration measured on this box. Prose and code editing
stay losses (0.93x, 0.92x). Acceptance is near-identical on both backends, so
what changed is the price of checking drafts, not their quality. Do not carry a
Vulkan speculation verdict onto CUDA, in either direction.

**On Vulkan: ngram is a regression, and stacking it with MTP is worse than MTP
alone.**
Measured on Qwen3.8-27B on b10715, on general prose: `ngram-cache` alone
**0.65x**, `draft-mtp,ngram-cache` 1.35x, `draft-mtp` alone 1.61x. So the stack
does beat ngram by itself -- it just costs you a sixth of what MTP was already
giving. Hit rates on novel generation are low, so you pay
for rejected drafts and the weak drafts displace good ones. The advice online
that advanced users should combine them is wrong on this box.

**It does not flip on long copy-heavy prompts either -- that was tested,
twice.** The obvious objection to the numbers above is that they were taken on a
seven-line edit, which is not the regime prompt-lookup drafting is built for.
Re-measured on Qwen3.8-27B against a 369-line file copied back with one
identifier renamed, 7 samples per cell, on **b10715**: `ngram-cache` **0.88x**,
`draft-mtp,ngram-cache` 1.61x, `draft-mtp` alone **1.82x**. The objection was
sound and did not rescue it -- acceptance climbs from **0%** on the seven-line
edit to **58%** on the long copy, and 58% is still below what it costs to draft.
Copy-heavy work is where prompt-lookup does least badly here (0.88x, against
0.65x on prose), not where it wins. Adding ngram to MTP *lowered* acceptance
from 100% to 87%, which is the weak-drafts-displace-good-ones effect made
visible.

**A speculation measurement is only as good as the build under it.** The first
attempt at the above ran on the then-pinned **b10628** and had to be thrown
away: llama.cpp
[PR #27812](https://github.com/ggml-org/llama.cpp/pull/27812) fixed a Vulkan
graph optimiser that reordered nodes across aliased tensor views, so the target
accepted draft tokens it had not chosen -- wrong output at temperature 0, and
*invalid acceptance numbers*. The bug **flattered** ngram rather than
handicapping it: the same sweep read 0.87x at 47% acceptance before the fix and
0.65x at 24% after. CUDA was never affected. **The pin has since moved to
b10715**, so the sweep's default and the served engine are the same build again.

**Draft width is a property of the backend, not of the drafter.**
`--spec-draft-n-max` sizes what any draft *model* proposes per verification
step; the n-gram modes have their own `--spec-ngram-*` knobs. Comparing two
drafters at two widths is two variables and reads as a verdict on the drafter --
that mistake was made and caught. Measured on Qwen3.8-27B, Vulkan, going from
width 3 to the 7 that DFlash2's README recommends: MTP loses 22% on prose and
DFlash2 14%, and acceptance falls for both (85% -> 63%, 80% -> 67%). **Take
llama.cpp's default of 3 on Vulkan**; the engine passes no width and so already
does. The mechanism is that Vulkan gets nothing from a wider batch -- pp512
1026.9 against pp4096 1014.0, where CUDA goes 1217.4 -> 1343.8 -- and verifying
k drafted tokens *is* a batched forward pass.

**That mechanism was tested on CUDA and confirmed.** Widening 3 -> 7 drops MTP's
acceptance to 64% on CUDA against 63% on Vulkan -- the same drafts wasted -- but
costs 7-8% of decode instead of 22%. On `long-copy`, where acceptance holds at
96%, width 7 is a **13-22% win** (106.5 tok/s against 94.0, or 115.1 stacked with
ngram, the fastest number on this box). So: **take 3 on Vulkan; on CUDA take 3
for prose and code editing and 7 for copy-heavy work.** It is a per-workload knob
(`--spec-draft-n-max`), not a default worth changing.

The copy-row figures are spans because the width-7 run drifted -- see the
baseline bullet below. Direction is solid; magnitude is +/-5%.

**A drafter costs VRAM, which is the resource the catalogue defends.** DFlash
and EAGLE-3 need a separate resident model, so they buy speed with context.
**DFlash2 was measured and rejected**: against the free MTP head at width 3 it
is 0.90-1.04x -- a win of 4% on code editing, losses of 3% and 10% on prose and
copying -- for 1.1 GB, which is ~36k tokens of context, taking the dense 27B
from 172k to 136k. Its published 2.26-3.55x is against *no speculation at all*,
which is not the floor here.

**CUDA does not rescue it, and that was a surprise** -- its published figures are
CUDA figures, and it was the rejection most expected to flip. On the CUDA engine
it is 0.92-1.01x against MTP, against 0.90-1.04x on Vulkan: unchanged. It was
never losing to the backend, it was losing to MTP, which is free and which a
faster backend speeds up too. **A lever measured against a moving comparator does
not gain on it just because both get faster** -- worth remembering before
re-testing anything else on new hardware.
Prefer MTP, which lives inside the checkpoint and costs only its own weights.

**Check whether a checkpoint has the head before downloading a variant:**
`python -c "from lllm3090 import gguf; print(gguf.has_mtp('<path>'))"`. Vendors
ship `-MTP-` repos alongside the plain ones.

## Failure modes worth knowing

Each of these presented as something other than what it was. That is why they
are here: the symptom points away from the cause in every case.

**The panel unit must not kill the engine.** The engine is launched by the panel
and tracked by a pidfile so a panel restart does not cost a multi-minute reload.
`start_new_session=True` escapes the process group but **not the cgroup**, so
with systemd's default `KillMode=control-group` every panel restart silently
killed the model — surfacing as "connection refused" from a client, and as
abandoned downloads. The unit therefore sets `KillMode=process`. If you touch
that unit, verify by restarting it and checking the engine's **pid is
unchanged**.

**Starting a model must not block the request.** `/api/start` launches and
returns; the status distinguishes `loading` from `running`. When it blocked for
the whole load, systemd's stop waited on the in-flight request and SIGKILLed the
panel after 90 seconds, which read as an outage every time a model was switched.

**"Loading" is not "failed".** llama-server binds its port long before the
weights finish uploading; a 15–21 GB model takes minutes. Check the engine log
before concluding anything.

**A start that fails on memory is failing on KV, not weights.** If the weights
fit and the context does not, allocation fails after the model has loaded.
Restart with a smaller `--ctx` rather than a smaller model.

**Downloads resume -- the panel's do.** They are threads and die with the panel,
leaving a `.part` file; the panel picks those up on startup. A repeatedly
failing download usually means the file was renamed upstream, not that the
network is bad.

**`hf_hub_download` does not.** Killed mid-transfer and restarted, it opens a
*new* `.incomplete` blob under a different suffix and re-fetches from zero,
leaving the old one on disk forever. Observed on 1.28.0: 8.3 GB orphaned and
18.2 GB re-downloaded. Check
`<local_dir>/.cache/huggingface/download/*.incomplete` after any interrupted
fetch and delete what the running download is not using.

**A model's chat template can reject a client outright.** Qwen3.8-27B's template
raises on a system message that is not first, and agent harnesses send them
mid-conversation, so every request after the first failed with a 500 and a Jinja
traceback naming nothing useful. It is served with a patched template from
`data/qwen3.8-27b.jinja`; any catalogue entry may carry a `chat_template:` field.
Check a new model's template for `raise_exception` before assuming it is fine.

## When a change appears not to take effect

**`uv tool install --force .` reuses a cached build** when the version string has
not changed — and `setuptools_scm` gives a stable version within a day and a
commit, so an edited tree installs the *previous* wheel and reports success. Two
consecutive "that's fixed now" claims were both false for this reason.

```bash
uv tool install --force --reinstall --no-cache .
lllm3090 install-service          # the unit ships in the package, so it changes too
```

For tests and docs use `uv run --extra dev`, which installs the project itself
editable. `uv run --with '.[dev]'` resolves from a previously built artifact and
will silently test stale code.

**Never `pkill -f` on a pattern that appears in your own command line** — it
matches the shell running it and kills the script partway with no error saying
why. Use the pidfile.

## The suite is green here and red on a runner

A CI runner has no GPU, so `hardware.detect()` borrows the reference profile
with the *documented* `DRIVER_RESERVE_MIB` rather than the live figure this card
reports through `nvidia-smi`. Every context number therefore lands a few
thousand tokens elsewhere, and a test that asserts an exact window -- or an
inequality that is close -- passes here and fails there.

Reproduce it before guessing:

```python
import unittest.mock as mock
from lllm3090 import hardware
with mock.patch.object(hardware, "_smi", return_value=None):
    ...
```

## Before you measure anything

On a single-GPU box the benchmark and the workload are the same machine.

- **A speed is a fact about one card.** `data/profiles.yaml` carries capacity
  and compute capability per GPU; the running card is matched **by name**, and
  an unrecognised one gets a profile synthesised from `nvidia-smi` so fit and
  context stay correct while nothing claims its speed. With no GPU at all the
  profile is `present=False`, and the CLI and panel say so. Speeds are never
  scaled between cards -- a bandwidth ratio produces a guess that prints like a
  measurement. `lllm3090 bench` is the only way a card gets real numbers.

- **Check the served model before *and* after each run -- via `/props`, not
  `/v1/models`.** The engine ignores the `model` field in a request and serves
  whatever is loaded, so a model switched mid-run produces numbers attributed to
  the wrong model. This has happened; the run was discarded. `/v1/models`
  reports the `--alias`, which a benchmark harness typically sets to a constant,
  so it answers the same string whatever is loaded and confirms nothing --
  `dev/spec-sweep.py` printed `serving: b` for months on end. `/props` carries
  `model_path`; compare it to the checkpoint you asked for.
- **The benchmark and the engine must not share the card, or the port.** Two
  engines on one GPU measures the contention. `dev/spec-sweep.py` runs on
  `ENGINE_PORT + 2` and refuses to start while the engine answers -- it used to
  bind 1919 and clear it with `pkill -f "llama-server --model <MODEL>"`, which
  matches the *installed* engine serving that same checkpoint to somebody.
- **Cold and warm are different measurements.** The prefix cache means a repeat
  of the same prompt is not a cold prefill. Use a unique prefix (a UUID in the
  text) when you want a genuine cold number.
- **Count reasoning tokens.** Models that stream thinking in a separate delta
  field produce zero `content` deltas until they finish, which reads as a hang
  or a zero rate.
- Discard anything that overlapped real use. A live session against the same
  engine once produced an apparent 47% throughput regression that was pure
  contention.
- **Every run carries its own baseline, even when re-running one variable.**
  `dev/spec-sweep.py` takes a comma-separated config list as its third argument,
  and dropping `baseline` from it to save eight minutes is a false economy: two
  CUDA runs 35 minutes apart on an idle card had baselines 0.5% apart on prose
  and **6.8% apart on `long-copy`**, which was a third of the effect being
  measured on that row. Ratios are only comparable within a run.
- **A baseline in the same run is not enough, and believing it was is a mistake
  already made here.** It cancels drift *between* runs, not *through* one: the
  baseline config runs first, so the card it measures is not the card the later
  configs get. Per-request rates in `sw-baseline.log` show it plainly -- a sweep
  started cold *gains* 3% as clocks ramp (42.5 -> 43.9), and one started right
  after another finished *loses 9% over ten minutes* (43.4 -> 41.6 -> 39.5), on
  the identical invocation. **Never start a sweep on a card that has just
  finished one**, let it idle first, and read the per-request rates out of the
  log before trusting a median. Randomised config order or an end-of-run baseline
  would bound this; neither is implemented.
- **Three samples is not a measurement when the variance is real.**
  Speculative decoding's acceptance rate swings with content: three runs of
  MTP on the A3B spanned 100.8 to 171.9 tok/s and the median read as *no gain
  at all*. Seven warm samples put it at 1.32x, with MTP's worst run above
  baseline's best. Where the spread is wide, take more samples before believing
  the middle one.
- **Discard the first request after a start.** It carries graph build and
  warm-up and has come in low every time -- 8.8 tok/s against a 30 steady
  state in one case.
- **The engine is not deterministic at temperature 0.** Two identical requests
  in one server produce different text; the *first* request after a restart is
  reproducible across restarts. So compare like with like, and never attribute
  a text difference to a flag without running that control first -- it is what
  stopped MTP being blamed for output drift that plain decode also has.
