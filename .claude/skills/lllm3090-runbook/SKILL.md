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

## The commands

```bash
lllm3090 doctor                    # six checks; exits non-zero and names the failure
lllm3090 models                    # catalogue: fits, installed, measured speed
lllm3090 start <Name> [--ctx N] [--parallel N]
lllm3090 stop                      # frees the VRAM; do this before gaming
lllm3090 status
lllm3090 claude                    # Claude Code against the local model
lllm3090 bench <Name>              # llama-bench, and a profile block to contribute
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
renamed, Qwen3.8-27B ran **32.2 -> 59.5** tok/s (**1.85x**) at **100%** draft
acceptance, because the next token is trivially predictable when the output is
copying a known input.

**ngram is a regression, and stacking it with MTP is worse than MTP alone.**
Measured on Qwen3.8-27B: `ngram-cache` alone 0.88x, `draft-mtp,ngram-cache`
1.42x, `draft-mtp` alone 1.62x. So the stack does beat ngram by itself -- it
just costs you a fifth of what MTP was already giving. Hit rates on novel generation are low, so you pay
for rejected drafts and the weak drafts displace good ones. The advice online
that advanced users should combine them is wrong on this box.

**It does not flip on long copy-heavy prompts either -- that was tested.** The
obvious objection to the numbers above is that they were taken on a seven-line
edit, which is not the regime prompt-lookup drafting is built for. Re-measured
on Qwen3.8-27B against a 369-line file copied back with one identifier renamed,
7 samples per cell: `ngram-cache` **0.92x**, `draft-mtp,ngram-cache` 1.60x,
`draft-mtp` alone **1.85x**. The mechanism behaved exactly as the objection
predicted and the throughput still did not follow -- acceptance climbed from
**0%** on the seven-line edit to **62%** on the long copy, and 62% is still
below what it costs to draft. Adding ngram to MTP *lowered* acceptance from 100%
to 85%, which is the weak-drafts-displace-good-ones effect made visible. Two
caveats worth keeping: ngram's spread was wide ([29.0 .. 34.5] against a tight
baseline [31.8 .. 32.8]), so individual runs did clear baseline while the median
did not; and this says nothing about a checkpoint with no MTP head, where the
comparison is ngram against nothing rather than ngram against a better drafter.

**A drafter costs VRAM, which is the resource the catalogue defends.** DFlash
and EAGLE-3 need a separate resident model, so they buy speed with context.
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

- **Check the served model before *and* after each run** (`/v1/models`). The
  engine ignores the `model` field in a request and serves whatever is loaded,
  so a model switched mid-run produces numbers attributed to the wrong model.
  This has happened; the run was discarded.
- **Cold and warm are different measurements.** The prefix cache means a repeat
  of the same prompt is not a cold prefill. Use a unique prefix (a UUID in the
  text) when you want a genuine cold number.
- **Count reasoning tokens.** Models that stream thinking in a separate delta
  field produce zero `content` deltas until they finish, which reads as a hang
  or a zero rate.
- Discard anything that overlapped real use. A live session against the same
  engine once produced an apparent 47% throughput regression that was pure
  contention.
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
