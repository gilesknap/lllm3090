---
name: freetoken-ws03
description: Operating the local FreeToken inference stack on ws03 — the systemd units, the :8080 control panel, switching the served model, and the failure modes that present as a hung service. Use when starting, stopping or restarting the local engine, switching models, reading engine logs, when :1919 stops answering or :1920 is held, when the engine "won't start", or before measuring anything on this box.
---

# FreeToken on ws03

RTX 3090 (24 GB, sm_86) · FreeToken 0.1.2 · everything inside
`/home/giles/freetoken-venv`, nothing system-wide. The system CUDA, drivers and
`~/.claude/settings.json` are untouched and must stay that way.

## Layout

```
/home/giles/freetoken-venv/          the whole stack (venv is disposable — do not put state here)
  lib/python3.12/site-packages/
    freetoken/                       engine + ft CLI
    nvidia/cu13/                     pip CUDA 13.0.88 toolchain (NOT /usr/local/cuda)
/home/giles/models/
  Qwen3.6-35B-A3B-NVFP4/             22 GB, the pinned default
  Gemma-4-26B-A4B-NVFP4/             19 GB, needs KV_POOL<=131072 (see below)
  gpt-oss-20b/                       13 GB
  ft-env.sh                          shared CUDA env — every entry point sources this first
  ft-daemon.sh                       supervisor on :1900
  ft-engine-start.sh                 engine on :1919, blocks until it truly answers
  ft-state/logs/serve-1919.log       the engine log the panel streams
  control/{app.py,index.html,run.sh} the :8080 panel
```

Ports: **1900** daemon control plane · **1919** engine (OpenAI + Anthropic) ·
**1920** torch.distributed rendezvous (see below) · **8080** control panel.

## It runs under systemd — do not start it by hand

```bash
systemctl --user status freetoken ft-control
systemctl --user restart freetoken     # always comes back on Qwen: the model is pinned in ft-engine-start.sh
systemctl --user stop freetoken        # frees all 23 GB — do this before gaming
```

`Linger=yes`, so it starts at boot without a login. A full engine crash
self-heals in ~80 s. The JIT caches survive restarts: a warm start reaches first
token in ~0.75 s, against 64 s for the very first build on a cold cache.

Because CUDA lives in site-packages rather than `/usr/local/cuda`, **any** shell
that touches `ft` must `. /home/giles/models/ft-env.sh` first. That is the only
reason those wrapper scripts exist.

## The control panel

`http://127.0.0.1:8080` — live engine state, VRAM bar, model switcher, and the
engine log over SSE. It **binds loopback by design**: `/api/start` and
`/api/stop` launch processes with no auth. Reach it remotely with
`ssh -L 8080:127.0.0.1:8080 giles@ws03`, never a LAN bind.

```
GET  /api/status      daemon_up, engine{running,pid,model,uptimeS,lastExitReason}, vram, models[]
GET  /api/logs        tail of serve-1919.log
GET  /api/logstream   SSE, colour-stripped, \r progress frames collapsed
POST /api/start?model=<dirname>   stops first, then runs ft-engine-start.sh (up to 900 s)
POST /api/stop
```

The model list auto-discovers any directory under `/home/giles/models/` holding
a `config.json`, so dropping a checkpoint there is the whole install step. A
409 means `_busy` is held by another switch — that lock is a feature, not a
race to retry.

`engine.running: false` with `lastExitReason: "stopped"` is a **deliberate
stop**, not a crash. Check this before diagnosing "the model isn't answering".

## The :1920 orphan — a crash loop that looks like a hang

FreeToken's `torch.distributed` rendezvous binds `serve_port + 1`. Kill the
serve parent and the `TP0-scheduler` child can survive still holding **:1920**.
Every subsequent start then dies with `EADDRINUSE`, and `--auto-restart` spins
in a ~10-second loop that reads as a hung service.

`ft-engine-start.sh` clears exactly the pid holding the port:

```bash
ss -ltnpH "sport = :1920" | grep -oP 'pid=\K[0-9]+' | head -1
```

**Keep that guard surgical.** An earlier, broader match killed the daemon from
its own `ExecStartPost` — the `ExecStart` sibling is not in `ExecStartPost`'s
ancestor chain, so "don't kill my own ancestors" does not protect it.

## Two KV numbers, set independently

```bash
KV_POOL=262144   # --kv-reserve-tokens    GLOBAL pool, shared by every concurrent request
MAX_SEQ=131072   # --max-seq-len-override what Claude Code sees; it still compacts at 131k
```

`--kv-reserve-tokens` is **not optional**. At its default of 8192,
`--moe-cache-auto` claims every spare byte for the expert cache and leaves
0.16 GiB of KV — the first Claude Code message dies with `Prompt is too long`.
The context the model advertises (262,144) is irrelevant; the KV allocation is
the real ceiling. See `claude-code-on-local-models` for why the pool is set to
twice the per-session limit.

Cutting the expert cache to pay for KV is nearly free on this card: two
independent cuts (−15%, then −19%) cost ~0 and +0.1% throughput respectively.
The 3090 is not expert-cache-bound. Spend the VRAM on KV.

**KV_POOL is per-model, and 262144 is not portable.** The default suits Qwen
(20 KiB/token). A model with sliding-window layers costs far more, because the
SWA tier is provisioned at `swa_full_tokens_ratio` (0.2) of the whole token
budget rather than at the window size:

| model | KiB/token | largest pool that fits |
|---|---|---|
| Qwen3.6-35B-A3B-NVFP4 | 20 | 262144 (63% experts cached) |
| Gemma-4-26B-A4B-NVFP4 | 60 | 131072 (40% cached — 1,529 of 3,840) |

Measured on ws03, 24 Aug 2026, both gated on an idle engine:

| | Qwen3.6-35B-A3B | Gemma-4-26B-A4B |
|---|---|---|
| decode, short prompt | **148 tok/s** | 70 tok/s |
| decode at depth | **111** @100k | 49 @81k |
| cold prefill | **30.7 s** @100k | 52.1 s @81k |
| warm TTFT (radix hit) | 0.91 s | **0.83 s** |
| KV pool | **262k** | 131k |
| host RAM pool | 18.19 GB | **12.85 GB** |

Gemma is kept as an option but **Qwen stays the pinned default**. Its sliding-window
layers were expected to make prefill at depth cheaper; they did not, because
attention resolves to `triton` rather than flashinfer for this architecture — the
same thing that happened to gpt-oss-20b. Cheap-looking attention geometry does not
survive contact with the backend that actually serves it.

Start a new model with an explicit pool:

```bash
KV_POOL=131072 MAX_SEQ=131072 ./ft-engine-start.sh /home/giles/models/Gemma-4-26B-A4B-NVFP4
```

If the pool is too big, `--moe-cache-auto` fails in arithmetic before allocating
anything — which is the good outcome, and it hands you the exact per-token cost:

```
AssertionError: cache budget too small: minimum plan (moe=256 slots, kv=262144 pages)
needs 16966811648 B > budget 13163095654 B
```

`needs ÷ pages` is the engine's own bytes-per-token. Halve KV_POOL and retry
rather than reaching for `--memory-ratio`; the (1 − memory_ratio) remainder is
CUDA-graph and activation headroom, not slack.

## Switching the served model

Through the daemon or the panel — never by editing scripts:

```bash
. /home/giles/models/ft-env.sh
ft daemon stop --timeout 40
ft daemon start /home/giles/models/gpt-oss-20b --port 1919 -- \
  --moe-cache-auto --kv-reserve-tokens 131072 --max-seq-len-override 131072
```

A `systemctl --user restart freetoken` always restores Qwen, because the model
is pinned in `ft-engine-start.sh`. Before adding a new checkpoint, run the
`llm-checkpoint-fit` method — this card has no headroom for guesses.

## Two CLI traps

- `ft daemon logs` **streams and never exits.** Always wrap it in `timeout`.
- **A failed start becomes a crash loop.** `--auto-restart` relaunches the engine
  on every failure, so a bad config retries forever and the panel keeps reporting
  `running: true` — the API binds before the weights load, so `answering` is not
  proof of a working engine. Stop it (`ft daemon stop --timeout 40`) before
  diagnosing, or you are reading a log that is being rewritten under you.
- **Every start rebuilds expert banks serially** (`low free RAM -> serial build`),
  which takes minutes on a fresh model and blows through `ft-engine-start.sh`'s
  450 s readiness poll. A "did not become ready in time" from the panel can mean
  *still loading*, not *failed* — check the log before concluding anything.
- `ft ctl` takes `--base-url`, not `--port`:
  `ft ctl --base-url http://127.0.0.1:1919 stats`.

## Never measure against a live session

On a single-GPU box the benchmark and the workload are the same machine. The
first measurement of the KV-pool change showed a catastrophic 47% throughput
collapse; it was a Claude Code session hitting the same engine mid-run.

Gate every run on `ft ctl stats` reporting `active=0` **both before and after**,
and discard contaminated runs. A result that looks dramatic is more likely
contention than discovery.
