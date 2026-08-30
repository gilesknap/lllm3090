---
name: freetoken-engine
description: FreeToken, the MoE expert-offload engine — what it buys over llama.cpp, the version pin chain its install depends on, and every failure mode hit while running it. Reference material for adding it to lllm3090 as a second engine (phase 2); not currently installed. Use when working on FreeToken integration, when an offloaded-MoE model will not load, when deciding whether a model needs offload at all, or when asking whether a given checkpoint is reachable on this box at all — for FreeToken the ceiling is host RAM rather than VRAM and this board caps at 128 GB, but a disk-streaming engine (Colibri) moves that ceiling to free disk and puts 200-400 GB checkpoints in reach at a speed recorded here.
---

# FreeToken (phase 2 — not currently installed)

lllm3090 ships llama.cpp only. FreeToken is the engine that would be added
alongside it, and this file is what was learned running it so that work does not
start from zero.

## What it buys, and what it costs

FreeToken keeps a **MoE model's routed experts in host RAM** and pages them to
the GPU through an LRU cache, so a checkpoint larger than VRAM can serve. That
is the whole reason to want it: it puts models on the card that otherwise need
different hardware.

Measured on a 3090 with Qwen3.6-35B-A3B-NVFP4 (23.4 GB checkpoint, 18.19 GB of
it experts in host RAM): **148 tok/s** short-prompt decode, 111 tok/s at 100k
context, 262k KV pool. llama.cpp cannot do this — it wants the whole GGUF
resident.

The cost is fragility. Everything in the next section is a pin, and taking the
newest release of any of them breaks the build.

## What it can and cannot reach on this box

**The ceiling is host RAM, not VRAM.** Keeping experts in host RAM is the
mechanism, so the largest checkpoint FreeToken can serve is bounded by what the
machine can hold — and on this box that is a hard, low number:

```
installed                     32 GB  (~30 GB usable)
ASRock X570M Pro4 maximum    128 GB  (4 x DDR4 DIMM, 32 GB per module)
```

DDR4 unbuffered non-ECC tops out at 32 GB per module, so 128 GB is the physical
end of this board. 192 GB needs 48/64 GB DIMMs, which are DDR5 only — a new
platform, not an upgrade.

**The target category is 20-28 GB checkpoints**, not enormous ones. The measured
case was Qwen3.6-35B-A3B-NVFP4: a 23.4 GB checkpoint with 18.19 GB of experts in
host RAM, 148 tok/s and a 262k KV pool.

**The payoff is context as much as capacity.** Moving experts out of VRAM frees
that VRAM for KV cache, which rescues a specific failure llama.cpp cannot avoid:
a checkpoint that loads but leaves no room to think in. `Ornith-1.5-35B-A3B` at
21.7 GB is the live example — it fits a 24 GB card and leaves **5k tokens** of
context, far under the 40k agent floor. Under FreeToken the experts leave VRAM
and that space becomes the cache.

**What is out of reach for FreeToken.** DeepSeek-V4-Flash needs a 147 GB expert
pool. That does not fit in 30 GB, and it does not fit in 128 GB either, so
maxing this board out does not reach it. For a RAM-offload engine it needs a
192 GB machine, where the pool sits inside the VRAM allocation and there is no
offload at all — see `local-inference-hardware`. That much is settled; do not
re-derive it.

**It is not out of reach absolutely, and that changed in 2026.** This paragraph
used to say "will stay out of reach", which was reasoning about RAM stated as if
it were about physics. [Colibri](https://github.com/JustVugg/colibri) (July
2026, pure C) streams routed experts from **disk** rather than host RAM, so the
ceiling becomes free disk space instead of DIMM slots. It lists DeepSeek-V4-Flash
at ~167 GB of disk and 16–32 GB of RAM, and GLM-5.2 (744B) at ~372 GB and 24 GB.
Both load on ws03 as it stands — `/` is a 4 TB NVMe with 3.2 TB free.

**The speed verdict stands, and is now measured rather than estimated.** The
2 tok/s figure above assumed a 7 GB/s Gen4 NVMe. Measured on this box with
`O_DIRECT`, 30 Aug 2026:

| | Lexar NQ790 (`/`, NVMe Gen4) | Crucial MX500 (`/home`, **SATA**) |
|---|---|---|
| sequential | 4.4 GB/s | 0.53 GB/s |
| random 64 KiB | 1.31 GB/s | 0.13 GB/s |
| random 1 MiB | 5.15 GB/s | 0.37 GB/s |

So the real disk is *below* what the estimate assumed, and Colibri's own figures
land below it in turn: 0.05–0.1 tok/s on a 25 GB laptop, ~1.8 tok/s on a 128 GB
CPU desktop, 1.07 on a single RTX 5070 Ti. Against a resident 126 tok/s that is
two to three orders of magnitude, and it cannot run an agent harness at all —
40k tokens of system prompt arrive before the first word.

**So: a capability, not a usable one.** Reach for it only for a single
overnight question where frontier quality beats latency, and only with the
checkpoint on `/`. `~/models` was on the SATA drive until Aug 2026 and the gap
is widest (14x) at exactly the large random reads expert streaming does.

## The version pin chain

Four things are held down, three of them downgrades from what pip installs by
default:

| package | pinned to | why |
|---|---|---|
| python | 3.12.11 | the wheel ecosystem (torch, flashinfer, sglang-kernel) had no 3.14 builds |
| `nvidia-cuda-nvcc` | 13.0.88 | must match PyTorch's `cu130`; latest gives 13.3.73 and fails flashinfer's bundled CCCL header check |
| `nvidia-cuda-crt`, `nvidia-nvvm` | 13.0.88 | pulled in alongside nvcc, must be dragged back in lockstep — a mixed toolchain fails at compile time, not install time |
| `flashinfer-python` | 0.6.17 | its vendored CCCL headers are what fix CUDA to the 13.0 line in the first place |

**Naming trap:** `nvidia-cuda-nvcc-cu13` is deprecated and fails to build. The
one that works is plain `nvidia-cuda-nvcc`.

**After changing any pin, clear both JIT caches** — `rm -rf ~/.cache/flashinfer
~/.cache/tvm-ffi` — or a stale compiled artefact reproduces the old failure and
the fix looks like it did nothing.

## The four fixes a recent distro needed

None were misconfiguration; each unblocked the next failure. Ubuntu 26.04 ships
glibc 2.43 and GCC 15.2, both ahead of what the CUDA toolkit supports. **Debian
13 ships older versions, so fixes 1 and 3 may not be needed there — verify
rather than assume.**

1. **Pin nvcc to the 13.0 line.** Latest nvcc collides with flashinfer's
   vendored CCCL: `"CUDA compiler and CUDA toolkit headers are incompatible"`.
2. **Add unversioned `.so` symlinks and a `lib64 -> lib` alias.** The pip CUDA
   wheels ship `libcudart.so.13` but no `libcudart.so`, which is what `-lcudart`
   resolves against. Twenty symlinks.
3. **Patch `crt/math_functions.h`.** glibc 2.43 declares `rsqrt`/`rsqrtf` with an
   exception specification CUDA's headers disagree with; adding `noexcept (true)`
   to both resolves it. Upstream, not local — NVIDIA does not yet support
   glibc 2.41+.
4. **Export the CUDA environment in the launcher.** CUDA lives in site-packages
   rather than `/usr/local/cuda`, so the JIT compilers need `CUDA_HOME`,
   `CUDA_PATH`, `LIBRARY_PATH`, `LD_LIBRARY_PATH` and `CPATH` set explicitly.
   That is the entire reason a wrapper script exists.

## Backend selection on Ampere

FreeToken picks its NVFP4 expert kernel from compute capability, and on sm_86
the choice is narrower than it looks:

- `marlin` — documented as the sm_80–99 fast path, but it borrows kernels from
  vLLM and refuses to load without it rather than degrading.
- `flashinfer` / `b12x` — needs sm_120+. Permanently unavailable on a 3090.
- `triton` — the portable inline-dequant GEMV, and what `auto` settles on.

So the 148 tok/s figure is the *fallback* path. Installing vLLM would unlock
Marlin; untested, and the evidence suggests it would help cold prefill rather
than decode.

## A checkpoint that loads in llama.cpp may not load here

FreeToken's dense path sets `attn_quant = "nvfp4"` for **any** compressed-tensors
checkpoint showing one NVFP4 group, then looks for `.weight_packed` on the
attention projections. Builds that are NVFP4 on the MLP but **channel-wise fp8
W8A8** on attention (the unsloth and RedHatAI Qwen3.8-27B releases are both like
this) fail at load — it supports per-tensor fp8 and 128x128 block fp8, not
channel-wise W8A8. Same model, same bit width, different vendor, different
outcome. Check `quantization_config.config_groups` and its `targets` regexes
before downloading.

## Layout, as it was on ws03

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

## How it was wired (for reference)

```bash
systemctl --user status freetoken ft-control
systemctl --user restart freetoken     # always comes back on Qwen: the model is pinned in ft-engine-start.sh
systemctl --user stop freetoken        # frees all 23 GB — do this before gaming
```

`Linger=yes`, so it starts at boot without a login. A full engine crash
self-heals in ~80 s. The JIT caches survive restarts: a warm start reaches first
token in ~0.75 s, against 64 s for the very first build on a cold cache.

Because CUDA lives in site-packages rather than `/usr/local/cuda`, **any** shell
that touches `ft` must source the shared env script first. That is the only
reason those wrappers existed, and any phase-2 installer needs the equivalent.

## What the phase-2 integration has to do

lllm3090's panel and `engine.py` assume one process, one pidfile, one port.
FreeToken does not fit that shape without work:

- It runs a **supervisor** (`ft daemon`, port 1900) that owns the engine
  (port 1919), rather than being a single process to signal. Start and stop go
  through the daemon's HTTP control plane, not a pidfile.
- Its models are **directories of safetensors with a `config.json`**, not GGUF
  files, so `catalog.installed()` needs a second discovery rule and entries need
  a `kind` so the panel dispatches to the right engine.
- Both engines want **port 1919**, which is fine and deliberate — only one can
  have the GPU — but the start path must stop the other first, unconditionally.
  A stale process of either kind makes the other die on bind.
- The engine binds `serve_port + 1` for `torch.distributed` (see below), so
  port 1920 must be treated as part of the engine's footprint.

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

## CLI traps

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

## Measurement discipline (see also lllm3090-runbook)

On a single-GPU box the benchmark and the workload are the same machine. The
first measurement of the KV-pool change showed a catastrophic 47% throughput
collapse; it was a Claude Code session hitting the same engine mid-run.

Gate every run on `ft ctl stats` reporting `active=0` **both before and after**,
and discard contaminated runs. A result that looks dramatic is more likely
contention than discovery.
