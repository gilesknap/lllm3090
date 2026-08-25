---
name: llm3090-runbook
description: Operating the llm3090 stack — the panel, the llama.cpp engine, model downloads, and the failure modes that present as a hang. Use when starting, stopping or switching models, when the engine will not come up, when a download stalls, when the panel and the engine disagree, or before measuring anything on the box.
---

# Operating llm3090

One GPU, one engine. Everything below assumes that invariant.

## Layout

```
~/.local/share/llm3090/venv/        the package
~/.local/share/llm3090/llama.cpp/   the pinned engine build
~/.local/state/llm3090/engine.log   what the panel streams
~/.local/state/llm3090/engine.pid   the running engine
~/models/<Name>/*.gguf              checkpoints, one directory each
```

Ports: **1919** engine (OpenAI *and* Anthropic APIs) · **8080** panel.

## The commands

```bash
llm3090 doctor                 # six checks; exits non-zero and says which failed
llm3090 models                 # catalogue: fits, installed, expected speed
llm3090 start <Name> [--ctx N] # stops any running engine first
llm3090 stop                   # frees the VRAM; do this before gaming
llm3090 status
llm3090 claude                 # Claude Code against the local model
```

The panel at `http://127.0.0.1:8080` does the same things and streams the
engine log. It binds loopback **by design** — its endpoints start processes with
no authentication, so remote access is an SSH tunnel, never a LAN bind.

## Failure modes worth knowing

**"Loading" is not "failed".** llama-server binds its HTTP port long before the
weights finish uploading. A 15–21 GB model takes minutes. The panel separates
`loading` from `running` precisely so a slow load does not read as a hang —
check the engine log before concluding anything.

**A start that fails on memory is failing on KV, not weights.** If the weights
fit and the context does not, allocation fails after the model has loaded.
Restart with a smaller `--ctx` rather than a smaller model.

**Downloads resume.** A cancelled or crashed download leaves a `.part` file and
the next attempt continues from it. A repeatedly failing download usually means
the file was renamed upstream, not that the network is bad.

**Never `pkill -f` on a pattern that includes your own command line.** `pkill -f
'llama-server --model'` matches the shell that ran it and kills the script
mid-way. Use the pidfile.

## Before you measure anything

On a single-GPU box the benchmark and the workload are the same machine. Gate
every run on the engine being otherwise idle, and discard anything that
overlapped real use — a live session against the same engine once produced an
apparent 47% throughput regression that was pure contention.

Cold and warm numbers are different measurements and must be reported
separately. The prefix cache means the first turn on a long prompt costs minutes
under Vulkan while every later turn costs seconds; quoting either alone is
misleading.
