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
```

The panel at `http://127.0.0.1:8080` does the same and streams the engine log.
It binds loopback **by design** — its endpoints start processes with no
authentication, so remote access is an SSH tunnel, never a LAN bind.

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

**Downloads resume.** They are threads and die with the panel, leaving a `.part`
file; the panel picks those up on startup. A repeatedly failing download usually
means the file was renamed upstream, not that the network is bad.

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

## Before you measure anything

On a single-GPU box the benchmark and the workload are the same machine.

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
