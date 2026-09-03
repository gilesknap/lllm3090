# Command line

```
lllm3090 <command> [options]
```

| command | what it does |
|---|---|
| `doctor` | Check the machine can run the stack; exits non-zero on any failure |
| `setup [--model-folder PATH]` | Prepare the machine; choose where checkpoints live |
| `install-engine` | Fetch and checksum-verify the pinned llama.cpp build |
| `fetch-engine [--build TAG]` | Fetch a build to measure against, beside the install rather than over it |
| `build-cuda [--force]` | Compile a CUDA engine for this card, beside the installed one. Never switches to it |
| `models` | The catalogue: size, kind, achievable context, state, expected speed |
| `start <name> [--ctx N] [--parallel N] [--profile P] [--effort LEVEL]` | Stop any running engine and start this model |
| `stop` | Terminate the engine and wait for the VRAM to be released |
| `bench <name>` | Benchmark a model and print a profile block to contribute |
| `sweep [--gpu ID] [--limit N] [--yaml] [--skipped]` | Survey published GGUF models and price them against a card |
| `status` | Whether the engine is running, on what, and whether it answers yet |
| `panel [--port N]` | Run the control panel in the foreground |
| `tui [--url U]` | The panel drawn in the terminal, for a console with no browser |
| `claude [args…]` | Launch Claude Code against the local engine |
| `claude --print-env` | Print that environment instead of launching it |

See [](../how-to/context-and-slots.md) for choosing between them, and
[](../how-to/find-models.md) for widening the list.

`sweep` downloads no weights: it derives each candidate's KV cost from that
model's own `config.json` and runs it through the same arithmetic the panel
uses. `--gpu` prices against any profile in `profiles.yaml` instead of the
detected card. It never produces a speed — `bench` is what does that.

`start` computes the KV pool from what fits, splits it into `--parallel` slots
(default 2, so an agent's subagents have somewhere to go), and gives each slot
as much as it can up to the model's RoPE ceiling. `--ctx` overrides the whole
pool; an unknown GGUF gets a conservative 32768 per slot. Anything after `claude` is passed
through to Claude Code unchanged.

`--profile` picks what the engine guesses ahead with. The default is
multi-token prediction alone at the engine's own draft width, which is what
serves when nothing is asked for. `--profile copy` adds prompt-lookup drafting
and widens the draft to 7, taking the dense 27B from 94.0 to 115.1 tok/s on
copy-heavy work — and it is **refused on Vulkan**, where the same configuration
loses. The verdicts genuinely invert between backends, because verifying a
draft is a batched forward pass and Vulkan gets nothing from a wider batch; see
[](../explanations/going-faster.md).

`build-cuda` compiles llama.cpp at the pinned tag with CUDA, into
`engines/<tag>-cuda-sm<arch>` where the architecture comes from `nvidia-smi`
rather than from anything typed. It needs CUDA 13.3 or newer — not the
distribution's 13.1 — and prints NVIDIA's repository instructions when there is
none. It never replaces the installed engine: point `LLLM3090_LLAMA_DIR` at the
result to use it. `setup` offers the same build when it finds a usable toolkit,
and never builds unprompted, `--yes` included.

`--effort` sets how long the model thinks, for the life of the engine:
`minimal`, `low`, `medium`, `high`, `xhigh` or `max`, handed to the chat
template as llama.cpp's `--reasoning-effort`. Omit it and the model's own
default applies — `xhigh` on `Qwen3.8-27B`. It is a launch argument because
there is nowhere else to put it: a harness cannot set it per request, and
Claude Code's `/effort` does not reach the engine at all
(see [](../how-to/claude-code.md)). A level the model's template does not
implement fails the start rather than producing an engine that 500s on every
request.

`claude --print-env` writes the environment as `export` lines and nothing else
to stdout, so `eval "$(lllm3090 claude --print-env)"` sets up a shell for a
harness this project does not know about. Everything else it has to say —
warnings, refusals — goes to stderr.

## Environment

| variable | default | meaning |
|---|---|---|
| `LLLM3090_MODELS_DIR` | `~/models` | Where GGUF checkpoints live |
| `LLLM3090_STATE_DIR` | `~/.local/state/lllm3090` | Pidfile and engine log |
| `LLLM3090_LLAMA_DIR` | `~/.local/share/lllm3090/llama.cpp` | The engine build |
| `LLLM3090_ENGINE_PORT` | `1919` | Engine port |
| `LLLM3090_PANEL_PORT` | `8080` | Panel port |
| `LLLM3090_PREFIX` | `~/.local/share/lllm3090` | Install prefix (installer only) |
