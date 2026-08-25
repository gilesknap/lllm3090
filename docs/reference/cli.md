# Command line

```
lllm3090 <command> [options]
```

| command | what it does |
|---|---|
| `doctor` | Check the machine can run the stack; exits non-zero on any failure |
| `install-engine` | Fetch and checksum-verify the pinned llama.cpp build |
| `models` | The catalogue: size, kind, achievable context, state, expected speed |
| `start <name> [--ctx N] [--parallel N]` | Stop any running engine and start this model |
| `stop` | Terminate the engine and wait for the VRAM to be released |
| `bench <name>` | Benchmark a model and print a profile block to contribute |
| `status` | Whether the engine is running, on what, and whether it answers yet |
| `panel [--port N]` | Run the control panel in the foreground |
| `claude [args…]` | Launch Claude Code against the local engine |

See [](../how-to/context-and-slots.md) for choosing between them.

`start` computes the KV pool from what fits, splits it into `--parallel` slots
(default 2, so an agent's subagents have somewhere to go), and gives each slot
as much as it can up to the model's RoPE ceiling. `--ctx` overrides the whole
pool; an unknown GGUF gets a conservative 32768 per slot. Anything after `claude` is passed
through to Claude Code unchanged.

## Environment

| variable | default | meaning |
|---|---|---|
| `LLLM3090_MODELS_DIR` | `~/models` | Where GGUF checkpoints live |
| `LLLM3090_STATE_DIR` | `~/.local/state/lllm3090` | Pidfile and engine log |
| `LLLM3090_LLAMA_DIR` | `~/.local/share/lllm3090/llama.cpp` | The engine build |
| `LLLM3090_ENGINE_PORT` | `1919` | Engine port |
| `LLLM3090_PANEL_PORT` | `8080` | Panel port |
| `LLLM3090_PREFIX` | `~/.local/share/lllm3090` | Install prefix (installer only) |
