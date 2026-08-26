# lllm3090

Local LLM serving for a single RTX 3090, with a browser control panel.

A llama.cpp engine, a web UI on loopback that starts and stops it and downloads
models, and a curated model list where every entry has been checked to fit
24 GB with a usable context left over.

## Install

Debian 13 or a derivative (Ubuntu 24.04 / 26.04), an RTX 3090, and the NVIDIA
driver already working:

```bash
# uv, if you do not have it: https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh

uv tool install lllm3090
lllm3090 setup
```

`setup` checks the hardware, installs the one apt package the engine needs,
fetches a pinned llama.cpp build and starts the panel as a user service. It is
safe to re-run and skips whatever is already done.

It touches nothing outside `$HOME` except `libvulkan1`, and downloads **no model
weights** — you pick those from the panel.

Then open <http://127.0.0.1:8080>, download `Qwen3-8B` (5 GB) to prove the
install works, and `Qwen3.6-35B-A3B` (17.7 GB) for real use — at 126 tok/s
it is the fastest model that also reaches a full 262k context, which is what
agentic work needs. `gpt-oss-20b` decodes faster still, at 160, in half the
room.

## Use

```bash
lllm3090 models          # what exists, what fits, what is downloaded
lllm3090 start Qwen3.6-35B-A3B
lllm3090 status
lllm3090 claude          # launch Claude Code against the local model
lllm3090 stop            # free the VRAM
```

The engine exposes both the OpenAI API (`/v1/chat/completions`) and Anthropic's
(`/v1/messages`) on `127.0.0.1:1919`, so Claude Code and OpenAI-compatible
clients both work against it without a translation proxy.

## Why it is scoped to one GPU

Every figure in the model catalogue — download size, resident VRAM, KV cache
cost per token, achievable context, expected tokens per second — is computed
for 24 GB of GDDR6X at compute capability 8.6. On another card the software
would still run and every number would be wrong, so the installer checks and
warns.

## Documentation

<https://gilesknap.github.io/lllm3090>

## The panel

![The lllm3090 control panel](docs/images/panel.png)

Engine state and VRAM at the top, the models you have with start/stop, the
curated list with what fits this card and what it will do, and the engine log
streaming underneath. Downloads run in the background with progress, and resume
from a part file if interrupted.
