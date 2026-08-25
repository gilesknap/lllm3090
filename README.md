# llm3090

Local LLM serving for a single RTX 3090, with a browser control panel.

A llama.cpp engine, a web UI on loopback that starts and stops it and downloads
models, and a curated model list where every entry has been checked to fit
24 GB with a usable context left over.

## Install

Debian 13 or a derivative (Ubuntu 24.04 / 26.04), an RTX 3090, and the NVIDIA
driver already working:

```bash
git clone https://github.com/gilesknap/llm3090
cd llm3090
./install.sh
```

The installer touches nothing outside `$HOME` except a handful of apt packages,
and downloads **no model weights** — you pick those from the panel.

Then open <http://127.0.0.1:8080>, download `Qwen3-8B` (5 GB) to prove the
install works, and `Qwen3.8-27B` (15 GB) for real use.

## Use

```bash
llm3090 models          # what exists, what fits, what is downloaded
llm3090 start Qwen3.8-27B
llm3090 status
llm3090 claude          # launch Claude Code against the local model
llm3090 stop            # free the VRAM
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

<https://gilesknap.github.io/llm3090>
