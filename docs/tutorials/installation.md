# Installation

## What you need

- **An RTX 3090** (24 GB, compute capability 8.6). The installer checks and
  warns loudly on anything else — see [](../explanations/one-gpu.md).
- **Debian 13 or a derivative** — Ubuntu 24.04 and 26.04 are tested.
- **A working NVIDIA driver**, 550 or newer. `nvidia-smi` must run.
- **~20 GB free disk** for a first model, more if you collect them.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/gilesknap/lllm3090/main/install.sh | bash
```

:::{tip}
Piping a script from the internet into a shell means trusting whoever controls
that URL. If you would rather look first — and you should, with anything that
runs `sudo` — download it, read it, then run it:

```bash
curl -fsSL https://raw.githubusercontent.com/gilesknap/lllm3090/main/install.sh -o install.sh
less install.sh
bash install.sh
```
:::

Cloning does the same thing, and is what you want if you plan to change
anything — run from a checkout, the installer installs *that checkout* rather
than the published package:

```bash
git clone https://github.com/gilesknap/lllm3090
cd lllm3090 && ./install.sh
```

To install a branch or a fork instead, set `LLLM3090_SOURCE` to anything pip
understands:

```bash
LLLM3090_SOURCE="git+https://github.com/you/lllm3090.git@my-branch" bash install.sh
```

The installer:

1. checks the OS, GPU, driver and Vulkan ICD, and stops with a specific message
   rather than a stack trace if something is missing;
2. installs four apt packages if absent (`python3-venv`, `python3-pip`,
   `libvulkan1`, `curl`) — the only thing it touches outside `$HOME`;
3. creates a virtualenv at `~/.local/share/lllm3090/venv` and installs the
   package — from PyPI when piped, or from the checkout when run inside one;
4. downloads a **pinned** llama.cpp build and verifies its SHA-256;
5. installs and starts a systemd *user* service for the panel, and enables
   linger so it survives logout.

It downloads **no model weights**. That is deliberate: the right first model
depends on what you want to do, and a 15 GB surprise during an install is rude.

## Check it worked

```bash
lllm3090 doctor
```

```
  [ ok ] os           Ubuntu 26.04 LTS
  [ ok ] gpu          NVIDIA GeForce RTX 3090 (24576 MiB, compute 8.6)
  [ ok ] driver       driver 595.84
  [ ok ] vulkan       Vulkan ICD at /usr/share/vulkan/icd.d/nvidia_icd.json
  [ ok ] engine       ~/.local/share/lllm3090/llama.cpp/llama-server
  [ ok ] models dir   /home/giles/models (766 GB free)
```

Then open <http://127.0.0.1:8080>.

:::{note}
The panel binds loopback only, by design: its endpoints start processes and
write to disk with no authentication. To reach it from another machine use an
SSH tunnel — see [](../how-to/remote-access.md).
:::

## Why Vulkan and not CUDA

llama.cpp publishes prebuilt CUDA binaries for Windows only. Building the CUDA
backend on Linux needs a host compiler CUDA accepts (GCC ≤ 14), which recent
Debian and Ubuntu releases no longer default to. The prebuilt **Vulkan** binary
needs no compiler, works on any NVIDIA driver with the Vulkan ICD installed, and
decodes at close to CUDA speed.

The trade-off is prompt processing: Vulkan prefill is roughly 3–4× slower than
CUDA. That shows up as a slow *first* turn on a very long prompt (about two
minutes at 80k tokens) and not at all afterwards, because the prefix cache
carries the prompt between turns.
