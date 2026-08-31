# Installation

## What you need

- **An RTX 3090** (24 GB, compute capability 8.6). The installer checks and
  warns loudly on anything else — see [](../explanations/one-gpu.md).
- **Debian 13 or a derivative** — Ubuntu 24.04 and 26.04 are tested.
- **A working NVIDIA driver**, 550 or newer. `nvidia-smi` must run.
- **~20 GB free disk** for a first model, more if you collect them.

## Install

[uv](https://docs.astral.sh/uv/) manages the tool and brings its own Python, so
there is nothing to build and no virtualenv to think about:

```bash
# uv, if you do not have it: https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh

uv tool install lllm3090
lllm3090 setup
```

The uv installer itself needs `curl`, which a desktop install will have but a
minimal Debian image will not (`apt install curl`). See uv's
[installation docs](https://docs.astral.sh/uv/getting-started/installation/) for
other methods, including `pipx` and a standalone binary.

`uv tool install` puts `lllm3090` on your `PATH`; `lllm3090 setup` does
everything uv cannot do for itself:

1. checks the OS, GPU, driver and Vulkan ICD, and stops with a specific message
   rather than a stack trace if something is missing;
2. installs `libvulkan1` if absent — the only thing it touches outside `$HOME`,
   and the only apt package the engine needs;
3. downloads a **pinned** llama.cpp build and verifies its SHA-256;
4. writes a systemd *user* unit for the panel and starts it.

It downloads **no model weights**. That is deliberate: the right first model
depends on what you want to do, and a 15 GB surprise during setup is rude.

`setup` is safe to re-run — every step is skipped when already done — so it
doubles as the repair command after an upgrade.

### Installing something other than the release

To run a branch, a fork, or a local checkout, point `uv` at it:

```bash
uv tool install "git+https://github.com/gilesknap/lllm3090.git@my-branch"
uv tool install --editable .        # from a checkout, for development
```

Then `lllm3090 setup` as before.

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

## Why Vulkan by default, and what CUDA would buy

llama.cpp publishes prebuilt CUDA binaries for Windows only, so on Linux CUDA
means installing a 4–6 GB toolkit and compiling. The prebuilt **Vulkan** binary
needs neither, works on any NVIDIA driver with the Vulkan ICD installed, and
arrives as one download verified against a digest recorded in this repository.
That is why it is what installs.

CUDA is faster. Measured here on an RTX 3090, both builds from the same
llama.cpp commit, on the dense 27B — the numbers that matter are the ones with
multi-token prediction on, because that is how the engine has always run:

| | Vulkan | CUDA | |
|---|---|---|---|
| cold prefill, 80k tokens | 118.2 s | 90.0 s | 1.31× |
| decode, no speculation | 32.9 tok/s | 42.2 tok/s | 1.28× |
| decode, as each backend serves it | 54.8 tok/s | 84.9 tok/s | **1.55×** |
| decode, copy-heavy, best config | 59.7 tok/s | 115.1 tok/s | **1.93×** |

The first turn on a very long prompt is slow either way — about two minutes at
80k against a minute and a half — and not slow at all afterwards, because the
prefix cache carries the prompt between turns.

`lllm3090 setup` offers to build a CUDA engine when it finds a toolkit new
enough, and `lllm3090 build-cuda` does it on its own. Neither builds unprompted,
and neither switches: Vulkan keeps serving until you point
`LLLM3090_LLAMA_DIR` at what was built. What it costs:

- **CUDA 13.3 or newer, from NVIDIA's repository.** Not `apt install
  nvidia-cuda-toolkit` — Ubuntu 26.04's 13.1 declares `rsqrt`/`rsqrtf` without
  an exception specifier where glibc 2.43 declares them `noexcept(true)`, and
  `nvcc` then refuses every file that includes `<math.h>`.
- **A binary tied to one card.** Compiled `sm_<arch>-real` from what
  `nvidia-smi` reports, so the directory is named `b10715-cuda-sm86` and a card
  swap is a legible absence rather than a mystery crash.
- **About 14% of the context window.** The dense 27B holds 196k tokens on
  Vulkan and 168k on CUDA: KV costs ~10% more per token there and the backend
  carries ~230 MiB more fixed overhead.
- **A weaker identity.** A downloaded build is a tag and a digest reviewed in a
  diff. A compiled one reports `build 1, commit <sha>`, because a shallow clone
  has no tag history — only the commit is real, and nobody attests to the
  binary. That is a different promise, and it is the reason nothing switches
  the active engine for you.

See [](../explanations/going-faster.md) for what the trade is made of.
