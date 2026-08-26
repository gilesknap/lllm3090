"""Paths, ports and the hardware envelope this project targets.

Everything is overridable by environment variable so the test suite and a
second machine do not have to agree with the defaults.
"""

from __future__ import annotations

import os
from pathlib import Path


def _path(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, default)).expanduser()


#: Where GGUF checkpoints live. One directory per model.
MODELS_DIR = _path("LLLM3090_MODELS_DIR", Path.home() / "models")
#: Pidfiles and engine logs.
STATE_DIR = _path("LLLM3090_STATE_DIR", Path.home() / ".local/state/lllm3090")
#: The unpacked llama.cpp build.
LLAMA_DIR = _path("LLLM3090_LLAMA_DIR", Path.home() / ".local/share/lllm3090/llama.cpp")

ENGINE_PORT = int(os.environ.get("LLLM3090_ENGINE_PORT", "1919"))
PANEL_PORT = int(os.environ.get("LLLM3090_PANEL_PORT", "8080"))
ENGINE_URL = f"http://127.0.0.1:{ENGINE_PORT}"

ENGINE_LOG = STATE_DIR / "engine.log"
ENGINE_PID = STATE_DIR / "engine.pid"

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
# Capacity and compute capability come from a profile (see hardware.py), not
# from constants here: the same catalogue has to tell the truth on a 16 GB card
# as on a 24 GB one. What stays here is what is card-independent.

#: The profile whose card the catalogue's speeds were measured on.
REFERENCE_PROFILE = "rtx-3090"

#: Minimum driver that carries a working Vulkan ICD for this stack.
MIN_DRIVER_VERSION = 550

#: VRAM held by a typical desktop session (compositor, browser). Subtracted so
#: "will it fit" is honest for a machine someone is also sitting at.
DESKTOP_RESERVE_MIB = 2400
#: Compute buffers, CUDA/Vulkan graphs and fragmentation headroom.
WORKSPACE_RESERVE_MIB = 1024

#: Extra VRAM held back when a multimodal projector is loaded, on top of the
#: workspace reserve above. The vision tower needs its own compute buffers, and
#: they are not the projector file's size: measured on a 3090, Gemma-4-26B-A4B's
#: 1.19 GB projector cost 1376 MiB resident, and at a full KV pool the engine
#: then loaded happily and failed *every* request with
#: ``vk::Device::allocateMemory: ErrorOutOfDeviceMemory``. Counting only the file
#: promises context the card cannot serve.
VISION_WORKSPACE_RESERVE_MIB = 1024

#: Tokens an agent harness spends on system prompt and tool definitions before
#: any of your work, every turn. Claude Code sits around 40k. A model whose
#: per-conversation window is below this cannot run it at all -- the first
#: message fails -- and a window only slightly above it leaves no room to work.
AGENT_PROMPT_FLOOR = 40_000

#: How many conversations must fit at once.
#:
#: The KV cache is a single pool shared by every concurrent request, not a
#: per-conversation budget. An agent that spawns subagents therefore needs room
#: for more than one: with a pool sized for exactly one session, a parent
#: holding most of it leaves nowhere to admit a subagent, so the scheduler
#: serialises them -- and the subagent's prefill evicts the parent's cached
#: prefix, so the parent then pays a full cold prefill on its next turn.
#: Two is the minimum that keeps a parent and one subagent resident together.
DEFAULT_PARALLEL = 2

#: Ceiling on slots handed out automatically.
#:
#: A model that reaches its RoPE ceiling before it exhausts VRAM can have extra
#: slots for free -- the spare cache cannot become context, so it may as well
#: become admission. But llama.cpp sizes some compute buffers per slot, and the
#: value of a fifth concurrent conversation is speculative on a single-GPU box,
#: so the automatic grant stops here. Ask for more explicitly if you want it.
MAX_AUTO_PARALLEL = 4
