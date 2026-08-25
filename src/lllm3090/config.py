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
# Hardware envelope
# ---------------------------------------------------------------------------
# This project is deliberately scoped to one card. Every figure in models.yaml
# is sized against it, so serving a different GPU would silently invalidate the
# whole catalogue rather than merely perform differently.

#: Compute capability this project supports (Ampere GA102).
TARGET_COMPUTE_CAPABILITY = "8.6"
#: Total board memory, MiB, as nvidia-smi reports it for a 24 GB card.
TARGET_VRAM_MIB = 24576
#: Minimum driver that carries a working Vulkan ICD for this stack.
MIN_DRIVER_VERSION = 550

#: VRAM held by a typical desktop session (compositor, browser). Subtracted
#: from the budget so the catalogue's "will it fit" answers are honest for a
#: machine someone is also sitting at.
DESKTOP_RESERVE_MIB = 2400
#: Compute buffers, CUDA/Vulkan graphs and fragmentation headroom.
WORKSPACE_RESERVE_MIB = 1024

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

#: Tokens an agent harness spends on system prompt and tool definitions before
#: any of your work, every turn. Claude Code sits around 40k. A model whose
#: per-conversation window is below this cannot run it at all -- the first
#: message fails -- and a window only slightly above it leaves no room to work.
AGENT_PROMPT_FLOOR = 40_000

#: Ceiling on slots handed out automatically.
#:
#: A model that reaches its RoPE ceiling before it exhausts VRAM can have extra
#: slots for free -- the spare cache cannot become context, so it may as well
#: become admission. But llama.cpp sizes some compute buffers per slot, and the
#: value of a fifth concurrent conversation is speculative on a single-GPU box,
#: so the automatic grant stops here. Ask for more explicitly if you want it.
MAX_AUTO_PARALLEL = 4


def usable_vram_mib(desktop: bool = True) -> int:
    """VRAM available for weights plus KV cache.

    Args:
        desktop: subtract a desktop session's allocation. False for a headless
            box, which buys roughly 2.4 GiB and a meaningful slice of context.
    """
    budget = TARGET_VRAM_MIB - WORKSPACE_RESERVE_MIB
    if desktop:
        budget -= DESKTOP_RESERVE_MIB
    return budget
