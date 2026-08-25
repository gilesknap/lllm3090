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
MODELS_DIR = _path("LLM3090_MODELS_DIR", Path.home() / "models")
#: Pidfiles and engine logs.
STATE_DIR = _path("LLM3090_STATE_DIR", Path.home() / ".local/state/llm3090")
#: The unpacked llama.cpp build.
LLAMA_DIR = _path("LLM3090_LLAMA_DIR", Path.home() / ".local/share/llm3090/llama.cpp")

ENGINE_PORT = int(os.environ.get("LLM3090_ENGINE_PORT", "1919"))
PANEL_PORT = int(os.environ.get("LLM3090_PANEL_PORT", "8080"))
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
