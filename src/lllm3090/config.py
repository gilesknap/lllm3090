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
#: Builds kept for measurement, one directory per upstream tag. Separate from
#: LLAMA_DIR because deciding whether to move the pin means running a candidate
#: against the incumbent, which is impossible if measuring one replaces it.
ENGINES_DIR = _path(
    "LLLM3090_ENGINES_DIR", Path.home() / ".local/share/lllm3090/engines")

ENGINE_PORT = int(os.environ.get("LLLM3090_ENGINE_PORT", "1919"))
PANEL_PORT = int(os.environ.get("LLLM3090_PANEL_PORT", "8080"))
ENGINE_URL = f"http://127.0.0.1:{ENGINE_PORT}"
#: Where a front end that is not a browser goes looking for the panel.
PANEL_URL = f"http://127.0.0.1:{PANEL_PORT}"

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

#: VRAM the driver holds back before any process allocates a byte -- page
#: tables, the console framebuffer, and the card's own bookkeeping.
#: ``nvidia-smi`` reports it as ``memory.reserved`` and it is *not* part of
#: what a process can claim, so a budget computed from the nameplate capacity
#: overstates the card by exactly this much.
#:
#: Measured at 451 MiB on the 3090 here, on a text console with nothing else
#: running. Leaving it out is what let a plan of 2 x 262144 tokens be issued
#: with 52 MiB of margin against a card that had already given 451 away: the
#: engine loaded, prefill degraded from 88 to 21 tok/s over three batches as
#: compute buffers fought for room that was not there, and the run ended in
#: ``vk::DeviceLostError`` with the GPU spinning at 100% and zero memory
#: traffic. This is the fallback for a profile that is not the running card;
#: ``hardware.detect`` substitutes the live figure when nvidia-smi reports one.
DRIVER_RESERVE_MIB = 512

#: What a token of KV cache really costs, against the nominal
#: ``kv_kib_per_token``. The nominal figure is the tensor arithmetic; llama.cpp
#: also carries per-cell bookkeeping and allocates the pool whole at load, so
#: resident cost runs above it.
#:
#: Measured on Gemma-4-26B-A4B at two pool sizes 344k tokens apart: solving the
#: two peaks for a fixed cost plus a per-token cost gives 11.2 KiB/token against
#: a nominal 10, and the implied fixed cost agreed between the two runs to
#: within 1 MiB. Without this, a plan sized to the last byte of the nominal
#: cache overruns the card by 12% of the pool.
KV_OVERHEAD_FACTOR = 1.12

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

#: How many conversations an agent harness needs at once.
#:
#: The KV cache is a single pool shared by every concurrent request, not a
#: per-conversation budget. An agent that spawns subagents therefore needs room
#: for more than one: with a pool sized for exactly one session, a parent
#: holding most of it leaves nowhere to admit a subagent, so the scheduler
#: serialises them -- and the subagent's prefill evicts the parent's cached
#: prefix, so the parent then pays a full cold prefill on its next turn.
#: Two is the minimum that keeps a parent and one subagent resident together.
#:
#: **This is no longer what ``plan()`` hands out automatically.** It used to
#: be, which meant every model's window was halved whether or not the second
#: half was ever used -- the 35B-A3B gave 169k twice on a desktop when one
#: conversation could have had the full 256k. ``plan()`` now fills the model's
#: ceiling first and grants a second slot only where it costs nothing (see
#: :func:`lllm3090.catalog.plan`). This value remains what an agent should ask
#: for explicitly -- ``lllm3090 start <model> --parallel 2`` -- and what an
#: uncatalogued GGUF falls back to.
DEFAULT_PARALLEL = 2

#: How much more total context a split must buy before it is worth taking.
#:
#: Filling one conversation to the model's RoPE ceiling is the priority, but it
#: is not worth stranding the rest of the card to do it. Where the pool is much
#: larger than the ceiling, refusing to split wastes the difference: a model
#: whose pool holds 2.8 windows gets one full window and 1.8 windows of cache
#: that nothing can ever use.
#:
#: So a split is taken when it raises *total* usable context by this factor or
#: more, and the number of slots is then the fewest that consume the whole pool
#: -- which keeps each window as long as it can be. Below the threshold the
#: single conversation keeps the ceiling and the remainder is accepted as
#: unusable, because halving a window to recover a little cache is a bad trade.
#:
#: 1.5 is a judgement, not a measurement. Raise it to favour one long
#: conversation, lower it to favour concurrency.
SLOT_SPLIT_GAIN = 1.5

#: Ceiling on slots handed out automatically.
#:
#: A model that reaches its RoPE ceiling before it exhausts VRAM can have extra
#: slots for free -- the spare cache cannot become context, so it may as well
#: become admission. But llama.cpp sizes some compute buffers per slot, and the
#: value of a fifth concurrent conversation is speculative on a single-GPU box,
#: so the automatic grant stops here. Ask for more explicitly if you want it.
MAX_AUTO_PARALLEL = 4
