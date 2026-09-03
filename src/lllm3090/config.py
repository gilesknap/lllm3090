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

#: The backend they were measured on, which is also the one that installs.
#:
#: A speed is a speed *of* a configuration, and since there can now be two
#: backends on one machine, the card is no longer the whole of "of". The same
#: dense 27B serves 54.8 tok/s under Vulkan and 84.9 under CUDA -- a bigger
#: spread than most of the catalogue's entries have between them. So a figure
#: shown without saying which backend produced it is not a small imprecision,
#: it is a number that is wrong by more than the thing it is being compared to.
#:
#: Capacity does not work this way and is not treated this way: what fits is
#: arithmetic and travels to any card, so it is recomputed rather than
#: qualified. Speed is a measurement and does not travel at all.
REFERENCE_BACKEND = "vulkan"

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

# ---------------------------------------------------------------------------
# What a token of KV cache really costs
# ---------------------------------------------------------------------------
# This used to be one constant, ``KV_OVERHEAD_FACTOR = 1.12``, calibrated on
# Gemma-4-26B-A4B: a Vulkan engine running a model with no MTP head. It was
# covering three unrelated things at once -- allocator overhead, the backend,
# and multi-token prediction's extra cache -- and being wrong about a fourth,
# because the caller divided the nominal f16 figure by two to get q8_0.
#
# Splitting it is not tidying. Two of those terms move independently now that a
# machine can carry two backends and the engine can turn speculation on by
# itself, so a single number is wrong in a different direction for each
# combination.
#
# What is deliberately NOT attempted here: a per-model measured constant.
# Resident VRAM is not linear in context on a hybrid MoE -- Qwen3.6-35B-A3B-MTP
# reads 8.99, 11.57 and 16.78 KiB/token by segment where the dense 27B is
# linear at ~35.0 to within 0.5% -- so a measured "KiB per token" for it is an
# average over a curve and depends on which two points produced it. Every term
# below is either arithmetic or a ratio in which the curvature divides out.

#: Bytes per value of a ``q8_0`` cache, against ``f16``'s two.
#:
#: ``q8_0`` is 34 bytes per block of 32 values -- 32 quantised bytes plus an
#: f16 scale -- so it is 1.0625 bytes per value and **0.53125x** of f16, not
#: half. Every call site divided ``kv_kib_per_token`` by two, which under-counts
#: the cache by 6% on every model in the catalogue.
Q8_0_RATIO = 34 / 32 / 2

#: What the allocator holds on top of the tensor arithmetic.
#:
#: The nominal per-token figure is the tensors; llama.cpp also carries per-cell
#: bookkeeping and allocates the pool whole at load. This is what is left after
#: the backend and MTP are named separately, and the one term with no
#: derivation behind it.
#:
#: **This is the old 1.12, restated rather than re-measured.** That figure came
#: from Gemma-4-26B-A4B at two pool sizes 344k tokens apart: resident cost
#: solved to 11.2 KiB/token, against a nominal that had been computed as half
#: of f16. The measurement was right and the attribution was not -- the true
#: q8_0 nominal is 10.625, so the allocator's share is 11.2/10.625, and the
#: rest of what 1.12 appeared to be was the arithmetic error above.
#:
#: Which is why correcting ``Q8_0_RATIO`` must not simply make everything 6%
#: more conservative: that would double-count an error the constant had already
#: absorbed, and would price Gemma 6% above what it was measured at. Written
#: this way, a model with no MTP head is priced *identically* to before, and
#: the only entries that move are the ones that were never paying for their
#: draft cache.
ALLOCATOR_OVERHEAD = 1.12 / 2 / Q8_0_RATIO

#: What each backend multiplies the per-token cost by, against Vulkan.
#:
#: The only term the measurements showed generalising. CUDA costs
#: **1.100-1.136x** per token across two models and both speculation settings;
#: because it is a ratio of two measurements taken the same way, the
#: non-linearity above divides out of it. The top of the range is taken, since
#: this multiplies a budget rather than reporting a result.
BACKEND_KV_FACTOR = {"cuda": 1.136}

#: VRAM a backend holds before any cache is allocated, beyond Vulkan's.
#:
#: CUDA reports 24125 MiB of device memory where Vulkan reports 24822 on the
#: same card, and ~230 MiB of that difference shows up as a flat cost rather
#: than as anything per-token. Flat, so it is taken off the budget rather than
#: multiplied into the slope.
BACKEND_FIXED_MIB = {"cuda": 230}

#: What multi-token prediction's own cache costs, as a multiple of one
#: full-attention layer.
#:
#: The MTP head **is** one more full-attention layer, which is why
#: ``block_count`` in the GGUF header reads 65 for a 64-layer model. Predicting
#: it as one layer gives 4.00 KiB/token for the dense 27B and 2.00 for the
#: 35B-A3B against 4.80 and 2.46 measured at f16 -- a consistent x1.20 and
#: x1.23, which this carries.
#:
#: The head's cache is quantised like any other now (see
#: ``lllm3090.engine.CACHE_TYPE``), so the layer is priced at ``Q8_0_RATIO``
#: rather than at f16. That is the whole of what quantising the draft cache
#: bought: 2.45 KiB/token back on the dense 27B, about 412 MiB at a 168k window.
MTP_LAYER_OVERHEAD = 1.25

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


#: Where the chosen engine is remembered, so that a choice survives a panel
#: restart and a package upgrade.
#:
#: In the state directory rather than beside the code: ``uv tool install
#: --force`` replaces the package wholesale, and a preference that lived there
#: would be silently reset by an upgrade -- putting the user back on Vulkan
#: with the panel still showing what they picked.
ENGINE_CHOICE = STATE_DIR / "engine.json"

#: Whether ``LLLM3090_LLAMA_DIR`` was set in the environment.
#:
#: An explicit override outranks anything stored, and the front ends need to
#: know the difference to say *why* the switch is unavailable rather than
#: showing a control that silently does nothing.
LLAMA_DIR_FROM_ENV = "LLLM3090_LLAMA_DIR" in os.environ

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

#: Where the published documentation lives.
#:
#: The panel is loopback-only and these links are not, so on a machine with no
#: route out they fail as a browser error page rather than as anything the
#: panel has to handle. That is the right trade: the alternative is a badge
#: that names a concept and offers no way to find out what it means.
DOCS_URL = "https://gilesknap.github.io/lllm3090"

#: Where each badge on a model row goes when clicked.
#:
#: Here rather than in the page because a heading anchor is generated from the
#: heading *text*: rewording "Where guesses come from" silently breaks the link
#: and nothing in a browser would say so. Held in Python, it can be -- and is
#: -- checked against the documentation source by the test suite, so the
#: rewording fails CI instead of shipping.
#:
#: Keyed by the badge's own label, which is what the row renders, so a badge
#: with no entry is simply not a link. ``vision`` has no page of its own; the
#: field table is where ``mmproj`` is defined and is the honest destination
#: rather than an invented one.
BADGE_DOCS = {
    "dense": "explanations/dense-vs-moe",
    "moe": "explanations/dense-vs-moe",
    "vision": "reference/catalogue#fields",
    "mtp": "explanations/what-makes-it-fast#where-guesses-come-from",
    "template": "how-to/claude-code#a-patched-chat-template-only-for-qwen3-8-27b",
}
