"""The curated model list, and the arithmetic that decides what fits.

Two sources feed the panel's model list: ``data/models.yaml`` (things you could
download) and whatever GGUF files are already on disk (things you can run). A
catalogue entry carries the numbers needed to answer "will this fit, and how
much context do I get" *before* anything is downloaded -- see
:func:`fit` for the arithmetic and the reasoning behind it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from . import config, hardware

#: Bytes per KiB/MiB, spelled out so the arithmetic below reads unambiguously.
MIB = 1024 * 1024


@dataclass(frozen=True)
class Model:
    """One entry in the curated catalogue."""

    id: str
    name: str
    repo: str
    file: str
    size_gb: float
    kind: str  # "dense" | "moe"
    params: str
    #: KV cache cost per token at f16, in KiB. Derived from the architecture:
    #: ``full_attention_layers x 2 (K,V) x kv_heads x head_dim x 2 bytes``.
    #: Linear-attention layers contribute nothing per token; sliding-window
    #: layers are counted at the engine's provisioning ratio, not the window.
    kv_kib_per_token: float
    #: The model's own RoPE ceiling. Context beyond this is incoherent, not
    #: merely expensive, so it caps every calculation here.
    max_ctx: int
    expected_tok_s: int | None = None
    #: Optional replacement chat template shipped in ``lllm3090.data``. Used
    #: where a model's own template rejects something a client legitimately
    #: sends; see ``docs/explanations`` and the file's own comment for why.
    chat_template: str | None = None
    #: Multimodal projector shipped alongside the weights in the same repo.
    #: Present means the model can see images: the engine is given --mmproj,
    #: and the projector is downloaded with the weights.
    mmproj: str | None = None
    #: The projector's size. It occupies VRAM like any other weights, so it is
    #: counted against the budget -- otherwise the panel promises context that
    #: the projector has already spent.
    mmproj_gb: float = 0.0
    verified: bool = False
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def weights_mib(self) -> float:
        return (self.size_gb + self.mmproj_gb) * 1000 * 1000 * 1000 / MIB

    @property
    def vision(self) -> bool:
        return self.mmproj is not None


@dataclass(frozen=True)
class Fit:
    """Whether a model fits, and how many tokens of cache it leaves room for.

    ``pool_*`` are total tokens across all concurrent conversations, bounded by
    VRAM alone. A single conversation is additionally bounded by the model's
    RoPE ceiling -- see :func:`plan`.
    """

    fits: bool
    pool_f16: int
    pool_q8: int
    spare_mib: int

    @property
    def headline(self) -> str:
        if not self.fits:
            return "does not fit"
        return f"{self.pool_q8 // 1024}k tokens of cache"


@dataclass(frozen=True)
class Plan:
    """How to actually start a model: pool size, slots, and what each gets."""

    pool: int
    parallel: int
    per_session: int
    #: What stopped the window growing -- the card, the architecture, or
    #: nothing at all. ``default`` is a GGUF the catalogue has never seen,
    #: whose KV cost is unknown, so its window is a fixed conservative figure.
    capped_by: str  # "vram" | "rope" | "default"

    @property
    def summary(self) -> str:
        return (
            f"{self.per_session // 1024}k x {self.parallel} "
            f"(pool {self.pool // 1024}k, limited by {self.capped_by})"
        )


def fit(
    model: Model, desktop: bool = True, profile: hardware.Profile | None = None
) -> Fit:
    """Compute the context a model leaves room for on the target card.

    The KV cache is what actually decides context, and it is compressible: the
    engine is run with ``q8_0`` key and value caches, which halve the per-token
    cost and are close to lossless. ``q4_0`` would halve it again but degrades
    long-context reasoning, so it is deliberately not offered here.
    """
    profile = profile or hardware.detect()
    budget = profile.usable_vram_mib(desktop)
    if model.vision:
        # The vision tower's compute buffers are not the projector's file size.
        budget -= config.VISION_WORKSPACE_RESERVE_MIB
    spare = budget - model.weights_mib
    if spare <= 0:
        return Fit(False, 0, 0, int(spare))

    def ctx_for(kib_per_token: float) -> int:
        # Total tokens VRAM can hold, rounded down to whole 1024-token pages.
        # Deliberately NOT clamped to the RoPE ceiling: that bounds one
        # conversation, while this is the pool they all share. plan() applies it.
        tokens = spare * 1024 / kib_per_token
        return int(math.floor(tokens / 1024) * 1024)

    return Fit(
        fits=True,
        pool_f16=ctx_for(model.kv_kib_per_token),
        pool_q8=ctx_for(model.kv_kib_per_token / 2),
        spare_mib=int(spare),
    )


def plan(
    model: Model,
    parallel: int | None = None,
    desktop: bool = True,
    profile: hardware.Profile | None = None,
) -> Plan:
    """Decide the pool size and per-conversation window for a model.

    Two limits apply and they are not the same limit:

    * **VRAM** bounds the whole pool, shared across concurrent conversations.
    * **RoPE** bounds one conversation. Past the model's ceiling, output becomes
      incoherent rather than merely expensive, so extra pool beyond
      ``parallel x max_ctx`` buys nothing and is not requested.

    Spare capacity therefore goes to concurrency rather than to a window the
    model cannot use -- which is what leaves room for an agent's subagents.
    """
    requested = parallel
    parallel = parallel or config.DEFAULT_PARALLEL
    f = fit(model, desktop, profile)
    if not f.fits:
        return Plan(0, parallel, 0, "vram")

    share = f.pool_q8 // parallel
    if share >= model.max_ctx:
        # The architecture runs out before the card does. Spare cache cannot be
        # turned into a longer conversation, so turn it into more of them: hand
        # out every slot that still gets the full window, up to the automatic
        # ceiling. An explicit --parallel is honoured as given.
        if requested is None:
            affordable = f.pool_q8 // model.max_ctx
            parallel = max(parallel, min(affordable, config.MAX_AUTO_PARALLEL))
        return Plan(model.max_ctx * parallel, parallel, model.max_ctx, "rope")
    # Round the per-session window down to whole pages so the pool divides evenly.
    share = max(1024, (share // 1024) * 1024)
    return Plan(share * parallel, parallel, share, "vram")


#: Per-slot window for a GGUF that is not in the catalogue.
#:
#: Nothing is known about its KV cost per token, so there is no arithmetic to
#: do. Guessing high produces an engine that loads and then fails every request
#: out of device memory, which is the expensive mistake; this is the cheap one.
UNKNOWN_MODEL_CTX = 32768


def launch_plan(name: str, parallel: int | None = None) -> Plan:
    """How to start an installed model, whether or not the catalogue knows it.

    Every front end -- the CLI, the panel and the terminal UI -- has to answer
    the same question before it can launch anything, and they must answer it
    identically: a model started from the console and the same model started
    from the panel are the same engine on the same card.
    """
    if parallel is not None and parallel < 1:
        # Zero would be swallowed by the default below and negative divides the
        # pool the wrong way, producing a Plan the engine would be started with.
        raise ValueError(f"parallel must be at least 1, not {parallel}")
    parallel = parallel or config.DEFAULT_PARALLEL
    known = next((m for m in load_catalog() if m.name == name), None)
    if known is not None:
        # Whether a desktop is holding VRAM is a property of the machine, not
        # of the front end asking, so it is resolved here rather than at each
        # call site -- otherwise the console and the panel could size the same
        # model differently on the same card.
        return plan(known, parallel, desktop=hardware.graphical())
    pool = UNKNOWN_MODEL_CTX * parallel
    return Plan(pool, parallel, UNKNOWN_MODEL_CTX, "default")


def load_catalog() -> list[Model]:
    """The curated list shipped with the package."""
    raw = resources.files("lllm3090.data").joinpath("models.yaml").read_text()
    return [Model(**entry) for entry in yaml.safe_load(raw)["models"]]


def installed(models_dir: Path | None = None) -> list[dict[str, Any]]:
    """GGUF checkpoints present on disk, one entry per directory."""
    root = models_dir or config.MODELS_DIR
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        ggufs = sorted(d.glob("*.gguf"))
        if not ggufs:
            continue
        # A projector is a GGUF but not a checkpoint: handing it to --model
        # starts an engine that loads and then answers nothing useful.
        proj = [f for f in ggufs if "mmproj" in f.name.lower()]
        weights = [f for f in ggufs if f not in proj]
        if not weights:
            continue
        # A multi-part GGUF is loaded by pointing at its first shard.
        first = weights[0]
        out.append(
            {
                "name": d.name,
                "path": str(first),
                "mmproj": str(proj[0]) if proj else None,
                "gb": round(sum(f.stat().st_size for f in ggufs) / 1e9, 2),
            }
        )
    return out


def vram_needed_mib(model: Model, ctx: int) -> float:
    """Roughly what a pool of ``ctx`` tokens costs, weights and projector included.

    The q8 cache halves the per-token figure, which is stored at f16.
    """
    return model.weights_mib + ctx * (model.kv_kib_per_token / 2) / 1024


def startup_vram_mib(model: Model, ctx: int) -> float:
    """What the card must have free for this plan to load *and keep serving*.

    ``vram_needed_mib`` is what the load itself allocates. The engine then needs
    room to work in on top of that -- the compute buffers and fragmentation
    headroom ``fit()`` holds back out of the budget, and the vision tower's own
    buffers when a projector is loaded. Both are already subtracted when the
    plan is computed, so a check that compares only the load against free VRAM
    passes plans that load and then fail every request, which is the exact
    failure the check exists to catch.

    The desktop reserve is deliberately *not* added: free VRAM is a measurement,
    and a compositor that is running has already taken its share out of it.
    """
    needed = vram_needed_mib(model, ctx)
    needed += config.WORKSPACE_RESERVE_MIB
    if model.vision:
        needed += config.VISION_WORKSPACE_RESERVE_MIB
    return needed


def free_vram_warning(model: Model | None, ctx: int) -> str | None:
    """Warn if the card cannot hold this plan *now*, or ``None`` if it can.

    The plan is computed against fixed reserves, which describe a machine at
    rest. This is the measurement, and it is what catches the estimate being
    wrong: a model sized on a text console and started under a desktop, or
    started beside anything else holding VRAM, loads and reports itself healthy
    before failing every request out of device memory.

    Both front ends have to make this check -- an engine started from the panel
    is the same engine on the same card as one started from the console -- so
    the comparison and its wording live here rather than at each call site.
    Call it after ``engine.stop()``, or the outgoing engine's VRAM is counted
    as used against its own replacement.
    """
    if model is None:
        # Nothing is known about an uncatalogued GGUF's cache cost, so there is
        # no requirement to compare it against.
        return None
    free = hardware.free_vram_mib()
    if free is None:
        return None
    needed = startup_vram_mib(model, ctx)
    if needed <= free:
        return None
    return (
        f"Warning: this plan needs about {needed / 1024:.1f} GB and only "
        f"{free / 1024:.1f} GB is free. The engine may load and then fail every "
        "request. Close what is using the GPU, run headless, or ask for a "
        "smaller context."
    )


def catalog_for_panel(desktop: bool | None = None) -> list[dict[str, Any]]:
    """Catalogue entries decorated with fit, plan and installed-state, for the UI.

    ``speed_applies`` is false when the GPU in this machine is not the one the
    speeds were measured on. Fit and context are computed for the real card;
    speeds are never scaled to it, because a bandwidth ratio produces a guess
    and the UI would show it in the same typeface as a measurement.
    """
    have = {m["name"] for m in installed()}
    profile = hardware.detect()
    if desktop is None:
        desktop = hardware.graphical()
    rows = []
    for m in load_catalog():
        f = fit(m, desktop, profile)
        p = plan(m, desktop=desktop, profile=profile)
        rows.append(
            {
                "id": m.id,
                "name": m.name,
                "repo": m.repo,
                "file": m.file,
                # What you download and what occupies VRAM, projector included.
                "gb": round(m.size_gb + m.mmproj_gb, 2),
                "mmproj_gb": m.mmproj_gb,
                "kind": m.kind,
                "params": m.params,
                "verified": m.verified,
                "speed_applies": profile.measured,
                "measured_on": hardware.reference().name,
                "notes": m.notes,
                "tags": m.tags,
                "expected_tok_s": m.expected_tok_s,
                "chat_template": m.chat_template,
                "mmproj": m.mmproj,
                "vision": m.vision,
                "fits": f.fits,
                "desktop": desktop,
                "max_ctx": p.per_session,
                "parallel": p.parallel,
                "pool": p.pool,
                "plan": p.summary,
                "headline": p.summary,
                "installed": m.name in have,
            }
        )
    return rows
