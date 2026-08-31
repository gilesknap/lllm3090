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
    #: The oldest GPU generation that can run this entry at all, as an
    #: ``nvidia-smi`` compute capability -- ``"12.0"`` for a model whose
    #: quantisation only has Blackwell kernels, say.
    #:
    #: Deliberately the *only* hardware requirement that is typed by hand.
    #: Everything about memory is arithmetic (:func:`min_vram_mib`) and travels
    #: to any card on its own; a kernel that does not exist for an older
    #: architecture cannot be derived from a file size, so it has to be
    #: recorded. Left unset means "no known floor", which is the common case.
    min_compute_capability: str | None = None
    verified: bool = False
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def weights_mib(self) -> float:
        return (self.size_gb + self.mmproj_gb) * 1000 * 1000 * 1000 / MIB

    @property
    def vision(self) -> bool:
        return self.mmproj is not None

    @property
    def min_vram_gb(self) -> float | None:
        """Smallest card that runs this model at a window an agent can use.

        Computed with a desktop session held back, because that is the machine
        most people are actually sitting at, and over-promising is the failure
        this arithmetic exists to prevent. ``None`` means no card is enough --
        see :func:`min_vram_mib`.
        """
        need = min_vram_mib(self)
        return None if need is None else round(need / 1024, 1)


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
    def agent_ready(self) -> bool:
        """Whether one conversation clears the harness's own system prompt.

        Fitting and being usable are different questions, and the gap between
        them is wide: a model can load with room to spare and still leave less
        context than Claude Code spends before your first word. See
        ``config.AGENT_PROMPT_FLOOR``.
        """
        return self.per_session > config.AGENT_PROMPT_FLOOR

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
        # The nominal per-token figure is the tensor arithmetic; what the engine
        # actually holds resident is larger, so it is what gets divided into the
        # budget. See config.KV_OVERHEAD_FACTOR.
        tokens = spare * 1024 / (kib_per_token * config.KV_OVERHEAD_FACTOR)
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
    f = fit(model, desktop, profile)
    if not f.fits:
        return Plan(0, requested or config.DEFAULT_PARALLEL, 0, "vram")

    if requested is not None:
        # An explicit --parallel is an instruction, not a hint: divide the pool
        # as asked, even where that leaves each slot short of the ceiling.
        #
        # Except where the pool cannot seat the request at all. Rounding a
        # slot's share up to one page to keep it non-zero is what makes that
        # dangerous: ask for more slots than there are pages and every slot
        # gets a page anyway, so the plan's pool exceeds the cache that exists
        # and the engine starts, allocates, and dies of VRAM exhaustion. A
        # refusal here is the same refusal, before anything is launched.
        if f.pool_q8 // requested < 1024:
            raise ValueError(
                f"{requested} slots do not fit: {model.name} leaves "
                f"{f.pool_q8} tokens of cache on this card, which is under "
                f"one 1024-token page each. The most it can seat is "
                f"{max(1, f.pool_q8 // 1024)}."
            )
        share = f.pool_q8 // requested
        if share >= model.max_ctx:
            return Plan(
                model.max_ctx * requested, requested, model.max_ctx, "rope"
            )
        share = (share // 1024) * 1024
        return Plan(share * requested, requested, share, "vram")

    # Automatically: fill one conversation to the model's ceiling, and split
    # only when refusing to would strand too much of the card.
    #
    # The pool is a fixed number of tokens, so splitting it does not create
    # capacity -- it shortens each conversation and buys concurrency with the
    # difference. One long conversation is therefore the default. But the
    # window is bounded by RoPE as well as by VRAM, and past that ceiling the
    # remaining cache can never become a longer conversation: a pool holding
    # 2.8 windows would give one full window and strand 1.8 windows of cache.
    #
    # So the test is on *total* usable context. If splitting raises it by
    # ``SLOT_SPLIT_GAIN`` or more, split; otherwise keep the single window and
    # accept the remainder as unusable. Both halves of that matter -- the first
    # stops a large surplus going to waste, the second stops a small one from
    # halving a conversation to reclaim it.
    def window(slots: int) -> int:
        """Per-conversation window at this slot count, in whole pages."""
        return max(1024, (min(model.max_ctx, f.pool_q8 // slots) // 1024) * 1024)

    alone = window(1)
    slots = 1
    if f.pool_q8 >= config.SLOT_SPLIT_GAIN * alone:
        # Worth splitting. How far is a second question, and answering it with
        # "as far as consumes the whole pool" is wrong: it produces a cliff at
        # the threshold where a *larger* pool yields a *shorter* conversation.
        # Qwen3.6-35B-A3B was the case -- 256k on a desktop and 184k headless,
        # so freeing the compositor's VRAM made the window worse. See
        # ``test_headless_never_offers_less_than_a_desktop``.
        #
        # So take each further slot only while it recovers more cache than it
        # costs window, proportionally. Muse-Glimmer's third slot recovers 28%
        # of a stranded pool for 8% of its window and is taken; the A3B's third
        # recovers 7% for 28% and is not.
        slots = 2
        while slots < config.MAX_AUTO_PARALLEL:
            here, further = window(slots), window(slots + 1)
            if further <= 1024:
                break
            gained = ((slots + 1) * further) / (slots * here)
            lost = here / further
            if gained <= lost:
                break
            slots += 1
    share = window(slots)
    capped = "rope" if share >= model.max_ctx else "vram"
    return Plan(share * slots, slots, share, capped)


def _pages_up(tokens: int) -> int:
    """``tokens`` rounded up to whole 1024-token cache pages."""
    return int(math.ceil(tokens / 1024) * 1024)


def min_vram_mib(
    model: Model, desktop: bool = True, parallel: int | None = None
) -> float | None:
    """The smallest card that leaves this model a window an agent can use.

    The inverse of :func:`fit`. Where ``fit`` asks what context a given card
    leaves, this asks what card the agent floor demands -- the question someone
    shopping for hardware, or reading this catalogue on a card it was not
    curated for, is actually asking.

    It is derived rather than declared, and that is deliberate. A hand-typed
    figure in ``models.yaml`` would be a second source of truth beside this
    arithmetic, free to disagree with it and silently invalidated by every
    change to a reserve -- ``VISION_WORKSPACE_RESERVE_MIB`` was introduced
    after the catalogue shipped and moved the answer for every vision entry.

    ``None`` means no amount of memory is enough: the model's RoPE ceiling is
    already at or below the floor, so its window is bounded by the architecture
    and a bigger card cannot move it. ``Qwen3-8B`` is that case, at 32k.
    """
    if model.max_ctx <= config.AGENT_PROMPT_FLOOR:
        return None
    # One slot, because that is what plan() hands out automatically at the
    # margin. At the smallest card that works, the pool is smaller than the
    # model's ceiling, so the whole-window rule grants exactly one conversation
    # and every byte goes to it. Computing this against two slots would name a
    # card half again too large -- and it did, until plan() changed under it.
    parallel = parallel or 1
    # The smallest window that clears the floor, page-aligned the way plan()
    # hands one out, and never more than the architecture can address.
    window = min(model.max_ctx, _pages_up(config.AGENT_PROMPT_FLOOR + 1))
    # Inverting fit(): a pool of this many tokens costs this much cache at q8,
    # with the engine's own per-cell overhead restored on top.
    per_token_kib = (model.kv_kib_per_token / 2) * config.KV_OVERHEAD_FACTOR
    need = model.weights_mib + window * parallel * per_token_kib / 1024
    need += config.DRIVER_RESERVE_MIB + config.WORKSPACE_RESERVE_MIB
    if model.vision:
        need += config.VISION_WORKSPACE_RESERVE_MIB
    if desktop:
        need += config.DESKTOP_RESERVE_MIB
    return float(math.ceil(need))


def _capability(text: str) -> tuple[int, ...] | None:
    """A compute capability as comparable numbers, or None if it is not one."""
    try:
        return tuple(int(part) for part in text.split("."))
    except ValueError:
        return None


def capability_ok(model: Model, profile: hardware.Profile) -> bool:
    """Whether this card's architecture is new enough to run this model at all.

    Unknown in either direction is no objection. A missing
    ``min_compute_capability`` is the common case and means nothing is known to
    be required; a card that will not report its own capability cannot be
    *proven* inadequate, and hiding half the catalogue on that basis is a worse
    answer than letting a download fail.
    """
    if model.min_compute_capability is None:
        return True
    want = _capability(model.min_compute_capability)
    have = _capability(profile.compute_capability)
    if want is None or have is None:
        return True
    return have >= want


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
        # Zero would be swallowed by the fallback below and negative divides
        # the pool the wrong way, producing a Plan the engine would start with.
        raise ValueError(f"parallel must be at least 1, not {parallel}")
    known = next((m for m in load_catalog() if m.name == name), None)
    if known is not None:
        # Passed through as-is, including None: substituting a default here
        # would reach plan() as an explicit request and silently disable the
        # automatic whole-window rule.
        # Whether a desktop is holding VRAM is a property of the machine, not
        # of the front end asking, so it is resolved here rather than at each
        # call site -- otherwise the console and the panel could size the same
        # model differently on the same card.
        return plan(known, parallel, desktop=hardware.graphical())
    # An uncatalogued GGUF has no known ceiling, so there is no window to fill
    # and the whole-window rule has nothing to reason about. It keeps the old
    # behaviour: a conservative per-slot figure, and enough slots for an agent
    # and one subagent, which is the safe guess when nothing else is known.
    slots = parallel or config.DEFAULT_PARALLEL
    return Plan(UNKNOWN_MODEL_CTX * slots, slots, UNKNOWN_MODEL_CTX, "default")


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

    The q8 cache halves the per-token figure, which is stored at f16, and
    ``config.KV_OVERHEAD_FACTOR`` restores what the engine holds on top of it.
    """
    per_token = (model.kv_kib_per_token / 2) * config.KV_OVERHEAD_FACTOR
    return model.weights_mib + ctx * per_token / 1024


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


#: What a catalogue entry is, on the card in this machine.
#:
#: Three states rather than two, because "fits" and "usable" are not the same
#: claim. ``Ornith-1.5-35B-A3B`` loads on a 24 GB card and leaves 5k of
#: context; ``DeepSeek-R1-Distill-Qwen-32B`` loads and leaves 8k. Both are
#: widely recommended for this card, and neither can run an agent harness whose
#: system prompt alone is 40k. A UI with one "fits" flag has to show them as
#: successes.
STATUS_OK = "ok"
STATUS_TIGHT = "tight"
STATUS_TOO_BIG = "too-big"
STATUS_CAPABILITY = "capability"


def _advise_gb(need: float) -> int:
    """A required card size, rounded the only safe way: up.

    ``min_vram_gb`` is a threshold, not an estimate. Rounding 24.4 to nearest
    advises a 24 GB card for a model that needs more than 24 GB, which is the
    one direction this arithmetic must never err in -- it is advice to go and
    buy the wrong hardware.
    """
    return math.ceil(need)


def status(
    model: Model, plan_: Plan, fit_: Fit, profile: hardware.Profile
) -> tuple[str, str]:
    """This model's state on this card, and the phrase every front end shows.

    The wording lives here rather than in the panel, the terminal UI and the
    CLI separately, because all three answer the same question about the same
    card and disagreeing about it is how a promise gets made in one place and
    withdrawn in another.
    """
    if not capability_ok(model, profile):
        return STATUS_CAPABILITY, (
            f"needs compute capability {model.min_compute_capability}; "
            f"this card is {profile.compute_capability}"
        )
    if not fit_.fits:
        need = model.min_vram_gb
        room = f"needs about {_advise_gb(need)} GB" if need else "does not fit"
        return STATUS_TOO_BIG, f"too big for this card -- {room}"
    if not plan_.agent_ready:
        short = (
            f"fits, but leaves {plan_.per_session // 1024}k per conversation -- "
            f"under the {config.AGENT_PROMPT_FLOOR // 1000}k an agent harness "
            "spends before your first word"
        )
        need = model.min_vram_gb
        if need is None:
            # Bounded by RoPE, not by memory. Saying "needs a bigger card"
            # here would send someone shopping for one that does not exist.
            return STATUS_TIGHT, (
                f"{short}. Its {model.max_ctx // 1024}k ceiling is the "
                "architecture's, so no card lifts it"
            )
        return STATUS_TIGHT, f"{short}. About {_advise_gb(need)} GB would clear it"
    return STATUS_OK, ""


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
        state, why = status(m, p, f, profile)
        rows.append(
            {
                "status": state,
                "status_note": why,
                "agent_ready": p.agent_ready,
                "capability_ok": capability_ok(m, profile),
                "min_compute_capability": m.min_compute_capability,
                # What card this would need, independent of the one detected --
                # so a reader on a 16 GB card learns what would change it.
                "min_vram_gb": m.min_vram_gb,
                "card_gb": round(profile.vram_mib / 1024),
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
