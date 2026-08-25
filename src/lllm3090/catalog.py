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

from . import config

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
    verified: bool = False
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def weights_mib(self) -> float:
        return self.size_gb * 1000 * 1000 * 1000 / MIB


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
    capped_by: str  # "vram" | "rope"

    @property
    def summary(self) -> str:
        return (
            f"{self.per_session // 1024}k x {self.parallel} "
            f"(pool {self.pool // 1024}k, limited by {self.capped_by})"
        )


def fit(model: Model, desktop: bool = True) -> Fit:
    """Compute the context a model leaves room for on the target card.

    The KV cache is what actually decides context, and it is compressible: the
    engine is run with ``q8_0`` key and value caches, which halve the per-token
    cost and are close to lossless. ``q4_0`` would halve it again but degrades
    long-context reasoning, so it is deliberately not offered here.
    """
    budget = config.usable_vram_mib(desktop)
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


def plan(model: Model, parallel: int | None = None, desktop: bool = True) -> Plan:
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
    f = fit(model, desktop)
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
        # A multi-part GGUF is loaded by pointing at its first shard.
        first = ggufs[0]
        out.append(
            {
                "name": d.name,
                "path": str(first),
                "gb": round(sum(f.stat().st_size for f in ggufs) / 1e9, 2),
            }
        )
    return out


def catalog_for_panel(desktop: bool = True) -> list[dict[str, Any]]:
    """Catalogue entries decorated with fit and installed-state, for the UI."""
    have = {m["name"] for m in installed()}
    rows = []
    for m in load_catalog():
        f = fit(m, desktop)
        p = plan(m, desktop=desktop)
        rows.append(
            {
                "id": m.id,
                "name": m.name,
                "repo": m.repo,
                "file": m.file,
                "gb": m.size_gb,
                "kind": m.kind,
                "params": m.params,
                "verified": m.verified,
                "notes": m.notes,
                "tags": m.tags,
                "expected_tok_s": m.expected_tok_s,
                "chat_template": m.chat_template,
                "fits": f.fits,
                "max_ctx": p.per_session,
                "parallel": p.parallel,
                "pool": p.pool,
                "plan": p.summary,
                "headline": p.summary,
                "installed": m.name in have,
            }
        )
    return rows
