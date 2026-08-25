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
    default_ctx: int
    expected_tok_s: int | None = None
    verified: bool = False
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def weights_mib(self) -> float:
        return self.size_gb * 1000 * 1000 * 1000 / MIB


@dataclass(frozen=True)
class Fit:
    """Whether a model fits, and what context it leaves room for."""

    fits: bool
    max_ctx_f16: int
    max_ctx_q8: int
    spare_mib: int

    @property
    def headline(self) -> str:
        if not self.fits:
            return "does not fit"
        return f"up to {self.max_ctx_q8 // 1024}k context"


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
        tokens = spare * 1024 / kib_per_token
        # Round down to a whole number of 1024-token pages, then clamp to the
        # architecture's ceiling.
        return int(min(math.floor(tokens / 1024) * 1024, model.max_ctx))

    return Fit(
        fits=True,
        max_ctx_f16=ctx_for(model.kv_kib_per_token),
        max_ctx_q8=ctx_for(model.kv_kib_per_token / 2),
        spare_mib=int(spare),
    )


def load_catalog() -> list[Model]:
    """The curated list shipped with the package."""
    raw = resources.files("llm3090.data").joinpath("models.yaml").read_text()
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
                "default_ctx": m.default_ctx,
                "fits": f.fits,
                "max_ctx": f.max_ctx_q8,
                "headline": f.headline,
                "installed": m.name in have,
            }
        )
    return rows
