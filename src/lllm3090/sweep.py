"""Survey what is published, and price it against the card in this machine.

The curated list in ``data/models.yaml`` is small and hand-checked, and the
field moves in weeks. This is the tool that widens it: it asks HuggingFace what
GGUF checkpoints exist, derives each one's KV cost from its own ``config.json``,
and runs the result through the same :func:`lllm3090.catalog.fit` and
:func:`lllm3090.catalog.plan` the panel uses -- so a candidate is judged by
exactly the arithmetic that will judge it once it is in the catalogue.

Nothing here downloads weights. A survey costs one config file per candidate,
which is how a 20 GB mistake gets avoided for the price of a few kilobytes.

**It never produces a speed.** Tokens per second is a measurement on one card,
and the roofline that would derive one is calibrated between 20% and 35% of roof
for a resident MoE -- wider than the error bar that would imply. Entries come
out of here with ``verified: false`` and no ``expected_tok_s``; ``lllm3090
bench`` is what fills that in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from . import catalog, config, hardware

#: Quantisations worth considering, best first.
#:
#: The catalogue lives at roughly four bits per weight, which is what fits this
#: class of card with context left over. Unsloth's ``UD-`` dynamic quants come
#: first because two of them are already in the list and have been measured;
#: ``MXFP4`` is here for gpt-oss, which ships in nothing else.
QUANT_PREFERENCE = (
    "UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q4_K_S", "UD-Q4_K_M",
    "MXFP4", "IQ4_XS", "Q4_K_M", "Q4_K_S", "Q4_0",
)

#: Bytes per element in an f16 KV cache. The catalogue records the f16 figure
#: and halves it at plan time, because the engine is run with q8 caches.
F16 = 2


@dataclass(frozen=True)
class Candidate:
    """A published checkpoint, priced but not downloaded."""

    repo: str
    file: str
    name: str
    size_gb: float
    kv_kib_per_token: float
    max_ctx: int
    kind: str
    params: str
    downloads: int = 0
    mmproj: str | None = None
    mmproj_gb: float = 0.0

    def as_model(self) -> catalog.Model:
        """The catalogue entry this would become, for pricing.

        Deliberately built through the real :class:`lllm3090.catalog.Model`
        rather than a parallel structure: if a candidate cannot be expressed as
        a catalogue entry, it cannot be added to the catalogue, and the sweep
        should find that out here rather than at paste time.
        """
        return catalog.Model(
            id=self.name.lower().replace(".", "-").replace("_", "-"),
            name=self.name,
            repo=self.repo,
            file=self.file,
            size_gb=self.size_gb,
            kind=self.kind,
            params=self.params,
            kv_kib_per_token=self.kv_kib_per_token,
            max_ctx=self.max_ctx,
            mmproj=self.mmproj,
            mmproj_gb=self.mmproj_gb,
            verified=False,
        )


@dataclass(frozen=True)
class Priced:
    """A candidate with the verdict this card gives it."""

    candidate: Candidate
    fit: catalog.Fit
    plan: catalog.Plan
    status: str
    note: str

    @property
    def keep(self) -> bool:
        return self.status == catalog.STATUS_OK


# ---------------------------------------------------------------------------
# The arithmetic, kept pure so it can be tested without a network
# ---------------------------------------------------------------------------


def _text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """The language model's own config, which multimodal entries nest.

    A vision or audio model puts the transformer under ``text_config`` and
    keeps tower settings at the top level. Reading the top level for those
    would take the *vision* tower's head count as the language model's.
    """
    inner = cfg.get("text_config")
    return inner if isinstance(inner, dict) else cfg


def _full_attention_layers(t: dict[str, Any]) -> int:
    """How many layers hold per-token KV.

    Three architectures matter here and they are counted differently:

    * **Linear attention** (``GatedDeltaNet`` and friends) holds a fixed-size
      recurrent state, not a per-token cache. It costs nothing per token.
    * **Sliding-window attention** holds only its window, so its cost is a
      constant rather than a rate. llama.cpp provisions it that way, and the
      catalogue's measured figures agree -- ``gpt-oss-20b``'s twelve sliding
      layers contribute nothing to its 24 KiB/token.
    * **Full attention** is the only kind that grows with the conversation.

    A config with no ``layer_types`` is uniform full attention.
    """
    types = t.get("layer_types")
    if isinstance(types, list) and types:
        return sum(1 for kind in types if kind == "full_attention")
    return int(t.get("num_hidden_layers", 0))


def _head_dim(t: dict[str, Any]) -> int:
    explicit = t.get("head_dim")
    if explicit:
        return int(explicit)
    hidden, heads = t.get("hidden_size"), t.get("num_attention_heads")
    if hidden and heads:
        return int(hidden) // int(heads)
    raise Unsupported("no head_dim, and hidden_size/num_attention_heads missing")


class Unsupported(Exception):
    """This config cannot be priced, and guessing would be worse than skipping.

    Raised rather than returning a fallback figure on purpose. A wrong KV cost
    does not fail loudly: it produces a plan the card cannot honour, an engine
    that loads and reports itself healthy, and a failure at the first request.
    """


def kv_kib_per_token(cfg: dict[str, Any]) -> float:
    """KV cache cost per token at f16, in KiB, from a model's own config.

    ``full_attention_layers x 2 (K, V) x kv_heads x head_dim x 2 bytes``.

    The subtlety is which heads and which head dimension. Gemma-4 gives its
    full-attention layers a geometry of their own -- ``num_global_key_value_heads``
    and ``global_head_dim`` -- separate from the sliding layers' figures, and
    reading the sliding layers' numbers instead is wrong by 2x on the 26B and 4x
    on the 12B. Where a config draws that distinction, the global figures are
    the ones that cost per token, because the global layers are the only ones
    that do.

    Reproduces every figure in the shipped catalogue exactly; see
    ``tests/test_sweep.py``, which is what keeps that true.
    """
    t = _text_config(cfg)
    if "kv_lora_rank" in t:
        # Multi-head latent attention caches a compressed latent instead of K
        # and V, so the formula below does not describe it -- and whether
        # llama.cpp stores the latent or materialises full K/V is a property of
        # the engine build, not of the config. Not guessable from here.
        raise Unsupported("multi-head latent attention (kv_lora_rank)")

    layers = _full_attention_layers(t)
    if not layers:
        raise Unsupported("no attention layers found")

    # Gemma-4's global layers carry their own head count and width.
    heads = t.get("num_global_key_value_heads") or t.get("num_key_value_heads")
    if not heads:
        raise Unsupported("no num_key_value_heads")
    dim = int(t.get("global_head_dim") or 0) or _head_dim(t)

    return layers * 2 * int(heads) * dim * F16 / 1024


def max_ctx(cfg: dict[str, Any]) -> int:
    """The model's own RoPE ceiling, past which output degrades rather than costs."""
    t = _text_config(cfg)
    ceiling = t.get("max_position_embeddings")
    if not ceiling:
        raise Unsupported("no max_position_embeddings")
    return int(ceiling)


def kind(cfg: dict[str, Any]) -> str:
    """``moe`` or ``dense``, which is what decides whether size predicts speed."""
    t = _text_config(cfg)
    moe = any(
        t.get(k) for k in ("num_experts", "num_local_experts", "n_routed_experts")
    )
    return "moe" if moe else "dense"


def params(cfg: dict[str, Any]) -> str:
    """The one-line architecture summary a catalogue entry carries."""
    t = _text_config(cfg)
    layers = int(t.get("num_hidden_layers", 0))
    full = _full_attention_layers(t)
    bits = [f"{layers} layers"]
    if full != layers:
        bits.append(f"{full} full-attention + {layers - full} other")
    experts = t.get("num_experts") or t.get("num_local_experts")
    if experts:
        top = t.get("top_k_experts") or t.get("num_experts_per_tok")
        bits.append(f"{experts} experts" + (f", {top} per token" if top else ""))
    return ", ".join(bits)


def price(
    candidate: Candidate,
    profile: hardware.Profile | None = None,
    desktop: bool = True,
) -> Priced:
    """Run one candidate through the catalogue's own arithmetic."""
    profile = profile or hardware.detect()
    model = candidate.as_model()
    f = catalog.fit(model, desktop, profile)
    p = catalog.plan(model, desktop=desktop, profile=profile)
    state, note = catalog.status(model, p, f, profile)
    return Priced(candidate, f, p, state, note)


# ---------------------------------------------------------------------------
# Emitting a catalogue entry
# ---------------------------------------------------------------------------

#: What a swept entry cannot know, and must not invent.
#:
#: ``notes`` is prose about why someone would pick this model over its
#: neighbours, and no arithmetic produces it. Emitting a plausible-sounding
#: sentence would be worse than emitting nothing, because it would read exactly
#: like the hand-written notes around it.
TODO_NOTE = "TODO: why would someone pick this over the entry above it?"


def to_yaml(results: list[Priced], profile: hardware.Profile | None = None) -> str:
    """Catalogue entries for these candidates, ready to paste into models.yaml.

    Every field that is arithmetic is filled in. Every field that is judgement
    -- notes, tags -- is left as a marker, and the speed is left absent
    entirely. A block from here parses back into :class:`lllm3090.catalog.Model`
    unchanged, which ``tests/test_sweep.py`` asserts, so a paste that loads is
    a paste that the panel can already price.

    ``profile`` must be the one the results were priced against. The plan in
    each note is a claim about a specific card, so naming a different one --
    the detected card, when ``--gpu`` priced for another -- would attach a
    correct figure to the wrong hardware, which is worse than omitting it.
    """
    card = (profile or hardware.detect()).name
    out: list[str] = []
    for r in results:
        c = r.candidate
        out.append(f"""
  - id: {c.as_model().id}
    name: {c.name}
    repo: {c.repo}
    file: {c.file}
    size_gb: {c.size_gb}
    kind: {c.kind}
    params: {c.params}
    kv_kib_per_token: {c.kv_kib_per_token:g}
    max_ctx: {c.max_ctx}""".rstrip())
        if c.mmproj:
            out.append(f"    mmproj: {c.mmproj}\n    mmproj_gb: {c.mmproj_gb}")
        out.append(f"""    verified: false
    tags: [TODO]
    notes: >-
      {TODO_NOTE}
      Swept, not measured: {r.plan.summary} on
      {card}. Run 'lllm3090 bench' before setting
      expected_tok_s or verified.""")
    return "\n".join(out).lstrip("\n")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _pick_quant(files: dict[str, float]) -> tuple[str, float] | None:
    """The best ~4-bit single-file GGUF in a repo, by the preference order.

    Multi-part checkpoints are skipped: they are startable -- ``installed()``
    points the engine at the first shard -- but the catalogue's ``file`` field
    names one file, and a sweep should not emit an entry whose shape the
    downloader has never been given.
    """
    for want in QUANT_PREFERENCE:
        for name, size in sorted(files.items()):
            if want.lower() not in name.lower():
                continue
            if "mmproj" in name.lower() or "-of-" in name.lower():
                continue
            return name, size
    return None


def _config_for(api: Any, repo: str, info: Any) -> dict[str, Any]:
    """A repo's ``config.json``, following ``base_model`` when it has none.

    A GGUF repo is a conversion, and about half of them keep the source
    config.json alongside the weights while the rest do not. The Hub records
    what a conversion came from, so the ones that do not can still be priced.
    """
    from huggingface_hub import hf_hub_download

    names = {s.rfilename for s in (info.siblings or [])}
    if "config.json" in names:
        return json.loads(open(hf_hub_download(repo, "config.json")).read())
    for tag in info.tags or []:
        if tag.startswith("base_model:") and "/" in tag:
            base = tag.split(":")[-1]
            return json.loads(open(hf_hub_download(base, "config.json")).read())
    raise Unsupported("no config.json, and no base_model to borrow one from")


def survey(
    limit: int = 100,
    profile: hardware.Profile | None = None,
    desktop: bool = True,
    known: set[str] | None = None,
) -> tuple[list[Priced], list[Priced], list[tuple[str, str]]]:
    """Price the most-downloaded GGUF repositories against this card.

    Returns what is worth adding, what was priced and rejected, and what could
    not be priced at all with the reason. The rejections are returned rather
    than dropped because they are the part worth writing down: two of the
    models most often recommended for a 24 GB card fit it and leave a window an
    agent cannot use, and a survey that printed only its successes would lose
    that both times it ran.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    profile = profile or hardware.detect()
    known = known or {m.repo for m in catalog.load_catalog()}

    keep: list[Priced] = []
    reject: list[Priced] = []
    skipped: list[tuple[str, str]] = []

    for entry in api.list_models(filter="gguf", sort="downloads", limit=limit):
        repo = entry.id
        if repo in known:
            continue
        try:
            info = api.model_info(repo, files_metadata=True)
            files = {
                s.rfilename: (s.size or 0) / 1e9
                for s in (info.siblings or [])
                if s.rfilename.lower().endswith(".gguf")
            }
            if not files:
                raise Unsupported("no GGUF files")
            chosen = _pick_quant(files)
            if chosen is None:
                raise Unsupported("no ~4-bit single-file quant")
            cfg = _config_for(api, repo, info)
            proj = next(
                (n for n in files if "mmproj" in n.lower() and "f16" in n.lower()),
                None,
            )
            candidate = Candidate(
                repo=repo,
                file=chosen[0],
                name=repo.split("/")[-1].removesuffix("-GGUF").removesuffix("-gguf"),
                size_gb=round(chosen[1], 2),
                kv_kib_per_token=kv_kib_per_token(cfg),
                max_ctx=max_ctx(cfg),
                kind=kind(cfg),
                params=params(cfg),
                downloads=entry.downloads or 0,
                mmproj=proj,
                mmproj_gb=round(files[proj], 2) if proj else 0.0,
            )
        except Unsupported as exc:
            skipped.append((repo, str(exc)))
            continue
        except Exception as exc:  # network, rate limit, an unexpected shape
            skipped.append((repo, f"{type(exc).__name__}: {exc}"))
            continue

        priced = price(candidate, profile, desktop)
        (keep if priced.keep else reject).append(priced)

    keep.sort(key=lambda r: -r.candidate.downloads)
    reject.sort(key=lambda r: -r.candidate.downloads)
    return keep, reject, skipped


def agent_floor_note() -> str:
    """Why a model that fits can still be rejected, said once."""
    return (
        f"Rejected entries fit the card. They are rejected because one "
        f"conversation gets less than the {config.AGENT_PROMPT_FLOOR // 1000}k "
        "an agent harness spends on its system prompt before your first word."
    )
