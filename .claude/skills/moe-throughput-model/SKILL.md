---
name: moe-throughput-model
description: Predict decode tokens/sec before downloading a model or buying hardware, and know which of two regimes you are in — experts crossing PCIe from host RAM, or a checkpoint resident in VRAM, where the same formula under-predicts by 2-5x — expert bytes fetched per token, what the cache-hit fraction is really worth, and which link (VRAM, PCIe, system RAM) is the actual roof. Use when asked how fast a model will run on a given box, when sizing RAM or VRAM or costing a new machine for local inference, when a model fits but you need to know whether it is usable, or when measured throughput is far from what you predicted.
---

# What will it actually run at?

"It fits" and "it is usable" are different questions. A model can load and still
decode at 8 tok/s, and you can find that out from the config file rather than
from a 150 GB download or a £10k purchase.

Decode is memory-bound, so the whole estimate is **bytes moved per token divided
by the bandwidth of the slowest link they cross**. Everything below is that one
idea — but *which* link is the slowest is the whole question, and getting it
wrong is worth a factor of five.

## First decide which case you are in

```
Does the whole checkpoint fit in VRAM?
├── no, experts live in host RAM   → the PCIe formula below. Roof ~26 GB/s.
└── yes, everything is resident    → the roof is VRAM bandwidth, not PCIe.
                                     dense:        ~60% of roofline
                                     resident MoE: ~33% of roofline
```

**Applying the offload formula to a resident model under-predicts by 2-5x.**
That is not a hypothetical: it happened here, to four models out of five, and
hid behind a "±30%" caveat for weeks. The measured evidence is in
[Correction](#correction-this-model-is-for-offloaded-moe-only) below; read it
before using anything above it.

## The arithmetic (offloaded MoE only)

```
expert_bytes  = 3 × moe_intermediate_size × hidden_size × bytes_per_weight   (gate, up, down)
              + block-scale bytes if the quant carries them
fetched/token = num_experts_per_tok × num_layers × expert_bytes × (1 − cache_hit)
tok/s         ≈ link_bandwidth / fetched_per_token          → take 60–80% of it
```

`bytes_per_weight` is 0.5 for NVFP4/MXFP4, 1 for FP8, 2 for bf16. Add the
resident term — attention and embeddings read from VRAM at VRAM bandwidth — but
on an offload box it is almost always dwarfed by the expert fetch.

**Which link is the roof.** Experts live in system RAM and cross PCIe to the
GPU, so the roof is `min(RAM bandwidth, PCIe bandwidth)` — and on any consumer
box that is PCIe by a wide margin:

| link | real throughput |
|---|---|
| RTX 3090 VRAM | ~936 GB/s |
| PCIe 4.0 x16 | ~26 GB/s |
| DDR4-3200 dual channel | ~40 GB/s |
| LPDDR5x unified (DGX Spark, Strix Halo) | ~256–273 GB/s |
| M3 Ultra unified | ~800 GB/s |

That 36:1 gap between VRAM and PCIe is why the cache-hit fraction dominates
everything else, and why unified-memory boxes change the shape of the problem
rather than just the size of it.

## Calibrate before you trust it

Run the model against a machine you have measured. On ws03, Qwen3.6-35B-A3B-NVFP4
(top-8, 40 layers, moe_intermediate 512, hidden 2048, NVFP4):

```
expert_bytes  = 3 × 512 × 2048 × 0.5           = 1.57 MB
fetched/token = 8 × 40 × 1.57 MB × (1 − 0.63)  = 186 MB
tok/s         = 26 GB/s ÷ 186 MB               = 140      (measured: 148)
```

Treat that as **±20%, not ±5%**: counting the fp8 block scales as well drops the
prediction to ~124. It is a decision tool — "is this 10 tok/s or 100 tok/s" —
not a benchmark.

A second calibration on the same box, Gemma-4-26B-A4B-NVFP4 (top-8, 30 layers,
moe_intermediate 704, hidden 2816, 1,529 of 3,840 experts resident):

```
expert_bytes  = 3 × 704 × 2816 × 0.5 + scales   = 3.35 MB
fetched/token = 8 × 30 × 3.35 MB × (1 − 0.40)   = 482 MB
tok/s         = 26 GB/s ÷ 482 MB                = 54       (measured: 70)
```

Note which way the error goes — both predictions came in **under** the measured
figure. The resident *fraction* is a lower bound on the hit rate: routing is
skewed, popular experts stay cached, so the real hit rate beats the fraction.
Predictions built on the resident fraction are pessimistic, which is the safe
direction for a purchase decision.

The two models side by side are the whole argument for computing this before
downloading: 8 × 40 × 1.57 MB against 8 × 30 × 3.35 MB is **2.6× the expert
traffic per token** for the model with fewer parameters, and it showed up as
148 vs 70 tok/s measured. Nothing in either model card would have told you.

## Correction: this model is for OFFLOADED MoE only

Measured on a 3090 with five models served entirely from VRAM by llama.cpp,
the predictions above were wrong -- not marginally, and always in the same
direction:

| model | predicted | measured | error |
|---|---|---|---|
| Qwen3.6-35B-A3B | ~90 | **126** | −29% |
| Qwen3.6-35B-A3B-Q4KS | ~90 | **124** | −27% |
| gpt-oss-20b | ~80 | **160** | −50% |
| Qwen3-8B (dense) | ~60 | **115** | −48% |
| Qwen3.8-27B (dense) | ~35 | **35** | 0% |

The one that was right is the one whose arithmetic actually applied.

**The formula divides by PCIe bandwidth. That is only the roof when experts
cross PCIe.** When the whole checkpoint is resident in VRAM nothing is fetched,
the roof is VRAM bandwidth, and predicting from 26 GB/s under-calls by 2-5x.
Applying an offload model to a resident model is a category error, and it hid
behind "±30%" for three of these.

What the resident case actually looks like on this card:

```
dense:        bytes/token = weights − embeddings, read at 936 GB/s
              Qwen3.8-27B: 16 GB → 58 roofline → 35 measured (60% of roof)

resident MoE: bytes/token = active experts + attention, also at 936 GB/s
              gpt-oss-20b: ~1.9 GB → 492 roofline → 160 measured (33% of roof)
```

So dense reaches ~60% of its roofline and resident MoE only ~33%: the expert
GEMMs are small and the kernel spends proportionally more time not moving
weights. **Use 60% of roofline for dense and 33% for resident MoE**, and reserve
the PCIe formula for models whose experts genuinely live in host RAM.

## Worked example (offloaded): a model that fits and is still unusable

DeepSeek-V4-Flash — top-6, 43 layers, moe_intermediate 2048, hidden 4096, FP4
experts, a 147 GB expert pool:

```
expert_bytes  = 3 × 2048 × 4096 × 0.5 + scales   = 13.4 MB
fetched/token = 6 × 43 × 13.4 MB                 = 3.46 GB   uncached
24 GB card caches ~7% of a 147 GB pool           → 3.2 GB crosses PCIe
tok/s         = 26 GB/s ÷ 3.2 GB                 ≈ 8
```

Even at a generous 30% hit rate it is 10.8 tok/s. The conclusion — *more system
RAM will not make this usable* — costs one config file to reach.

## What this tells you to buy

Rearranged, the formula says something blunt:

- **System RAM buys the pool** — whether the model loads at all.
- **VRAM buys the hit rate** — whether it runs at a useful speed.
- **PCIe is fixed**, so on a single consumer card the only lever you have is
  cache hit, i.e. VRAM.

So for any candidate model, compute the VRAM needed for a *target* hit rate
before pricing anything:

```
VRAM ≈ resident_weights + KV_at_target_context + (target_hit × expert_pool)
```

Wanting 50% of a 147 GB pool resident means ~75 GB of cache — that is the number
that decides between "a bigger RAM kit" and "a different class of machine", and
it is not a number anyone quotes in a model card.

## Where the model does not apply

- **Dense models.** No expert fetch, nothing offloads: bytes/token is the whole
  resident weight set minus embeddings, read from VRAM at VRAM bandwidth. Use
  `llm-checkpoint-fit` — for dense models, fitting *is* the question.
- **Prefill.** Compute-bound, not bandwidth-bound, and dominated by attention
  geometry: sliding-window layers cost nothing at depth while full-attention
  layers cost quadratically. Prefill and decode scale with different things —
  never quote one figure for both.
- **SSD streaming.** Sometimes proposed as a way around a small expert pool.
  A Gen4 NVMe is ~7 GB/s, so the DeepSeek example above becomes 0.5 s per token.
  FreeToken does not do it, and the arithmetic says not to want it.
