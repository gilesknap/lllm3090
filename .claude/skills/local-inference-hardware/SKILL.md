---
name: local-inference-hardware
description: What hardware can actually hold a large local model, what it costs, and why the 2026 DRAM shortage has inverted the usual upgrade-versus-replace maths. Covers unified-memory boxes above 128 GB, the DDR4/DDR5 price crisis, and the bandwidth argument that decides between adding RAM to an existing machine and buying a dedicated one. Use when sizing or costing hardware for local LLM serving, when asked whether to upgrade RAM or buy a mini PC, or when comparing DGX Spark, Strix/Gorgon Halo, Mac Studio and x86+GPU options.
---

# Hardware for local inference

> **Prices verified 24 August 2026 and nothing here will age well.** The DRAM
> market is moving monthly and every figure below is a snapshot. Re-check before
> quoting any of it. What *is* durable is the structure: the shortage, the
> unified-memory-versus-offload argument, and the upgrade-versus-replace test.

## The fact that drives everything else

DRAM is in a severe shortage because fab capacity has been reallocated from
consumer DRAM to HBM for AI datacentres, which is far more profitable. Forecasts
put it running to **at least Q4 2027**, so memory prices are rising, not falling,
and any "wait for it to get cheaper" plan is backwards.

What it has already broken:

- **DDR4 is up roughly 5× in 15 months.** A 32 GB DDR4-3200 kit was ~$48 in May
  2025 and listed at $188 in August 2026. 256 GB DDR4 kits are over $3,000.
- **Apple withdrew its large configurations.** The 256 GB Mac Studio went in
  March 2026 and the 512 GB in May; the M3 Ultra now ships **96 GB maximum**,
  from $3,999 / £4,199. Apple is not a large-memory option at any price right
  now, whatever older guides say. An M5 Ultra refresh (~Oct 2026) was tested to
  768 GB, but supply may stop it shipping at that size.
- **The upgrade calculus inverted.** See the test below.

## What exists above 128 GB (Aug 2026)

| box | memory | stack | price (Aug 2026) |
|---|---|---|---|
| Framework Desktop, Ryzen AI Max+ Pro 495 | **192 GB unified, 160 GB as VRAM** | ROCm | previewed, unpriced, **no ship date** — Framework says "a pretty substantial jump" over $3,449, so >$4,000 |
| Minisforum MS-02 Ultra | 256 GB DDR5 SO-DIMM | **x86 + CUDA** | from ~$2,899 |
| Mac Studio M3 Ultra | 96 GB (256/512 GB withdrawn) | MLX | £4,199 |
| 2 × DGX Spark | 256 GB as two linked 128 GB pools | aarch64 CUDA | ~$4k each |
| DGX Station GB300 | 496 GB LPDDR5X + 252 GB HBM3e | CUDA | ~$90–95k |

Everything else Strix Halo is **capped at 128 GB**. DGX Spark is 128 GB with no
successor announced. Medusa Halo brings LPDDR6 (~460 GB/s) but is 2027–28.

**The 128 GB tier roughly doubled between June and August 2026** — this is the
single most important thing on this page, because it is the anchor everything
else prices against:

| box (128 GB) | launch/earlier | August 2026 |
|---|---|---|
| Framework Desktop | $1,999 (Feb 2025) | **$3,449** |
| GMKtec EVO-X2 128 GB/2 TB | $1,999 list, $2,199 in June 2026 | **$3,649** (£3,699 Amazon UK) |
| Beelink GTR9 Pro | $1,985 launch MSRP | **$4,349** |
| ONEXStation | — | $2,999 (cheapest seen) |
| AMD own-brand Ryzen AI Halo | — | $3,999 |

Note what this does to the own-brand premium: AMD's $3,999 box looked like a 2×
markup in June and is merely mid-market in August. **Any price in this file older
than about a month is fiction.** These figures moved by ~70% inside the span of a
single working session.

## Unified memory beats offload, and it is not close

An offload engine fetches experts across PCIe; a unified-memory box does not
fetch at all. That single difference is worth 4–8× on decode. Use
`moe-throughput-model` for the arithmetic; the bandwidths that go into it:

| path | real throughput |
|---|---|
| discrete GPU VRAM (RTX 3090) | ~936 GB/s |
| Mac Studio unified | ~800 GB/s |
| Strix/Gorgon Halo unified | ~215 GB/s measured (256 theoretical) |
| DDR5 dual channel @4400 (256 GB populated) | ~70 GB/s |
| **PCIe 4.0 x16 — the offload roof** | **~26 GB/s** |

Worked on gpt-oss-120b (65 GB pool, 1.79 GB expert traffic per token):

```
RTX 3090 + 128 GB DDR4, ~22% of the pool cached   → ~19 tok/s
Strix Halo 128 GB, whole pool in unified memory   → ~60-90 tok/s
```

The corollary is worth stating plainly: **an offload engine is a workaround for
not having enough unified memory.** Buy a box that holds the pool and the
engine — and all the tuning around it — stops being necessary.

## Where this project actually stands (26 August 2026)

**Sticking with the current hardware through the initial testing phase.** The
question of whether to spend anything is deferred until lllm3090 has shown it is
useful in real work -- thoth first, then other projects. An upgrade *is* on the
table after that, so the costings below are worth keeping current; they are just
not a decision yet.

What is being tested on, and what it caps at:

```
Ryzen 7 5800X, ASRock X570M Pro4, RTX 3090 24 GB
32 GB DDR4 installed, 128 GB board maximum (4 x DDR4, 32 GB per module)
```

Two consequences worth carrying into any later conversation:

- **128 GB is this platform's ceiling, and it is not enough for the top tier.**
  48 and 64 GB DIMMs are DDR5 only, so 192 GB means a new board, CPU and memory
  -- a replacement, not an upgrade. DeepSeek-V4-Flash needs 147 GB of experts,
  so even a fully-populated X570M Pro4 does not reach it.
- **The cheap intermediate step is real.** Two DIMM slots are free. Going to
  64 GB is non-destructive and roughly doubles what an expert-offload engine
  could hold -- see `freetoken-engine` for what that buys. Populating all four
  AM4 slots usually forces a memory-speed drop, so 2 x 32 GB beats 4 x 16 GB.

Judge a purchase against what the testing phase actually shows, not against what
the catalogue could theoretically run.

## The upgrade-versus-replace test

Run this before any RAM purchase, because the answer has changed:

```
cost of the RAM upgrade   vs   cost of a whole box that holds the same pool
predicted tok/s on each   (moe-throughput-model)
```

For ws03 (X570M Pro4, AM4, 4 × DDR4 slots, hard ceiling 128 GB because
unbuffered DDR4 tops out at 32 GB per module and AM4 takes no RDIMM):

| | cost | gpt-oss-120b |
|---|---|---|
| +128 GB DDR4 | ~£1,000 | ~19 tok/s |
| 128 GB unified box | **~£2,800–3,700** | ~60–90 tok/s |

This test has now flipped **twice in fifteen months**, on identical hardware:

- At 2025 DDR4 prices (~£250) the upgrade was obviously right.
- At the June 2026 box price (~£1,750) the upgrade was obviously wrong — you were
  paying two-thirds of a whole machine for a quarter of the speed.
- At August 2026 prices (~£3,700) it is roughly proportional again: 3.5× cheaper
  for ~4× less throughput, and the box's advantage is that it is a *separate*
  machine rather than that it is better value.

**Nothing about the silicon changed in either flip.** Recompute this from current
prices every time; never carry the conclusion forward.

The deeper point the shortage exposes: **every option here is priced by DRAM
except the one that avoids buying it.** An offload engine is a way to serve a
150 GB model on 32 GB of system RAM and a 24 GB card. That is a workaround for
thin hardware, and in a DRAM shortage a workaround for thin hardware is worth
more, not less.

## Sizing against real models

| model | host memory needed | fits in |
|---|---|---|
| gpt-oss-120b | 65 GB pool | 128 GB |
| MiniMax-M2.5 | ~140 GB | 192 GB |
| DeepSeek-V4-Flash | 147 GB experts + 8 GB resident | 192 GB |
| GLM-4.7 | ~230 GB | 256 GB+ |

Note DeepSeek-V4-Flash on a 192 GB Gorgon box: the entire expert pool sits
inside the 160 GB VRAM allocation, so there is no fetch and no offload engine —
~60 tok/s against ~8 tok/s for the same model offloaded from a 24 GB card.
Its KV is ~59 KiB/token, so budget context separately (`llm-checkpoint-fit`).

## Re-checking this file

Search for: current street price of the 128 GB Strix Halo boxes (the anchor for
everything), whether Gorgon Halo 192 GB has shipped and at what price, whether
Apple has restored large Mac Studio configurations, and the DDR4/DDR5 price
index. If the shortage has eased, the upgrade-versus-replace test may flip back.
