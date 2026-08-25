# Dense and sparse models want different hardware

Every model in the catalogue is one or the other, and the distinction predicts
both speed and which machine you should buy next.

## The mechanism

Decoding is memory-bound. Each token requires reading the active weights out of
memory, so:

```
tokens/second ≈ memory bandwidth ÷ bytes read per token
```

- **Dense**: every parameter is active. A 27B model at 4 bits reads ~16 GB per
  token. On a 3090's 936 GB/s that is a ceiling of about 58 tok/s, and ~35 in
  practice.
- **Sparse (Mixture of Experts)**: a router picks a handful of experts per
  token. Qwen3.6-35B-A3B has 35B parameters but activates 3B, so it reads a
  fraction of its size per token and decodes far faster than a dense model of
  the same footprint.

This is why a 35B model can be quicker than a 27B one. Parameter count is not a
speed indicator; *active* bytes per token is.

## Which to pick

**Sparse** for agentic work and long sessions. It decodes faster, and the
models here happen to pair sparsity with hybrid attention that also makes the
KV cache cheap — Qwen3.6-35B-A3B reaches the full 262k window where the dense
27B stops around 200k.

**Dense** when you want the most capable model per gigabyte. A dense 27B is
stronger per parameter than a sparse 35B with 3B active; you pay for that in
tokens per second.

## The hardware corollary

The two want opposite machines, which matters when someone recommends a box:

- **Dense is bandwidth-bound.** All weights cross the bus every token. A fast,
  narrow GPU beats a slow, wide one. Capacity barely matters once it fits.
- **Sparse is capacity-bound at scale.** Very large MoE models have expert pools
  of 60–150 GB. What matters is holding them close to the compute — which is
  what unified-memory machines are for, and why they lose badly on dense models
  where their lower bandwidth is fully exposed.

A concrete case: a dense 27B at Q4 runs at ~35 tok/s on a 3090 and ~15 on a
128 GB unified-memory box costing several times more, because 936 GB/s beats
273 GB/s and 16 GB fits in 24 GB either way. Reverse the model and the
conclusion reverses with it.
