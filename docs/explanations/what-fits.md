# What actually decides whether a model fits

Two numbers, and the second surprises people.

## Weights

All of them stay resident. There is no offloading here: llama.cpp can spill
layers to system RAM, but doing so drops throughput off a cliff, so this project
treats "fits entirely in VRAM" as the only interesting case.

A 24 GB card has less than 24 GB to spend:

```
24576 MiB  total
 -1024     compute buffers, graphs, fragmentation
 -2400     a typical desktop session (compositor, browser)
========
21152 MiB  for weights + KV cache
```

Running headless recovers that 2400 MiB, which is worth a meaningful slice of
context. The panel computes with the desktop reserve by default because most
people are sitting at the machine.

## KV cache — the number that actually binds

Every token in the context costs cache, and the cost per token is a property of
the architecture, not of the model's size:

```
bytes/token = full_attention_layers × 2 (K,V) × kv_heads × head_dim × dtype_bytes
```

Layers that are *not* full attention change this completely:

- **Linear attention** (GatedDeltaNet, Mamba) layers hold **no** per-token KV at
  all. They carry a fixed-size recurrent state per sequence instead.
- **Sliding-window** layers are bounded, but not by the window: engines
  provision them as a fraction of the whole token budget, because a prefix cache
  over sliding layers must retain far more than one window to serve a hit.

The consequence is that KV cost does not track parameter count:

| model | attention layout | KiB/token |
|---|---|---|
| Qwen3-8B | 36 full, 8 heads × 128 | **144** |
| Qwen3.8-27B | 16 full of 64, 4 heads × 256 | 64 |
| gpt-oss-20b | 12 full of 24, 8 heads × 64 | 24 |
| Qwen3.6-35B-A3B | 10 full of 40, 2 heads × 256 | **20** |

The 8B model costs seven times more cache per token than the 35B. Size and
context are independent axes.

## Putting them together

```
context = (budget − weights) ÷ bytes_per_token
```

capped by the model's own RoPE ceiling, past which output becomes incoherent
rather than merely expensive.

The cache is also compressible. This project runs the engine with `q8_0` key and
value caches, which halves the per-token cost for close to no quality loss and
is what makes 200k context on a 15 GB model possible at all. `q4_0` would halve
it again but degrades long-context reasoning noticeably, so it is not offered.

That is the whole calculation the panel performs, in
`llm3090.catalog.fit`. It runs before any download, which is the point: a
20 GB download is an expensive way to learn something a config file could have
told you.
