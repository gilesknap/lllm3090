# The model catalogue

`src/lllm3090/data/models.yaml` is the curated list. Every entry has been checked
to exist on HuggingFace and to fit a 24 GB card with usable context left.

## Fields

| field | meaning |
|---|---|
| `id` | Stable slug, used in API paths |
| `name` | Display name, **and the directory name** under the models dir |
| `repo` / `file` | HuggingFace repository and filename to download |
| `size_gb` | Download size, from the HuggingFace API |
| `kind` | `dense` or `moe` — see [](../explanations/dense-vs-moe.md) |
| `params` | Human description of the architecture |
| `kv_kib_per_token` | KV cache cost per token at f16 |
| `max_ctx` | The model's own RoPE ceiling; never exceeded |
| `expected_tok_s` | Decode rate on this card |
| `mmproj` | Optional multimodal projector in the same repo; present means vision |
| `mmproj_gb` | The projector's size, counted against VRAM like any other weights |
| `verified` | `true` = measured here; `false` = derived, treat as ±30% |
| `notes` | What the model is good and bad at |
| `tags` | Free-form labels |

`name` doubles as the directory name so that a download and an existing
checkout of the same model are recognised as the same thing.

## Deriving `kv_kib_per_token`

The one field that takes thought, and the one that makes the panel's context
promises true or false:

```
bytes/token = full_attention_layers × 2 × num_key_value_heads × head_dim × 2
```

Read `num_hidden_layers`, `layer_types` (or `full_attention_interval`),
`num_key_value_heads` and `head_dim` from the model's `config.json`. Count only
full-attention layers: linear-attention layers hold no per-token KV, and
sliding-window layers are a separate case covered in
[](../explanations/what-fits.md).

Some models specify a *different* geometry for their full-attention layers than
the top-level fields suggest — Gemma-style configurations carry
`num_global_key_value_heads` and `global_head_dim`, and using the flat fields
gives an answer several times wrong. Check for them.

## Current entries

| model | size | kind | KiB/token | per conversation | speed (measured) |
|---|---|---|---|---|---|
| Qwen3.8-27B | 15.4 GB | dense | 64 | 101k × 2 | 35 tok/s |
| Qwen3.6-35B-A3B | 17.7 GB | moe | 20 | 212k × 2 | **126 tok/s** |
| Qwen3.6-35B-A3B-Q4KS | 20.9 GB | moe | 20 | 61k × 2 | 124 tok/s |
| gpt-oss-20b | 12.1 GB | moe | 24 | 128k × 4 | **160 tok/s** |
| Qwen3-8B | 5.0 GB | dense | 144 | 32k × 4 | 115 tok/s |
| Gemma-4-26B-A4B | 18.2 GB | moe + vision | 20 | 189k × 2 | not measured |
| Muse-Glimmer-30B | 17.9 GB | dense + vision | 13 | 128k × 4 | not measured |
| Gemma-4-12B-QAT | 6.9 GB | dense + vision | 16 | 256k × 4 | not measured |

Sizes for the vision entries include their projector, because that is both what
gets downloaded and what occupies VRAM.

The first five figures are measured on an RTX 3090. The last three are **not
measured at all**, deliberately: the roofline method that would derive them is
calibrated between 20% and 35% of roof for resident MoE, which is a wider spread
than the ±30% this table's `verified: false` implies. A blank is honest where a
derived number would not be. `lllm3090 bench` is how they get filled in. The sparse models
decode 3-4× faster than the dense 27B despite being larger, which is the whole
argument in [](../explanations/dense-vs-moe.md).

"× 2" is the slot count: the cache is a pool shared by concurrent
conversations, and the default leaves room for two. See
[](../explanations/what-fits.md).

Nothing here is stored as a default. `lllm3090.catalog.plan` computes it at run
time from `size_gb`, `kv_kib_per_token` and `max_ctx`, so the figures stay true
if you run headless or ask for a different slot count. Storing a context default
would eventually disagree with what fits — and did, in an early version.
