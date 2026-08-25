# The model catalogue

`src/llm3090/data/models.yaml` is the curated list. Every entry has been checked
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
| `default_ctx` | What `start` uses when not told otherwise |
| `expected_tok_s` | Decode rate on this card |
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

| model | size | kind | KiB/token | context | speed |
|---|---|---|---|---|---|
| Qwen3.8-27B | 15.4 GB | dense | 64 | ~203k | 35 tok/s (measured) |
| Qwen3.6-35B-A3B | 17.7 GB | moe | 20 | 262k | ~90 tok/s |
| Qwen3.6-35B-A3B-Q4KS | 20.9 GB | moe | 20 | ~122k | ~90 tok/s |
| gpt-oss-20b | 12.1 GB | moe | 24 | 131k | ~80 tok/s |
| Qwen3-8B | 5.0 GB | dense | 144 | 32k | ~60 tok/s |

Context figures are computed at run time for a q8 KV cache with a desktop
session running; a headless box gets more.
