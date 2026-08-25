# Serve a GGUF that is not in the catalogue

Any GGUF works. Put it in its own directory under the models directory
(`~/models` by default, or `$LLLM3090_MODELS_DIR`):

```bash
mkdir -p ~/models/My-Model
mv My-Model-Q4_K_M.gguf ~/models/My-Model/
```

It appears under **Installed** in the panel immediately — discovery is just "a
directory containing at least one `.gguf`". Multi-part GGUFs work too: point at
the directory and the engine loads the first shard, which pulls in the rest.

## Context for an unknown model

The catalogue carries a `default_ctx` for models it knows. For anything else the
panel falls back to a conservative **32768**. To use more:

```bash
lllm3090 start My-Model --ctx 131072
```

If you ask for more than fits, llama-server fails at allocation and the panel
shows the error. To work out what will fit before trying, see
[](../explanations/what-fits.md) — the short version is that you need the
model's full-attention layer count, KV head count and head dimension, all of
which are in its `config.json` on HuggingFace.

## Adding it to the catalogue properly

If it is worth keeping, add it to `src/lllm3090/data/models.yaml` and send a pull
request. Every field is documented in [](../reference/catalogue.md); the one
that takes thought is `kv_kib_per_token`, and getting it wrong makes the panel
promise context the card cannot deliver.
