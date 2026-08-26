# Serving your first model

## Prove the install with a small one

Open <http://127.0.0.1:8080>. Under **Available to download**, pick
**Qwen3-8B** — 5 GB, so you find out the download and serve path works without
waiting for 15 GB.

Progress streams in the panel. When it finishes it moves to **Installed**;
select it and press **Start**.

Watch the engine log at the bottom of the panel. A first load looks like:

```
load_model: loading model '/home/giles/models/Qwen3-8B/Qwen3-8B-Q4_K_M.gguf'
...
srv  load_model: the model is loaded
```

The panel's engine pill goes `stopped` → `loading` → `running`. The distinction
matters: llama-server binds its HTTP port long before the weights finish
uploading to the card, so "the port is open" is not "the model is ready".

## Talk to it

```bash
curl -s http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"Hello"}],"max_tokens":30}'
```

Or from the CLI:

```bash
lllm3090 status
```

:::{note}
Qwen3-8B is a reasoning model and thinks at length before answering. With a
small `max_tokens` you may get a `thinking` block and no text at all — that is
the model spending its budget, not a broken endpoint. Give it a few hundred
tokens of headroom.
:::

## Move to a real model

`Qwen3-8B` is a smoke test — and too small for an agent harness, since its 32k
window cannot hold Claude Code's ~40k system prompt. For actual work:

**`Qwen3.6-35B-A3B`** (17.7 GB) is the default recommendation. Measured at
**126 tok/s**, and its hybrid attention makes the KV cache cheap enough
(20 KiB/token) to reach the model's full **262k** context — 212k per
conversation with a slot spare for a subagent.

**`gpt-oss-20b`** (12.1 GB) is faster still at **160 tok/s** and leaves most of
the card free, at 128k of context across four slots. Worth having.

**`Qwen3.8-27B`** (15.4 GB) is the dense option. It is the *slowest* model here
at 35 tok/s — every one of its 27B parameters is read per token, where the
sparse models read about 3B — and it gives 101k per conversation. Take it if
you want a dense model's qualities and can afford a quarter of the speed; the
comparison is [](../explanations/dense-vs-moe.md).

## Free the card

```bash
lllm3090 stop
```

or press **Stop** in the panel. Do this before gaming, or before anything else
that wants the VRAM — a loaded model holds 12–21 GB indefinitely.
