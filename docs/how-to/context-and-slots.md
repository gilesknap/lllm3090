# Choose a context window and slot count

Starting a model involves two numbers that are easy to confuse:

- the **pool** — total KV cache tokens, bounded by VRAM;
- the **per-conversation window** — what one request may use, bounded by the
  pool divided by the slot count, and by the model's RoPE ceiling.

```bash
lllm3090 start Qwen3.8-27B                 # default: 2 slots
lllm3090 start Qwen3.8-27B --parallel 1    # one conversation, biggest window
lllm3090 start Qwen3.8-27B --parallel 4    # four conversations, quarter each
lllm3090 start Qwen3.8-27B --ctx 131072    # set the whole pool by hand
```

The default is **two**, so an agent and one subagent fit at once. See
[](../explanations/what-fits.md) for why a pool sized for exactly one
conversation makes subagents serialise and evict their parent's cached prefix.

## What each model gives you

Per-conversation window at each slot count. "(max)" means the model's RoPE
ceiling was reached and there is spare VRAM that cannot be spent on context:

| model | `--parallel 1` | 2 (default) | 4 |
|---|---|---|---|
| Qwen3.8-27B | 203k | 101k | 50k |
| Qwen3.6-35B-A3B | 256k (max) | 212k | 106k |
| Qwen3.6-35B-A3B-Q4KS | 122k | 61k | 30k |
| gpt-oss-20b | 128k (max) | 128k (max) | 128k (max) |
| Qwen3-8B | 32k (max) | 32k (max) | 32k (max) |

Two rows behave differently from the rest, and the difference is worth
understanding. **`gpt-oss-20b` and `Qwen3-8B` hit their architectural ceiling
long before they run out of VRAM**, so extra slots cost them nothing at all —
four concurrent conversations, each with the full window. For those models
concurrency is free and there is no reason to run fewer than you might use.

The other three trade linearly: every doubling of slots halves the window.

## Which to choose

**`--parallel 1`** — one long conversation and nothing else. A single session
reading a large codebase, or a batch job. Gives the largest window the card can
hold.

**`--parallel 2`** (default) — anything agentic. Claude Code spawns subagents on
the same engine, and this is the minimum that keeps a parent and one subagent
resident together.

**`--parallel 4`** — several agents, or a box shared with other people or
services. Only comfortable on a model whose KV is cheap; `Qwen3.6-35B-A3B` still
gives 106k per conversation at four slots, while `Qwen3.6-35B-A3B-Q4KS` drops to
30k, which is below what an agent harness needs to function.

:::{note}
More slots do not mean more total throughput. Every slot shares one GPU, so
four concurrent requests each decode at roughly a quarter of the solo rate.
Slots buy **admission** — the ability to start without queueing behind another
conversation — not speed.
:::

## What Claude Code is told

`lllm3090 claude` reports the **per-conversation** window, never the pool. This
matters: told the pool size, Claude Code will happily fill the whole thing and
leave nothing for the subagents sharing it, which is the exact failure the slot
count exists to prevent.

So with `Qwen3.8-27B` at the default, Claude Code sees 101k and compacts there,
while the engine holds 202k across two slots.

## Overriding the pool directly

`--ctx` sets the whole pool and bypasses the planner:

```bash
lllm3090 start Qwen3.8-27B --ctx 262144 --parallel 2   # 131k each
```

If you ask for more than fits, llama-server fails at KV allocation and the panel
shows the error — it fails at startup rather than midway through a session, which
is the good outcome. The planner exists so you do not have to do this
arithmetic; reach for `--ctx` when you want something it would not choose, not
to work out what fits.
