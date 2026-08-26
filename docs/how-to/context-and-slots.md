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
| gpt-oss-20b | 128k (max) | **128k (max), 4 slots** | 128k (max) |
| Qwen3-8B | 32k (max) | **32k (max), 4 slots** | 32k (max) |

Two rows behave differently from the rest. **`gpt-oss-20b` and `Qwen3-8B` hit
their architectural ceiling long before they run out of VRAM**, so extra slots
cost them nothing: the spare cache cannot become a longer conversation, so it
becomes more of them.

Those two are therefore started with **four** slots rather than two, each still
holding the model's full window. You get the concurrency for free and give up
nothing. The automatic grant stops at four — llama.cpp sizes some buffers per
slot, and a fifth simultaneous conversation on one GPU is of speculative value —
but `--parallel 6` is yours if you want it.

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

It is also told **how many conversations there is room for**. Claude Code's own
default is 20 concurrent subagents, which against a two-slot pool is a promise
of twenty conversations where there is room for two. `lllm3090 claude` asks the
running engine (`GET /props` → `total_slots`), keeps one slot for the parent,
and sets `CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS` to the rest — one subagent at
the default, three at `--parallel 4`.

Overshooting the slot count is not an error, which is why this is worth doing:
llama.cpp queues the excess rather than refusing it, and each subagent prefills
into whichever slot it lands in, so the limit arrives disguised as the model
being slow. With the cap, Claude Code serialises deliberately instead.

:::{note}
The cap governs subagents, not requests. `/btw` — Claude Code's side-question
command — is a second concurrent request carrying the conversation, and it is
not a subagent, so nothing here restrains it. On two slots, a `/btw` issued
while a subagent is running waits for a slot and then re-prefills the whole
conversation into it.
:::

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
