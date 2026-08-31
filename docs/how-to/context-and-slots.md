# Choose a context window and slot count

Starting a model involves two numbers that are easy to confuse:

- the **pool** — total KV cache tokens, bounded by VRAM;
- the **per-conversation window** — what one request may use, bounded by the
  pool divided by the slot count, and by the model's RoPE ceiling.

```bash
lllm3090 start Qwen3.8-27B                 # automatic: fills the window first
lllm3090 start Qwen3.8-27B --parallel 2    # room for an agent and one subagent
lllm3090 start Qwen3.8-27B --parallel 4    # four conversations, quarter each
lllm3090 start Qwen3.8-27B --ctx 131072    # set the whole pool by hand
```

## The automatic rule: fill the window, then split only if it pays

The pool is a fixed number of tokens, so splitting it does not create capacity
— it shortens every conversation and buys concurrency with the difference. One
long conversation is therefore the default.

But the window is bounded by the model's **RoPE ceiling** as well as by VRAM,
and past that ceiling the remaining cache can never become a longer
conversation. Refusing to split then strands it: a pool holding 2.8 windows
would give you one full window and waste 1.8 windows of cache.

So the test is on **total** usable context. If splitting raises it by
`SLOT_SPLIT_GAIN` (**1.5x**, in `config.py`) or more, split;
otherwise keep the single window and accept the remainder as unusable.

How far to split is a second question, and "as far as consumes the whole pool"
is the wrong answer — it produces a cliff where a *larger* pool yields a
*shorter* conversation. `Qwen3.6-35B-A3B` hit exactly that: 256k on a desktop
and 184k headless, so freeing the compositor's VRAM made the window worse. Each
further slot is therefore taken only while it recovers more stranded cache than
it costs in window. Muse-Glimmer's third slot recovers 28% of its pool for 8% of
its window and is taken; the A3B's third recovers 7% for 28% and is not.

Raise `SLOT_SPLIT_GAIN` to favour one long conversation, lower it to favour
concurrency.

**If you are running an agent, ask for two.** An agent that spawns subagents
needs room for more than one conversation: with a single slot the scheduler
serialises them, and each subagent's prefill evicts the parent's cached prefix,
so the parent pays a full cold prefill on its next turn. `lllm3090 claude` says
so when it finds a one-slot engine.

## What each model gives you

Per-conversation window on a 24 GB card with a desktop session running. "(max)"
means the model's RoPE ceiling was reached:

| model | automatic | `--parallel 1` | `--parallel 2` |
|---|---|---|---|
| Qwen3.8-27B | **168k x1** | 168k | 84k |
| Qwen3.6-35B-A3B | **256k (max) x1** | 256k (max) | 169k |
| Qwen3.6-35B-A3B-MTP | **256k (max) x1** | 256k (max) | 148k |
| Qwen3.6-35B-A3B-Q4KS | **69k x1** | 69k | 34k |
| gpt-oss-20b | **128k (max) x4** | 128k (max) | 128k (max) |
| Qwen3-8B | **32k (max) x4** | 32k (max) | 32k (max) |
| Gemma-4-26B-A4B | **207k x1** | 207k | 103k |
| Muse-Glimmer-30B | **118k x3** | 128k (max) | 128k (max) |
| Gemma-4-12B-QAT | **256k (max) x4** | 256k (max) | 256k (max) |

Figures are computed for the detected card, so `lllm3090 models` is authoritative
on yours; these are the reference 3090's. Running headless returns about 2.4 GiB
and moves several rows up.

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
