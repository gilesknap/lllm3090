---
name: claude-code-on-local-models
description: Pointing Claude Code at a local Anthropic/OpenAI-compatible endpoint, and knowing what actually limits the session — the KV pool is global and shared with subagents, the system prompt sets a hard context floor, and context runs out long before tokens-per-second does. Use when launching an agent against a local model, sizing a KV pool, when subagents serialise or p95 latency spikes, or when comparing a local model against a hosted one.
---

# Driving Claude Code from a local model

FreeToken answers on Anthropic's protocol (`/v1/messages`) as well as OpenAI's,
so Claude Code cannot tell it is not talking to Anthropic. No shim, no proxy, no
plugin.

## Launch by environment, never by config file

```bash
lllm3090 claude       # sets, for that subprocess only:
ANTHROPIC_BASE_URL=http://127.0.0.1:1919
ANTHROPIC_AUTH_TOKEN=local
ANTHROPIC_MODEL=<checkpoint>
ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL=<checkpoint>
CLAUDE_CODE_SUBAGENT_MODEL=<checkpoint>
CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144
CLAUDE_CODE_MAX_OUTPUT_TOKENS=32768
```

`lllm3090 claude` writes **nothing** to `~/.claude/settings.json` — it hands
the variables to one subprocess, so a plain `claude` in another terminal still
reaches the hosted model on normal auth. That is what makes head-to-head
comparison possible, and it is worth preserving. Tools that write their configuration to disk instead (some agent CLIs do)
cannot be made non-invasive this way -- check before pointing one at a local
endpoint.

Point all three default model slots at the local checkpoint deliberately, so
`/model` inside that session stays local instead of silently falling back to the
paid API.

Claude Code believes whatever `CLAUDE_CODE_MAX_CONTEXT_TOKENS` says. Set it from
the context the engine was actually started with, never from the ceiling the
model advertises -- the two are routinely different by 2x.

## The floor nobody budgets for

Claude Code's system prompt and tool definitions are **~40k tokens on every
turn**, before any of your work. Consequences:

- A KV allocation under ~60k tokens cannot run the harness at all — the first
  message dies with `Prompt is too long`.
- "The model supports 262k context" is a statement about RoPE, not about what is
  served. The KV allocation is the ceiling.
- Sizing a card for agentic work means sizing the KV cache first and giving the
  weights what is left, not the other way round.

## The KV pool is global, not per-conversation

This is the detail that surprises people. Every concurrent request draws on one
shared pool, and subagents run on the same engine
(`CLAUDE_CODE_SUBAGENT_MODEL`).

With a 131k pool, one session sitting at 75k tokens holds 58% of it. There is
then no room to admit a subagent carrying its own ~40k system prompt, so the
scheduler serialises them — `#running-req: 1, #queue-req: 1` despite
`max_running_requests=4`. Worse, the subagent's prefill **evicts the parent's
radix-cached prefix**, so the parent's next turn pays a full cold prefill
instead of a 0.9 s warm hit. Measured p95: 73 s.

Set the two numbers independently:

```
--kv-reserve-tokens 262144      global pool: parent AND subagent resident
--max-seq-len-override 131072   what CC sees; it still compacts at 131k
```

Doubling the pool while leaving the reported context alone changed CC's
behaviour not at all and let two 60k sessions overlap (35.7 s, previously
serialised) for −5.3% throughput on short prompts and −1.1% at 40k. On a card
that is not expert-cache-bound this is close to free.

## Context is the constraint, not tokens per second

Measured on ws03, same repo, same prompt, local 35B-A3B against a hosted
frontier model:

| | local | hosted |
|---|---|---|
| wall clock | 2 m 14 s | 1 m 43 s |
| context used | **73%** (2% from auto-compact) | **9%** |

1.3× slower, not the 10× the price difference suggests — but it burned eight
times the window to do *less* exploration, and came within 2% of auto-compaction
on a single question. Small models pay for reasoning in context. That, and not
throughput, is what limits how long a local agentic session can run, and more
system RAM does not help it: the KV ceiling is VRAM.

Throughput barely degrades with depth (148 → 111 tok/s from empty to 100k), and
the radix prefix cache means a session pays the ~10 s cold prefill once and
returns to sub-second TTFT afterwards. Speed is rarely the thing to optimise.

## What a local model gets wrong

In a checked head-to-head the local model produced **no hallucinations**. Its
failure mode was **inflated quantifiers**: "called everywhere" for two call
sites, "every command requires this flag" for three of them. The shape of each
claim was right; the magnitude was asserted rather than counted.

So when reviewing local-model output, verify the *quantities*, not the
existence of what it names. And expect it to be weaker at groundedness —
citing lines, cross-referencing the issue tracker, excluding what the maintainer
already knows — rather than at reading code.

## Measuring fairly

- Run both sides against the same repo on the same machine, in two terminals.
- Gate every timing on the engine's `/slots` endpoint reporting
  `active=0` before *and* after; a live session on the same GPU once faked a 47%
  regression.
- On gpt-oss models, `reasoning_effort` changes output **length**, not speed
  (minimal 215 tokens, high 396, both ~145 tok/s). For agentic work that is the
  useful knob: it shortens the thinking pause.
- First request after a cold boot takes ~64 s — that is JIT warm-up, not a hang.
  Discard it.
