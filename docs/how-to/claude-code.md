# Use Claude Code against a local model

llama.cpp implements Anthropic's `/v1/messages`, including streaming and tool
calls, so Claude Code talks to it directly — no proxy, no shim.

```bash
llm3090 start Qwen3.8-27B
llm3090 claude
```

That sets Anthropic environment variables **for one subprocess only**:

```
ANTHROPIC_BASE_URL=http://127.0.0.1:1919
ANTHROPIC_AUTH_TOKEN=local
ANTHROPIC_MODEL=<the served model>
ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL=<the served model>
CLAUDE_CODE_SUBAGENT_MODEL=<the served model>
CLAUDE_CODE_MAX_CONTEXT_TOKENS=<the catalogue's default context>
CLAUDE_CODE_MAX_OUTPUT_TOKENS=32768
```

Nothing is written to `~/.claude/settings.json`, so a plain `claude` in another
terminal still reaches Anthropic on your normal account. You can run both side
by side and compare them on the same task.

All three model slots point at the local model deliberately, so switching with
`/model` inside that session stays local rather than silently falling back to
the paid API.

## What to expect

The honest version: it works, and it is not the same experience.

- **Speed is fine.** 35 tok/s on the 27B is perfectly usable for reading code
  and answering questions.
- **The first turn on a big repo is slow.** Vulkan prefill costs about two
  minutes at 80k tokens. Every turn after that returns in a few seconds,
  because the prefix cache holds the prompt.
- **Context is the real limit, not speed.** Claude Code's system prompt and tool
  definitions are ~40k tokens on *every* turn before your work starts. A local
  model also tends to spend far more of its window reasoning than a frontier
  model does on the same question.
- **Check quantities, not existence.** In a measured comparison the local model
  produced no hallucinations, but did inflate quantifiers — "called everywhere"
  for two call sites. The shape of its claims was right; the magnitudes were
  asserted rather than counted.

## Subagents share one KV pool

The context you configure is a *global* pool shared by every concurrent request,
not a per-conversation budget. A parent session at 75k tokens plus a subagent
carrying its own 40k system prompt will not both fit in a 131k pool: the
scheduler serialises them, and the subagent's prefill can evict the parent's
cached prefix so the parent then pays a full cold prefill.

If you use subagents, prefer a model whose KV is cheap — see the
`kv_kib_per_token` column in [](../reference/catalogue.md) — and give it the
largest context that still fits.
