# Use Claude Code against a local model

llama.cpp implements Anthropic's `/v1/messages`, including streaming and tool
calls, so Claude Code talks to it directly — no proxy, no shim.

```bash
lllm3090 start Qwen3.6-35B-A3B
lllm3090 claude
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

To see exactly what would be set — against the model actually running, with the
window actually computed — ask for it rather than reading the list above:

```bash
lllm3090 claude --print-env
```

That prints `export` lines and nothing else on stdout, so it also serves any
harness this project has no command for:

```bash
eval "$(lllm3090 claude --print-env)"
```

Claude Code's variables are not a versioned contract — a release can add one or
rename one. `--print-env` is what makes that checkable without starting a
session to find out.

All three model slots point at the local model deliberately, so switching with
`/model` inside that session stays local rather than silently falling back to
the paid API.

## What to expect

The honest version: it works, and it is not the same experience.

- **Speed is fine.** 126 tok/s on `Qwen3.6-35B-A3B` — the model the command
  above starts — is comfortably faster than you read. The dense `Qwen3.8-27B`
  manages 35, which is still usable for reading code and answering questions.
- **The first turn on a big repo is slow.** Vulkan prefill costs about two
  minutes at 80k tokens. Every turn after that returns in a few seconds,
  because the prefix cache holds the prompt.
- **Context is the real limit, not speed.** Claude Code's system prompt and tool
  definitions are ~40k tokens on *every* turn before your work starts. A local
  model also tends to spend far more of its window reasoning than a frontier
  model does on the same question. This is why the recommendation is the
  35B-A3B: 212k per conversation leaves room for the work after the harness has
  taken its share.
- **Check quantities, not existence.** In a measured comparison the local model
  produced no hallucinations, but did inflate quantifiers — "called everywhere"
  for two call sites. The shape of its claims was right; the magnitudes were
  asserted rather than counted.

## A patched chat template — only for `Qwen3.8-27B`

This section applies to the dense option, not to `Qwen3.6-35B-A3B`, which needs
no patch and is what the command at the top starts. Read it if you switch to
the 27B, or if you add a model of your own whose template rejects Claude Code.

`Qwen3.8-27B` is served with a modified chat template, shipped as
`lllm3090/data/qwen3.8-27b.jinja`. Its own template counts the system messages
at the start of a conversation and then raises on any later one:

```jinja
{{- raise_exception('System message must be at the beginning.') }}
```

Claude Code legitimately sends a system message mid-conversation, so without the
patch every request after the first fails with a 500 and a Jinja traceback that
does not obviously point at the cause. The patch renders that message as an
ordinary system turn instead of failing. It is a one-line change and the reason
is written into the template file.

Any catalogue entry can carry a `chat_template:` field; the engine passes
`--chat-template-file` only for models that declare one.

## Subagents share one KV pool

The context you configure is a *global* pool shared by every concurrent request,
not a per-conversation budget. A parent session at 75k tokens plus a subagent
carrying its own 40k system prompt will not both fit in a 131k pool: the
scheduler serialises them, and the subagent's prefill can evict the parent's
cached prefix so the parent then pays a full cold prefill.

`lllm3090` handles this for you: the engine is started with two slots by
default and `lllm3090 claude` reports the **per-conversation** window to Claude
Code, not the pool. Tell Claude Code the pool size and it will fill the whole
thing, which is precisely the failure above.

If you want more concurrent agents, `lllm3090 start <model> --parallel 4` — at
the cost of a quarter of the pool each. Models whose KV is cheap take this far
better; see the `kv_kib_per_token` column in [](../reference/catalogue.md).
