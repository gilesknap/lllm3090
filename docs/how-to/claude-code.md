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
CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS=<the engine's slots, less the parent's>
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

The subagent limit comes from the engine rather than from a guess: `GET /props`
reports `total_slots`, one stays with the parent, and the rest is the cap.
Claude Code's own default is 20, which on a two-slot engine would let it plan a
fan-out ten times wider than the card can hold — and llama.cpp queues the
excess rather than refusing it, so that arrives as slowness rather than as an
error. See [](context-and-slots.md).

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

## `/effort` does not reach the model — `start --effort` does

Claude Code's `/effort` command is accepted inside the session and changes
nothing about what the local model does. It travels as `output_config.effort`
in the `/v1/messages` body, and llama.cpp does not implement that field.
Unknown fields are dropped rather than refused: a request carrying
`output_config.effort` returns 200, exactly as one carrying an invented field
does. Because there is no 400, Claude Code never learns the level went nowhere
— it only latches "unsupported" on a rejection — so the TUI goes on reporting
an effort that never left the client. `CLAUDE_CODE_EFFORT_LEVEL` is the same
dead end; it only changes what goes into that ignored field.

The knob does exist, under another name. llama.cpp takes `reasoning_effort` and
hands it to the chat template — but only on `/v1/chat/completions`. On
`/v1/messages`, the endpoint Claude Code uses, even that is ignored, so a proxy
that injects the field into the Anthropic body will not help either. What is
left is the engine's own launch argument:

```bash
lllm3090 start Qwen3.8-27B --effort low
lllm3090 claude
```

That is `--reasoning-effort` on `llama-server`: one level for the life of the
process, shared by every session and every subagent on it. Levels are
`minimal`, `low`, `medium`, `high`, `xhigh` and `max`; omit the option and the
template's own default stands.

Two things worth knowing before you pick one:

- **The default is not neutral.** `Qwen3.8-27B`'s template resolves to `xhigh`
  when nothing is passed and injects "think carefully through the task,
  validate key assumptions, consider plausible alternatives" into the system
  turn. Every Claude Code turn against it is thinking at the longest setting
  unless you say otherwise. Where the window, not the clock, is what runs out
  first — see [](context-and-slots.md) — shorter thinking buys more than faster
  thinking would.
- **Not every model implements every level.** The level is passed through to
  the template, and a template is free to raise on one it does not know:
  `Qwen3.8-27B` implements `low`, `medium` and `xhigh`, folds `high` into
  `xhigh`, and raises on the rest. That failure is a 500 on every request from
  an engine that loads normally and answers `/health` — so `start` renders one
  prompt before reporting ready, and refuses the start rather than leaving that
  engine up.

On `gpt-oss-20B`, where the same control was measured, it changes output
*length* and not speed: 215 tokens at `minimal` against 396 at `high`, both at
about 145 tok/s. For agent work that is the useful direction — it shortens the
pause before the tool call.

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
