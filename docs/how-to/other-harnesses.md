# Use a client other than Claude Code

The engine speaks the **OpenAI API** as well as Anthropic's, so anything that
lets you set a base URL works — aider, Continue, LibreChat, the `openai` and
`anthropic` SDKs, or ten lines of `curl`.

```bash
export OPENAI_BASE_URL=http://127.0.0.1:1919/v1
export OPENAI_API_KEY=local          # unchecked, but most clients demand one
```

Use whatever `GET /v1/models` reports as the model name. There is no
authentication: any token is accepted, which is why the engine binds loopback
(see [](remote-access.md)).

## Why the client matters more than you would expect

Every request carries the harness's system prompt and tool definitions before
any of your work, and that overhead is subtracted from the window on **every
turn**. It is a floor, not a one-off cost.

Claude Code sends roughly **40k tokens** of system prompt and tool schemas. A
plain chat client sends a few hundred. That is not a small difference — it
decides which models are usable at all:

| model | window | with a ~40k harness | with a ~2k harness |
|---|---|---|---|
| Qwen3-8B | 32k | **cannot run** | ~30k of working room |
| gpt-oss-20b | 128k | ~88k | ~126k |
| Qwen3.8-27B | 101k | ~61k | ~99k |
| Qwen3.6-35B-A3B | 212k | ~172k | ~210k |

`Qwen3-8B` is the striking case. Its 32k window cannot hold Claude Code's system
prompt at all, so the first message fails — yet the same model is perfectly
usable from a lighter client with 30k of room to work in. **A model that "does
not work" may only be failing to fit its harness.**

## Measure your own client

Do not trust the figures above for a client you actually use — measure it. The
engine reports token counts on every response:

```bash
curl -s http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODEL","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"])'
```

Send `hi` through your harness and compare its `prompt_tokens` against that
baseline. The difference is what the harness costs you per turn, and it is the
number to plan context around.

For a prompt you have in a file, ask the engine directly without generating
anything:

```bash
curl -s http://127.0.0.1:1919/tokenize \
  -H 'Content-Type: application/json' \
  -d "{\"content\": $(python3 -c 'import json,sys;print(json.dumps(open(sys.argv[1]).read()))' prompt.txt)}" \
  | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["tokens"]), "tokens")'
```

## Pi

[Pi](https://github.com/earendil-works/pi) is a bring-your-own-key CLI coding
agent that is provider-agnostic by design, which makes it the least awkward
second harness to put beside Claude Code.

```bash
npm install -g --ignore-scripts @earendil-works/pi-coding-agent
```

Point it at the engine by adding a provider to `~/.pi/agent/models.json`. Use
whatever `GET /v1/models` reports as the model id — that is the name you started
the model under:

```json
{
  "providers": {
    "lllm3090": {
      "baseUrl": "http://127.0.0.1:1919/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [
        { "id": "Qwen3.8-27B" },
        { "id": "Qwen3.6-35B-A3B" }
      ]
    }
  }
}
```

Only `id` is required for a local model; the key is unchecked but most clients
insist on one. Add an entry per model you might start — the engine serves one at
a time, so the list is what you can switch between, not what runs at once.

Check Pi's own [models documentation](https://pi.dev/docs/latest/models) if the
config schema has moved; this file will drift and theirs will not.

## opencode

[opencode](https://opencode.ai/docs/providers/) works the same way and has a
fuller agentic feature set, at the cost of a larger configuration surface.
Both it and Pi write configuration to disk, which is worth knowing if you value
the property that `lllm3090 claude` has: it sets environment variables for a
single subprocess and leaves no trace, so a plain `claude` elsewhere still
reaches Anthropic on your normal account.

Neither tool offers that, so decide deliberately whether you want your default
`opencode` or `pi` invocation pointing at a local model or a hosted one.

## Slots and concurrency

A lighter client changes the slot arithmetic too. With a 40k floor, a 32k slot
is useless; with a 2k floor it is a working session. So a small-prompt harness
makes higher `--parallel` counts genuinely useful where an agent harness would
need every slot to be large — see [](context-and-slots.md).

## Anthropic-protocol clients

Anything speaking Anthropic's API works against `/v1/messages` in the same way,
which is how Claude Code connects without a proxy. Streaming and tool calls are
both supported; see [](claude-code.md).
