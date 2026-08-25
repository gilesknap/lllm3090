# HTTP API

## The panel — `127.0.0.1:8080`

Loopback only. No authentication: these endpoints start processes and write to
disk.

| method | path | purpose |
|---|---|---|
| `GET` | `/` | The control panel UI |
| `GET` | `/api/status` | Engine state, VRAM, installed models, catalogue, downloads |
| `POST` | `/api/start?model=NAME&ctx=N` | Stop any engine, start this model. `ctx` optional |
| `POST` | `/api/stop` | Stop the engine |
| `POST` | `/api/download/{id}` | Begin downloading a catalogue entry; returns immediately |
| `POST` | `/api/download/{id}/cancel` | Cancel; the part file is kept for resume |
| `GET` | `/api/logs?tail=N` | Last N engine log lines |
| `GET` | `/api/logstream` | Engine log as Server-Sent Events |

`start` and `stop` take a lock and return **409** if one is already in progress.
`/api/status` is safe to poll; the UI does so every two seconds.

## The engine — `127.0.0.1:1919`

llama-server's own API. The two that matter:

- **`POST /v1/chat/completions`** — OpenAI-compatible.
- **`POST /v1/messages`** — Anthropic-compatible, including streaming and tool
  calls. This is what lets Claude Code connect without a proxy.

Also available: `/health`, `/v1/models`, `/v1/completions`, `/v1/embeddings`,
`/props`, `/slots`, `/metrics`. Full documentation lives with
[llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

The engine is started with a `q8_0` KV cache and flash attention on; see
[](../explanations/what-fits.md) for why.
