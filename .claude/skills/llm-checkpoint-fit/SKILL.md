---
name: llm-checkpoint-fit
description: Decide whether a model checkpoint will actually load and serve on a given GPU, before downloading tens of GB of it — what offloads and what does not, KV bytes per token from config.json, reading a safetensors header over an HTTP range request, checking the quantization layout against what the engine's loader expects, and a decode roofline. Use when asked "can I run <model> on this card", when choosing between quantized releases of the same model, or when a checkpoint OOMs or dies at load.
---

# Will this checkpoint fit?

Answer it from four small files and some arithmetic. Downloading first is a
20-GB way to learn what a 250-KB range request would have told you.

Work the steps in order — each one can end the enquiry on its own.

## 1. Is the architecture even registered?

Every engine keeps a registry of the architectures it can load, and "related
to something supported" is not the same as supported.

- **llama.cpp**: the architecture must be in `convert_hf_to_gguf.py`, and the
  build serving it must be new enough. A GGUF someone else converted proves the
  first half only.
- **A Python engine** (FreeToken and friends): grep its model registry for
  `ForCausalLM` / `ForConditionalGeneration` and match against
  `architectures[0]` in the repo's `config.json`:

  ```bash
  grep -n 'ForCausalLM\|ForConditionalGeneration' \
    <venv>/lib/python*/site-packages/freetoken/models/register.py
  ```

  Registry comments are worth reading -- they record which *variants* are
  handled (dense vs MoE, which wrapper config, which weight layout), and the
  distinction is load-time fatal. See [[freetoken-engine]] for a case where the
  architecture was registered and the checkpoint still would not load.

Related-but-unsupported is the common trap. FreeToken parses GGUF, but the
registry maps **only** `Gemma4GGUFForCausalLM` — a Q4 GGUF of any other model is
not an option however well it would fit.

## 2. What is actually resident? (the decisive question)

An offload engine offloads **routed MoE experts** to host RAM, and nothing else.

- **MoE** → VRAM holds attention, embeddings, KV, and an LRU *cache* of experts.
  Host RAM holds the full expert pool. The checkpoint can exceed VRAM.
- **Dense** → there are no routed experts. **100% of the weights are resident.**
  The checkpoint must fit in VRAM with the KV cache still to pay for.

Confirm rather than assume — `num_experts: 0` in `config.json`, and in the AOT
model list dense entries are marked `store/index only, no expert banks`.

This is why a 35B MoE serves comfortably on a card that cannot hold a 27B dense.
Parameter count is nearly irrelevant; *active* bytes per token is what matters.

## 3. Weight bytes: read the header, not the repo size

Repo totals include the vision tower, an MTP head, and files the loader skips.
The safetensors header is a JSON blob at the front of the file — fetch it with a
range request and total up only what will be loaded.

```bash
# header length is the first 8 bytes, little-endian u64
curl -sSL -r 0-7 "$URL" | xxd
curl -sSL -r 0-<len+7> "$URL" -o hdr.bin
```

```python
import json, struct
raw = open('hdr.bin','rb').read()
h = json.loads(raw[8:8+struct.unpack('<Q', raw[:8])[0]])
h.pop('__metadata__', None)
DT = {'F32':4,'F16':2,'BF16':2,'F8_E4M3':1,'F8_E5M2':1,'U8':1,'I8':1,'I32':4,'I64':8,'BOOL':1}
def sz(v):
    n = 1
    for d in v['shape']: n *= d
    return n * DT[v['dtype']]
# skip what the loader skips — check its iter_weights for the real prefix list
SKIP = ('mtp.', 'model.visual.', 'visual.')
print(sum(sz(v) for k, v in h.items() if not k.startswith(SKIP)) / 2**30, 'GiB')
```

Group by module prefix and dtype while you are in there: it shows you at a
glance which parts are 4-bit, which are fp8, and how much of the total is a
bf16 embedding table that no quantization touched.

## 4. KV cache per token — usually the number that decides it

```
bytes/token = full_attention_layers × 2 (K,V) × num_key_value_heads × head_dim × dtype_bytes
```

Count `full_attention_layers` from `layer_types`, or from
`num_hidden_layers / full_attention_interval` in a hybrid model. Linear-attention
(GatedDeltaNet / Mamba) layers hold **no paged KV** — they carry a fixed
per-sequence recurrent state instead:

```
per-sequence state ≈ linear_layers × linear_num_value_heads × linear_key_head_dim × linear_value_head_dim × 4 (fp32)
```

which is small per token but ~150 MB *per concurrent sequence* on a 27B-class
hybrid. Budget it if subagents run.

**Sliding-window layers are not free, and this is the trap.** It is natural to
assume an SWA layer costs `window × bytes` and therefore nothing at depth. The
engine does not size it that way: FreeToken provisions the SWA tier as a
*fraction of the whole token budget* (`--swa-full-tokens-ratio`, default 0.2),
because a radix prefix cache over SWA layers has to retain far more than one
window to serve a cache hit. So:

```
bytes/token = full_tier + swa_full_tokens_ratio × swa_tier
```

and the SWA tier can dominate, because SWA layers often carry many more KV heads
than the full-attention layers do. Gemma-4-26B-A4B is the cautionary case: 5 full
layers at 2 heads × 512 (20,480 B/token) and 25 sliding layers at 8 heads × 256
(204,800 B/token raw). Counting the sliding layers as "a flat 1024-token window"
gives 20 KiB/token; the real figure is `20,480 + 0.2 × 204,800` = **60 KiB/token**,
3× the estimate, and the difference is what decides whether a 262k pool fits.

Two models from the same family, same generation:

| | attention layout | **KiB/token** | 131k pool |
|---|---|---|---|
| Qwen3.6-35B-A3B | 10 full of 40 (30 linear), 2 × 256 | **20** | 2.5 GiB |
| Qwen3.8-27B | 16 full of 64 (48 linear), 4 × 256 | **64** | 8.0 GiB |
| Gemma-4-26B-A4B | 5 full (2 × 512) + 25 SWA (8 × 256) × 0.2 | **60** | 7.7 GiB |

The smallest model needs the most KV. Never carry a bytes-per-token figure across
models, not even within a family.

**Then let the engine check your arithmetic.** `--moe-cache-auto` plans the split
before allocating anything and asserts in bytes rather than OOMing later:

```
AssertionError: cache budget too small: minimum plan (moe=256 slots, kv=262144 pages)
needs 16966811648 B > budget 13163095654 B
(raise memory_ratio, lower kv_reserve_tokens, or free GPU memory)
```

Divide that `needs` figure by the page count and you have the engine's own
per-token cost — 61,456 B against the 61,440 B the formula above predicts. A
failed start is a free, exact measurement; take the number from it rather than
arguing with your estimate.

## 5. Does the quant layout match what the loader expects?

A repo named `-NVFP4` is not necessarily NVFP4 throughout. `compressed-tensors`
checkpoints carry `quantization_config.config_groups`, one entry per scheme:

```bash
curl -sSL "https://huggingface.co/$REPO/resolve/main/config.json" |
  python3 -c "import json,sys; q=json.load(sys.stdin)['quantization_config']; print(json.dumps(q['config_groups'],indent=1)[:2000])"
```

Read `num_bits`, `type`, `group_size`, `strategy` **and the `targets` regexes**
for every group. A `format: mixed-precision` build commonly quantizes the MLP to
NVFP4 and leaves attention, `lm_head` and the last few MLP layers at fp8 — which
inflates the download well above the 4-bit arithmetic and, worse, may not load
at all.

Then check what the loader assumes. FreeToken's dense path sets
`attn_quant = "nvfp4"` for *any* compressed-tensors checkpoint showing one NVFP4
group (`models/qwen3_5_moe/config.py`), and then looks for `.weight_packed` on
the attention projections. A build storing those as channel-wise fp8 W8A8
fails at load — the engine supports per-tensor fp8 (W8A16) and 128×128 block
fp8, not channel-wise W8A8. Same model, same bit width, different vendor,
different outcome.

## 6. Speed, if it fits: a roofline in one line

```
tok/s_max = memory_bandwidth / bytes_read_per_token
```

`bytes_read_per_token` is the resident weights minus the embedding table for a
dense model. RTX 3090 = 936 GB/s. Expect 60–80% of the roofline in practice.
This is arithmetic, not a measurement — label it as such when reporting.

For an **MoE** the roof is not VRAM bandwidth but the PCIe link the expert
fetch crosses, and the answer turns on the cache-hit fraction — a model can fit
comfortably and still decode at 8 tok/s. Use `moe-throughput-model` for that
case; it is what turns "will it fit" into "is it worth having".

## Putting the budget together

```
usable VRAM  = total − desktop/compositor (~2.3 GiB here) − workspace & CUDA graphs (~1 GiB)
KV tokens    = (usable − resident weights) / bytes_per_token
```

Then sanity-check against the *floor* the workload needs, not the number that
sounds impressive. An agent harness sends 40k+ tokens of system prompt and tool
definitions on every turn, so a KV budget under ~60k tokens cannot run one.

Worked example — Qwen3.8-27B NVFP4 on a 24 GB 3090:

```
text tower (header sum, vision + MTP excluded)  20.16 GiB resident, nothing offloads
usable VRAM with the desktop up                 21.75 GiB
left for KV                                      ~1.6 GiB  →  ~24k tokens
needed for a 131k session                        8.0 GiB
```

Fails on capacity by ~7 GiB, and separately on quant layout (step 5). Two
independent blockers, either one sufficient — which is the normal outcome, and
the reason to run the steps in order rather than reaching for the download.
