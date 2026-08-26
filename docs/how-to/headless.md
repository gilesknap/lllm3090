# Run headless, and get the compositor's VRAM back

A desktop session holds VRAM that the model could be using. On a 24 GB card
that is not a rounding error: it is most of a 35B model's cache, and for one
entry in the catalogue it is **three times** the context.

## Do it

```bash
sudo systemctl isolate multi-user.target
```

That drops to a text console immediately. No reboot, nothing uninstalled, and
the panel and engine keep running — they are a user service and a detached
process, not part of the graphical target. Go back with:

```bash
sudo systemctl isolate graphical.target
```

Log in on the console (Ctrl-Alt-F3 if you need another one) and use
`lllm3090` exactly as before, or reach the panel over an SSH tunnel — see
[](remote-access.md).

## What it is worth

Measured on the reference RTX 3090, per conversation:

| model | desktop | headless | |
|---|---|---|---|
| Qwen3.6-35B-A3B-Q4KS | 61k × 2 | **181k × 2** | +197% |
| Gemma-4-26B-A4B | 138k × 2 | **256k × 2** | +86% |
| Qwen3.8-27B | 101k × 2 | 139k × 2 | +38% |
| Muse-Glimmer-30B | 128k × 3 | 128k × **4** | one more slot |
| Qwen3.6-35B-A3B | 212k × 2 | 256k × 2 | +21% |
| gpt-oss-20b, Qwen3-8B, Gemma-4-12B-QAT | — | — | no change |

The models that gain nothing are already stopped by their own RoPE ceiling
rather than by the card, so extra cache buys them nothing — see
[](../explanations/what-fits.md).

The models that gain most are the ones the desktop was squeezing hardest.
`Qwen3.6-35B-A3B-Q4KS` is 20.9 GB of weights on a 24 GB card, so the
compositor's share was coming almost entirely out of its cache.

## You do not have to tell it

`lllm3090` asks systemd whether `graphical.target` is running and plans
accordingly, so the figures in `lllm3090 models` and in the panel are already
the ones that apply to the session you are in. Switch targets and the numbers
change with it.

If it cannot tell — no `systemctl`, or the call fails — it assumes a desktop is
running. The two mistakes are not symmetric: over-reserving costs you context,
while under-reserving produces an engine that loads, reports itself healthy and
then fails every request with `ErrorOutOfDeviceMemory`.

:::{warning}
Context is chosen when the engine **starts**. Plan a model on a text console,
then switch back to `graphical.target` without restarting it, and the engine is
now sized for VRAM that the compositor has taken back. `lllm3090 start` compares
the plan against actually-free VRAM and warns, but it cannot see a desktop that
arrives afterwards. Restart the engine after switching.
:::

## Is it worth it?

For interactive use with an agent, usually yes when the model is a tight fit,
and not otherwise. If your model is already RoPE-capped on a desktop, headless
buys nothing at all and costs you a working desktop.

The place it pays is a long unattended run — an overnight batch, a large repo
reviewed in one pass — where nothing needs the screen and the extra cache is
the difference between one conversation and several, or between compaction and
none.
