# Troubleshooting

## `llm3090 doctor` fails on Vulkan

```
[FAIL] vulkan  NVIDIA Vulkan ICD missing
```

The engine reaches the GPU through Vulkan, not CUDA. Install your driver's
Vulkan support — on Debian that is usually `nvidia-driver-libs`, and
`/usr/share/vulkan/icd.d/nvidia_icd.json` should then exist.

## The engine says "loading" and never becomes "running"

Read the engine log in the panel. Two common causes:

- **Genuinely still loading.** A 15–21 GB model takes a while to upload to the
  card. The panel distinguishes `loading` from `running` precisely so this is
  visible rather than looking like a hang.
- **Out of memory at KV allocation.** The weights fit but the context does not.
  Start it with a smaller `--ctx`, or pick a smaller quantisation.

## It fails immediately with a port error

Something else holds 1919. `llm3090 stop` clears an engine this tool started;
if another process owns the port, find it with:

```bash
ss -ltnp 'sport = :1919'
```

## The model loads but answers are gibberish

Almost always a quantisation too far. `IQ2` and `Q2` builds of a small model
are frequently incoherent. Move up to `Q4_K_S` or better — the catalogue only
lists 4-bit and above for this reason.

## Downloads keep failing

The panel resumes: a cancelled or failed download leaves a `.part` file and the
next attempt continues from where it stopped. If a download fails repeatedly,
check free space (`llm3090 doctor` reports it) and that the file still exists in
the HuggingFace repository — model authors do rename and remove files.

## The panel is running but the engine died with it

It should not: the engine is started in its own session and tracked by a
pidfile, so restarting or upgrading the panel leaves a loaded model alone. If
you see otherwise, that is a bug worth reporting.
