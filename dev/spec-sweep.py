#!/usr/bin/env python3
"""Sweep llama.cpp's speculation backends on one model, several workloads.

Each config needs its own server, so the engine is restarted per config. Every
generation uses a unique prefix and cache_prompt=false, so a warm prefix cannot
flatter a later run. The GPU is checked idle before the first start.

Sampling follows the runbook: SAMPLES (7) timed generations per prompt, with one
discarded warm-up after each start, because three samples once made MTP read as
no gain at all when it was 1.32x.

The `long-copy` prompt exists because the other two do not test prompt-lookup
drafting at all. ngram-cache drafts from n-grams already present in the context,
so it can only pay when the output repeats a long stretch of the input. A
seven-line function cannot show that; a few hundred lines being copied back with
one identifier changed is the regime it is built for.

    dev/spec-sweep.py MODEL_GGUF [CHAT_TEMPLATE] [ONLY,CONFIGS]

This is a development tool, not part of the installed package: it starts and
stops a server per config, which is the opposite of what the engine lifecycle
promises. It exists in the repository because every `verified: true` figure in
the catalogue is only as good as the instrument that produced it.

It stays off the engine's port and kills only the servers it started, by pid.
It also refuses to run while the engine is up, because they would be competing
for the same card and the loser is whichever numbers you were about to believe.

Environment:

- ``SWEEP_BUILD``         llama.cpp tag to sweep (default the pinned one)
- ``SWEEP_LLAMA_DIR``     an engine directory, for a build with no tag
- ``SWEEP_SAMPLES``       timed generations per prompt per config (default 7)
- ``SWEEP_LOGDIR``        where per-config server logs land (default /tmp)
- ``SWEEP_DRAFT``         path to a separate drafter GGUF; enables `dflash`
- ``SWEEP_NMAX``          draft width, every draft model (default llama.cpp's 3)

Not every config applies to every model. `draft-mtp` needs a checkpoint with
MTP layers, and `dflash` needs a drafter published for that exact target — so a
config that cannot start is reported and skipped rather than aborting the run.

Two result sets are only comparable if they came from the same binary on the
same backend, so the header records both -- build number, and the device line
that says whether this is Vulkan or CUDA and which matrix cores it found. A
table of tokens per second that does not say what produced it is how the last
set of numbers became untrustworthy.

It never sweeps the *installed* engine. Deciding whether to move the pin means
running a candidate against the incumbent, so both are fetched to their own
directories by `lllm3090 fetch-engine` and the install is left to serve.
"""
import json, os, signal, subprocess, sys, time, urllib.error, urllib.request, uuid

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from lllm3090 import config, engines  # noqa: E402

#: Which engine to sweep, by tag. SWEEP_LLAMA_DIR is the escape hatch for a
#: build that has no tag to name it by -- one compiled here, which is how a
#: CUDA engine will have to arrive.
BUILD = os.environ.get("SWEEP_BUILD", engines.LLAMA_BUILD)
D = os.path.expanduser(
    os.environ.get("SWEEP_LLAMA_DIR", str(engines.bench_dir(BUILD))))
if not os.path.exists(f"{D}/llama-server"):
    sys.exit(f"no engine at {D} -- run: lllm3090 fetch-engine --build {BUILD}")
MODEL = sys.argv[1]
TPL = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None
#: Not the engine's port. Sharing it meant the sweep had to clear whatever held
#: it, which is how a benchmark came to kill the engine the panel was serving --
#: the exact thing fetching builds separately was meant to prevent.
PORT = config.ENGINE_PORT + 2
SAMPLES = int(os.environ.get("SWEEP_SAMPLES", "7"))
if SAMPLES < 1:
    sys.exit(f"SWEEP_SAMPLES must be at least 1, not {SAMPLES}")
LOGDIR = os.environ.get("SWEEP_LOGDIR", "/tmp")
# Made here rather than at the first open, which is after the first server has
# already been started and is a half-hour into a run on the second config.
os.makedirs(LOGDIR, exist_ok=True)

#: A real source file, copied back nearly verbatim. Any few-hundred-line file
#: will do; this one is checked in and stable. Resolved against the repository
#: rather than $HOME, so the corpus travels with the script.
BIG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "lllm3090", "engine.py")

#: DFlash2 ships as a second GGUF -- a block-diffusion drafter published for one
#: target model -- so unlike the other configs it needs a file to point at, and
#: is skipped when there is none.
DRAFT = os.environ.get("SWEEP_DRAFT")

#: How many tokens a draft *model* proposes per verification step, set here
#: rather than left to each config, because leaving it meant comparing DFlash2
#: at its published 7 against MTP at llama.cpp's 3 -- two variables, reading as
#: a verdict on the drafter when the width was worth more than the choice of
#: drafter. See docs/explanations/going-faster.md.
#:
#: It reaches only the draft-model configs. The n-gram modes are sized by their
#: own `--spec-ngram-*` knobs, and which of those `ngram-cache` reads has not
#: been established here -- so it keeps its defaults, and the header says the
#: width covers the draft models rather than claiming to cover everything.
NMAX = os.environ.get("SWEEP_NMAX", "3")
WIDTH = ["--spec-draft-n-max", NMAX]

CONFIGS = [
    ("baseline", []),
    ("draft-mtp", ["--spec-type", "draft-mtp", *WIDTH]),
    ("ngram-cache", ["--spec-type", "ngram-cache"]),
    ("mtp+ngram", ["--spec-type", "draft-mtp,ngram-cache", *WIDTH]),
]

if DRAFT:
    CONFIGS.append(("dflash", [
        "--spec-type", "draft-dflash", "--spec-draft-model", DRAFT, *WIDTH,
    ]))

CODE = """def process(items):
    result = []
    for item in items:
        if item.get('active') and item.get('score', 0) > 10:
            result.append({'id': item['id'], 'score': item['score'] * 2})
    return sorted(result, key=lambda x: x['score'], reverse=True)"""

with open(BIG_FILE) as f:
    BIG = f.read()

# The trailing open fence matters: this is a raw completion, so ending mid-code
# means the very first predicted token is already part of the copy. Without it
# the model spends its budget on a preamble and the measurement is of prose.
LONG_COPY = (
    "Here is a Python module:\n\n```python\n" + BIG + "\n```\n\n"
    "Rewrite it, renaming the constant `TAIL_BYTES` to `MAX_TAIL_BYTES` everywhere "
    "it appears and changing nothing else whatsoever. Reproduce the complete "
    "file exactly, including all comments and docstrings.\n\n```python\n"
)

#: prompt -> (text, n_predict). The copy case needs a long budget: at 250
#: tokens it would still be inside the module docstring.
PROMPTS = {
    "prose": (
        "Write a Python function that merges two sorted lists, then explain how "
        "it works in detail.", 250),
    "code-edit": (
        "Here is a function:\n\n" + CODE +
        "\n\nRewrite it to add type hints and a docstring, keeping the logic "
        "identical. Output the full function.", 250),
    "long-copy": (LONG_COPY, 900),
}


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def gen(text, n_predict):
    """One cold generation. The UUID defeats the prefix cache by itself, but
    cache_prompt=false is kept so a build that ignores it is still honest."""
    return post("/completion", {
        "prompt": f"[{uuid.uuid4()}] {text}",
        "n_predict": n_predict, "temperature": 0, "cache_prompt": False,
    })


def up(proc):
    """Wait for this server to answer, or for it to die trying.

    The exit check is the point: a speculation config the build does not support
    exits in under a second, and polling /health alone would then wait out the
    full ten minutes before reporting it as skipped.
    """
    for _ in range(300):
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            return True
        except Exception:
            time.sleep(2)
    return False


def stop(proc):
    """Terminate this server, by pid, and wait for it to actually go.

    Never `pkill -f`: the pattern would match any llama-server holding the same
    checkpoint, including the installed engine serving it to somebody. Waiting
    matters because the next config cannot load 16 GB until this one has
    released it, and a fixed sleep is a guess about when that happened.
    """
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def gpu():
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def served():
    """Which checkpoint is actually answering on this port.

    The engine ignores the `model` field in a request and serves whatever is
    loaded, so a run is only attributable if this is checked either side. It has
    to be `/props`: `/v1/models` reports the `--alias`, which this script sets
    to a constant, so it answered `b` for every config and confirmed nothing.
    """
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/props", timeout=10) as r:
            path = json.load(r).get("model_path", "?")
    except Exception as e:
        return f"?({e})"
    return os.path.basename(path) + ("" if path == MODEL else f"  !! not {MODEL}")


def probe(*flags):
    """Ask the binary about itself. LD_LIBRARY_PATH matters: the shared ggml
    libraries sit beside the binary, not on the system path."""
    try:
        r = subprocess.run([f"{D}/llama-server", *flags], capture_output=True,
                           text=True, timeout=60,
                           env=dict(os.environ, LD_LIBRARY_PATH=D))
        out = [ln.strip() for ln in (r.stdout + r.stderr).split("\n") if ln.strip()]
        return " | ".join(out[:4])
    except Exception as e:
        return f"?({e})"


print(f"model:   {MODEL}")
print(f"samples: {SAMPLES} timed + 1 discarded warm-up, per prompt per config")
print(f"drafter: {DRAFT}" if DRAFT
      else "drafter: none -- set SWEEP_DRAFT to include the dflash config")
print(f"n-max:   {NMAX} (every draft model, so a config is not also a width; "
      "the ngram modes keep their own defaults)")
print(f"engine:  {D}")
print(f"build:   {probe('--version')}")
# Names the backend -- 'Vulkan0' or 'CUDA0' -- and so distinguishes two runs
# that are otherwise the same table of numbers.
print(f"device:  {probe('--list-devices')}")
print(f"GPU before: {gpu()}\n", flush=True)

# Two engines on one card is not a measurement. The installed one is somebody's
# session; this one is a benchmark, and the benchmark is the one that yields.
try:
    urllib.request.urlopen(f"{config.ENGINE_URL}/health", timeout=3)
    sys.exit(f"the engine is up on {config.ENGINE_URL} -- run 'lllm3090 stop' "
             "first; a sweep sharing the card measures the contention")
except urllib.error.URLError:
    pass

results = {}
for label, extra in CONFIGS:
    if ONLY and label not in ONLY: continue
    argv = [f"{D}/llama-server", "--model", MODEL, "--alias", "b",
            "--host", "127.0.0.1", "--port", str(PORT), "--n-gpu-layers", "999",
            "--ctx-size", "40960", "--parallel", "1", "-fa", "on",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--jinja"]
    if TPL:
        argv += ["--chat-template-file", TPL]
    argv += extra
    log = open(f"{LOGDIR}/sw-{label}.log", "wb")
    proc = subprocess.Popen(argv, stdout=log, stderr=log,
                            env=dict(os.environ, LD_LIBRARY_PATH=D),
                            start_new_session=True)
    if not up(proc):
        print(f"  {label}: FAILED TO START -- see {LOGDIR}/sw-{label}.log",
              flush=True)
        stop(proc)
        continue
    if not results:
        # Only the load log names the matrix-core extension in use, and coopmat1
        # against coopmat2 was 2.2x against 4.4x on prefill. Printed once.
        with open(f"{LOGDIR}/sw-{label}.log", errors="replace") as f:
            for line in f:
                if "matrix cores" in line or "Device 0:" in line:
                    print(f"cores:   {line.strip()}", flush=True)
                    break
    print(f"--- {label} ---   serving: {served()}", flush=True)
    try:
        # Warm-up: graph build and upload land here, never on a timed sample.
        gen(PROMPTS["prose"][0], 32)
        for pname, (ptext, npred) in PROMPTS.items():
            rates, drafted, accepted = [], 0, 0
            for _ in range(SAMPLES):
                r = gen(ptext, npred)
                t = r["timings"]
                rates.append(t["predicted_per_second"])
                drafted += t.get("draft_n", 0) or 0
                accepted += t.get("draft_n_accepted", 0) or 0
            rates.sort()
            med = rates[len(rates) // 2]
            results[(label, pname)] = med
            base = results.get(("baseline", pname))
            gain = f"  ({med / base:.2f}x)" if base and label != "baseline" else ""
            acc = f"  accept {accepted / drafted:.0%}" if drafted else ""
            print(f"    {pname:<10} med {med:6.1f} tok/s  "
                  f"[{rates[0]:.1f} .. {rates[-1]:.1f}]{gain}{acc}", flush=True)
        print(f"    served after: {served()}", flush=True)
    finally:
        # In a finally because a generation that raises would otherwise leave
        # 16 GB on the card and the next config unable to load.
        stop(proc)

print(f"\nGPU after: {gpu()}")
