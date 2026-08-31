# Changelog

What changed between releases, and why it mattered. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html), and the tag is the
only place a version number is written — `setuptools_scm` derives the rest.

Entries describe the effect on someone using the thing, not the diff. If a
change is only visible in the source it does not need a line here.

## [Unreleased]

### Added

- **Multi-token prediction is turned on by itself** when a checkpoint carries
  the head. The model drafts its own next tokens and verifies them in one pass;
  measured on the reference 3090, `Qwen3.8-27B` goes from 34.9 to 56.6 tok/s
  (1.62x) and `Qwen3.6-35B-A3B-MTP` from 130.5 to 171.8 (1.32x, and 179.9 on
  code editing). Nothing to configure: the checkpoint is read at start, and the
  flag is added only when the tensors are actually there — a catalogue field
  claiming MTP would turn a working start into a failed one the day a repo
  shipped a build with the head stripped.
- `Qwen3.6-35B-A3B-MTP` in the catalogue, and it is now the recommendation. It
  is the same model as before with the head preserved: same context on this
  card, 46 tok/s faster.

### Changed

- **One conversation is filled to the model's ceiling before a second slot is
  opened.** The pool is a fixed number of tokens, so splitting it does not
  create capacity — it halves the window and buys concurrency with it. Slots
  are now granted only in whole windows, where the architecture has run out
  before the card has and the spare cache could not have become a longer
  conversation anyway.

  On a 24 GB desktop that is worth a great deal: `Qwen3.6-35B-A3B` goes from
  169k x 2 to the full 256k, `Qwen3.8-27B` from 84k x 2 to 168k, and
  `Gemma-4-26B-A4B` from 103k x 2 to 207k. Nothing loses window.

  **If you run an agent, ask for two slots** — `lllm3090 start <model>
  --parallel 2`. A single slot has nowhere to admit a subagent, so the
  scheduler serialises them and each one evicts the parent's cached prefix.
  `lllm3090 claude` now says so when it finds a one-slot engine.
- `Qwen3.6-35B-A3B-Q4KS` is no longer flagged as tight on a 24 GB desktop. It
  was tight only because the pool was being split; one window gives it 69k.

## [0.6.0] — 2026-08-28

### Added

- `lllm3090 sweep` — survey the published GGUF models and price them against
  your card, without downloading any of them. Each candidate costs one
  `config.json`, from which its KV cost per token is derived and run through
  the same arithmetic the panel uses. `--yaml` emits catalogue entries ready to
  paste; `--gpu <profile>` prices against a card you do not own.
- A third state in every front end: **tight**. A model that fits the card and
  still leaves less context than an agent harness spends on its own system
  prompt is now shown as a caution rather than a success, with the size of card
  that would clear it. Two catalogue entries turn out to be in that state on a
  24 GB card with a desktop running, and were previously indistinguishable from
  the ones that work.
- `min_compute_capability`, an optional catalogue field for a model whose
  kernels only exist on newer silicon. It is the only hardware requirement
  typed by hand: the memory a model needs is computed from its own figures and
  recomputes for whatever card is present.

### Fixed

- Six catalogue entries asserted figures in their notes that the panel
  computes per card — three quoted a context window, three a slot count. They
  were correct when written and stale by the time the driver reserve and the KV
  overhead factor landed, so a row could say 34k in its context column and 61k
  in its prose with nothing to say which was current. The notes now carry
  judgement and leave every number to the arithmetic, and a test keeps it that
  way.
- `lllm3090 models` showed an installed model as `installed` whatever the card
  made of it, so a model that could not hold an agent's system prompt read as a
  success. Disk state and card verdict are now both shown.

## [0.5.0] — 2026-08-26

### Added

- A terminal UI (`lllm3090 tui`), so a machine with no browser — a text
  console, an SSH session — can still see the same panel and drive it with the
  same keys.
- Vision support: a model's projector is fetched alongside its weights and
  passed to the engine, so a multimodal checkpoint can actually see.
- Three more models in the catalogue, and scrolling for the list that holds
  them.
- `lllm3090 claude --print-env` prints the environment `lllm3090 claude` would
  launch with, as `export` lines and nothing else on stdout, so it can be
  `eval`'d for a harness this project has no command for.
- A **clear** button on the engine log.
- Free disk space beside the VRAM figure, in both front ends — the download is
  chosen there, and running out of disk is the other way a model fails to
  arrive.
- An explanation of what else is out there and when to use it instead
  (`docs/explanations/landscape.md`).
- `lllm3090 claude` tells Claude Code how many conversations the engine can
  actually hold, by asking it (`/props` → `total_slots`) and keeping one slot
  for the parent. Claude Code's own default is 20 concurrent subagents, which
  on a two-slot engine is a fan-out ten times wider than the card has room
  for — and llama.cpp queues the excess rather than refusing it, so it arrived
  as the model being slow.

### Changed

- One list of models instead of two. The panel and the terminal UI both showed
  the same models twice, once as *installed* and once as *available*; each row
  now carries the action that belongs to it.
- The catalogue is computed for the GPU actually in the machine rather than for
  a 3090, and speeds are never scaled to a card they were not measured on.
- Context is planned against the VRAM the driver hands out, not the VRAM the
  card is sold with.
- `lllm3090 claude` is given the window the engine was really started with. It
  had been recomputing it with desktop defaults, capping the agent 39k tokens
  below what the engine was serving on a text console.
- Recommends `Qwen3.6-35B-A3B` rather than `Qwen3.8-27B` — measurement put the
  previous recommendation last on speed.

### Fixed

- A malformed `TimeoutStopSec` line in the systemd unit, which systemd had been
  rejecting on **every** start. A second, valid directive further down meant
  the behaviour was right and only the journal knew.
- The engine is refused a file that is not a GGUF, before anything is launched,
  rather than after the engine gives up on it in a log nobody is reading.
- A start now checks the VRAM needed to *serve*, not just to load, so a model
  can no longer report itself healthy and then fail every request with
  `ErrorOutOfDeviceMemory`.
- Five ways the terminal UI and the panel misread their own inputs.

## [0.4.0] — 2026-08-25

### Added

- Measured decode speed for every model in the catalogue.

### Changed

- An agent whose window cannot hold its own system prompt is refused rather
  than launched into a session where the first message fails.

## [0.3.0] — 2026-08-25

### Fixed

- A panel restart no longer kills the running engine. systemd's default
  `KillMode=control-group` was taking the engine down with the panel on every
  upgrade, and the next request got "connection refused" with no clue why.
- Claude Code works against `Qwen3.8-27B`, which needed a patched chat template
  — its own rejects a system message after the first turn.
- The panel is restarted on upgrade rather than merely enabled: the running
  process is executing code that has been swapped out from under it, which
  fails per request without the process ever exiting.

## [0.2.0] — 2026-08-25

### Added

- `lllm3090 setup`, and installation with `uv`.
- Free slots are given away: a pool sized for two conversations hands its spare
  capacity to whichever slot is asking.

### Fixed

- The panel no longer collapses when a download is in flight.

## [0.1.0] — 2026-08-25

First release. The package, the installer, the control panel and the model
catalogue, with a KV pool sized for concurrent conversations rather than for
one session.

[Unreleased]: https://github.com/gilesknap/lllm3090/compare/0.6.0...HEAD
[0.6.0]: https://github.com/gilesknap/lllm3090/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/gilesknap/lllm3090/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/gilesknap/lllm3090/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/gilesknap/lllm3090/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/gilesknap/lllm3090/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/gilesknap/lllm3090/releases/tag/0.1.0
