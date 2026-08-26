# Drive the panel from a text console

```bash
lllm3090 tui
```

The same things the browser panel shows — engine state, VRAM, one list of
every model with what fits and what it will do, downloads with progress, and
the tail of the engine log — drawn in the terminal.

It is for the case where the panel is running and nothing can reach it: a
machine sitting on `multi-user.target` with no compositor, an SSH session
without a tunnel, a console on Ctrl-Alt-F3. Dropping the desktop is worth real
context on a 24 GB card, and it is also what takes the browser away.

## Keys

| key | what it does |
|---|---|
| `↑` `↓`, `k` `j` | move the cursor |
| `PgUp` `PgDn` | move it five at a time |
| `s` | start the model under the cursor |
| `x` | stop the engine and free the VRAM |
| `d` | download the model under the cursor |
| `c` | cancel that download |
| `P` | start the panel, if it is not running |
| `q` | quit — the engine keeps running |

There is one list and one cursor, so a key means one thing. `s` on a row that
is not downloaded says to press `d` instead rather than doing nothing; `d` on
one already on disk says so.

Quitting stops nothing. The engine is a detached process with a pidfile, the
panel is a user service, and downloads belong to the panel — closing the UI
leaves all three exactly as they were.

## What it does when the panel is not running

Most of it still works, because most of it has a local answer. The catalogue
is arithmetic over the card's memory, the engine is a pidfile and an HTTP
probe, and the log is a file. Status, the model list, start and stop all work
with nothing listening on 8080.

Downloading does not, and the footer says so. A download is a thread writing to
a `.part` file, and the panel restarts any part file it finds when it starts —
so a second process fetching the same file would give it two writers and
produce a corrupt GGUF rather than a slow one. Press `P` and the terminal UI
starts the panel's user service for you; downloads work from the next refresh.

To drive a panel somewhere other than the default port:

```bash
lllm3090 tui --url http://127.0.0.1:8081
```

Only loopback is worth pointing it at. The panel's endpoints start processes
with no authentication, which is why it binds 127.0.0.1 — see
[](remote-access.md) for reaching another machine's panel through an SSH
tunnel, and then run `lllm3090 tui` on the far end of it.

## What it looks like

```
lllm3090 0.4.1.dev10+g092fad8a8                  NVIDIA GeForce RTX 3090 24 GB
------------------------------------------------------------------------------
engine   stopped  -                                      http://127.0.0.1:1919
VRAM     [###.............................] 2.0 / 24.0 GiB
*MODELS ----------------------------------------------------------------------
 >+ Qwen3.8-27B         15.4G dense        101k x2  ~35 tok/s          on disk
  + gpt-oss-20b         12.1G moe          128k x4  ~160 tok/s         on disk
  + Qwen3-8B             5.0G dense         32k x4  ~115 tok/s         on disk
    Qwen3.6-35B-A3B     17.7G moe          212k x2           [#######.]  85.6%
    Qwen3.6-35B-A3B-Q4KS  20.9G moe           61k x2  ~124 tok/s
    Gemma-4-26B-A4B     18.2G moe+vis      138k x2  ~128 tok/s
    Muse-Glimmer-30B    17.9G dense+vis    128k x3  ~44 tok/s
    Gemma-4-12B-QAT      6.9G dense+vis    256k x4  ~84 tok/s
 ENGINE LOG ------------------------------------------------------------------
  main: server is listening on http://127.0.0.1:1919 - starting the main loop
* running  + on disk       /home/giles/models
s start  x stop  d download  c cancel  P panel  q quit
```

One list, not two. Every model appears once, whether it is on this disk or
merely available, and the column between the cursor and the name says which:
`*` is the model the card is serving, `+` is one already downloaded, and a
blank is one you would have to fetch first. On-disk rows come first, so the
models you can start without waiting are at the top; a finished download moves
its row on the next refresh rather than under your fingers. The right-hand
field says the one thing that is true of the row — its download's progress,
`running`, `on disk`, or `too big` — and the message line spells the markers
out whenever it has nothing more urgent to say.

Every character it draws is ASCII. The Linux framebuffer console renders
whatever glyphs its font holds and the default fonts are Latin-1, so box
drawing, block elements and the web panel's typographic dashes are not reliably
available. A UI whose whole purpose is to work where a browser cannot is not
the place to gamble on a font.

It needs 60 columns by 18 rows. Below that it says so rather than drawing a
layout that half fits — an 80×24 console is comfortable, and a wider window
spends the room on the speed qualifiers and longer log lines.

## What it will not tell you

A speed is a measurement taken on one card. If the GPU in this machine is not
the one the catalogue's figures came from, every speed is labelled
`(other card)`, and in a window too narrow for that label the speed is dropped
rather than truncated into a bare number. An entry nobody has benchmarked says
`speed not measured`, which is the honest answer until `lllm3090 bench` gives
it a real one.
