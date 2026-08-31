"""The control panel on a text console, for a machine with no browser on it.

The panel binds loopback and speaks HTML, which is exactly right until the
machine is sitting on ``multi-user.target`` with no compositor -- the state
that hands a 24 GB card back most of a 35B model's KV cache. There is then no
browser within reach of ``127.0.0.1:8080`` and no way to start a model except
by remembering its name.

This module is that panel drawn with ``curses``. It shows the same things:
engine state, VRAM, one list of every model -- what is on disk and what is
merely available, with what fits and what it will do -- downloads with
progress, and the tail of the engine log.

Two decisions are worth stating, because both could reasonably have gone the
other way.

**It talks to the panel over HTTP, and falls back to the library.** Reading the
machine's state has a local answer that is always correct -- the catalogue is
arithmetic, the engine is a pidfile -- so status, the model list and start/stop
work with the panel stopped. Downloads do not divide so cleanly: the registry
of in-flight downloads is state inside the panel process, and a second process
fetching into the same ``.part`` file would append to bytes the first is still
writing. ``downloads.resume_interrupted`` would then cheerfully start a third.
So downloads are the panel's job alone, and when it is not running the terminal
UI says so and offers to start it.

**Everything it draws is ASCII.** The Linux framebuffer console renders
whatever glyphs its font holds, and the default fonts are Latin-1: box-drawing
characters, block elements and the panel's typographic dashes are not reliably
there. A UI whose entire purpose is to work where a browser cannot is not the
place to gamble on a font.

``curses`` is in the standard library, so this costs the project no dependency
at all -- which matters for something installed with ``uv tool install``. It is
imported inside :func:`run` rather than at module scope so that importing this
module, as the documentation build does, needs no terminal and no ncurses.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from . import catalog, config, engine, state

#: Below this there is no useful layout left, only a lie about one.
MIN_WIDTH = 60
MIN_HEIGHT = 18

#: How often the machine is re-read. The engine log is a local file and the
#: catalogue is arithmetic; the expensive call is ``nvidia-smi``, which is why
#: this happens on a worker thread and not in the keyboard loop.
REFRESH_SECONDS = 1.5

#: What each key does, in the order the footer lists them.
KEYS: tuple[tuple[str, str], ...] = (
    ("s", "start"),
    ("x", "stop"),
    ("d", "download"),
    ("c", "cancel"),
    ("P", "panel"),
    ("q", "quit"),
)


@dataclass
class Line:
    """One rendered row: the text, and which of the styles to draw it in."""

    text: str
    style: str = ""


@dataclass
class Ui:
    """What the user is looking at -- everything the snapshot does not say."""

    row: int = 0
    message: str = ""
    busy: str = ""

    def clamp(self, snap: dict[str, Any]) -> None:
        """Pull the cursor back inside a list that may have shrunk.

        The poller replaces the snapshot on its own thread, so a model deleted
        from disk -- or a catalogue that has not loaded yet -- can leave the
        cursor pointing past the end of the list it was placed in. Every read
        of that cursor is then an ``IndexError``, and an exception out of
        ``handle_key`` closes the UI on a keypress.
        """
        self.row = min(max(0, self.row), max(0, len(model_rows(snap)) - 1))


# ---------------------------------------------------------------------------
# Talking to the machine
# ---------------------------------------------------------------------------


class Control:
    """The panel if it is up, this process if it is not.

    ``reachable`` records which of the two answered last, because it changes
    what the UI may offer: without a panel there is no download.
    """

    def __init__(self, url: str | None = None) -> None:
        self.url = (url or config.PANEL_URL).rstrip("/")
        self.reachable = False

    def _call(self, path: str, method: str = "GET", timeout: float = 5.0) -> Any:
        req = urllib.request.Request(self.url + path, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
        return json.loads(body) if body else None

    def _post(self, path: str, timeout: float = 90.0) -> str:
        """POST to the panel and reduce whatever comes back to one line."""
        try:
            result = self._call(path, method="POST", timeout=timeout)
        except urllib.error.HTTPError as e:
            try:
                return str(json.loads(e.read()).get("error", e))
            except Exception:
                return f"panel returned HTTP {e.code}"
        except Exception as e:
            self.reachable = False
            return f"panel unreachable: {e}"
        if isinstance(result, dict):
            return str(result.get("detail") or result.get("error") or result)
        return str(result)

    def snapshot(self) -> dict[str, Any]:
        """The machine, from the panel if it answers and from here if not.

        The log is always read from the file. It is on this machine either way,
        and re-reading its last lines is cheaper and simpler than an SSE client
        -- the panel streams it because a browser cannot tail a file.
        """
        try:
            snap = self._call("/api/status", timeout=8)
            self.reachable = True
            snap["source"] = "panel"
        except Exception:
            self.reachable = False
            snap = state.snapshot()
            snap["busy"] = False
            snap["last"] = {"action": None, "ok": None, "detail": ""}
            snap["source"] = "local"
        snap["log"] = engine.tail(200)
        return snap

    def start(self, model: str) -> str:
        if self.reachable:
            return self._post(f"/api/start?model={urllib.parse.quote(model)}")
        entry = next((m for m in catalog.installed() if m["name"] == model), None)
        if entry is None:
            return f"{model} is not installed"
        known = next((m for m in catalog.load_catalog() if m.name == model), None)
        p = catalog.launch_plan(model)
        engine.stop()
        # After the stop, so the outgoing engine's VRAM is not charged against
        # its own replacement. This path starts the same engine on the same
        # card as the console and the panel, so it owes the same warning.
        warning = catalog.free_vram_warning(known, p.pool)
        # wait=0, exactly as the panel does it: a load takes minutes, and a UI
        # that stops repainting for the duration looks like a UI that has hung.
        _, detail = engine.start(
            entry["path"], model, p.pool, p.parallel, 0,
            known.chat_template if known else None, entry.get("mmproj"),
        )
        return f"{warning}\n{detail}" if warning else detail

    def stop(self) -> str:
        if self.reachable:
            return self._post("/api/stop")
        return engine.stop()[1]

    def download(self, model_id: str) -> str:
        """Ask the panel to fetch a model. There is no local fallback.

        A download is a thread writing to a ``.part`` file, and the panel
        resumes any ``.part`` it finds when it starts. Downloading from here as
        well would give one file two writers, which is not a slow download but
        a corrupt one.
        """
        if not self.reachable:
            return "downloads need the panel -- press P to start it"
        return self._post(f"/api/download/{model_id}", timeout=15)

    def cancel(self, model_id: str) -> str:
        if not self.reachable:
            return "downloads need the panel -- press P to start it"
        return self._post(f"/api/download/{model_id}/cancel", timeout=15)

    def start_service(self) -> str:
        """Start the panel's user unit, so downloads become possible."""
        if self.reachable:
            return "the panel is already running"
        if not shutil.which("systemctl"):
            return "no systemctl here; run 'lllm3090 panel' in another console"
        result = subprocess.run(
            ["systemctl", "--user", "start", "lllm3090-panel.service"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            return result.stderr.strip() or "systemctl refused to start the panel"
        return f"panel starting on {self.url}"


# ---------------------------------------------------------------------------
# Rendering: pure functions from a snapshot to lines of text
# ---------------------------------------------------------------------------


def bar(fraction: float, width: int) -> str:
    """An ASCII progress bar. ``fraction`` outside 0..1 is clamped, not trusted."""
    width = max(2, width)
    inner = width - 2
    filled = round(min(max(fraction, 0.0), 1.0) * inner)
    return "[" + "#" * filled + "." * (inner - filled) + "]"


def fmt_ctx(tokens: int) -> str:
    return f"{round(tokens / 1024)}k" if tokens >= 1024 else str(tokens)


def speed_label(row: dict[str, Any], brief: bool = False) -> str:
    """What may be claimed about a model's speed on the card in this machine.

    An entry nobody has benchmarked has no number, and no number is the honest
    thing to print: the browser panel once rendered such an entry as
    ``~null tok/s``, which reads as a broken panel rather than as an absence.
    A number measured on another card, or on another backend, is shown as
    measured elsewhere, never scaled to this one.

    ``brief`` is for a window with no room for the whole qualifier. It drops
    only ``(measured)``, which says the ordinary thing; a figure that does not
    describe this machine never loses the words that say so, because without
    them it reads as a measurement of it. What those words are is decided in
    :func:`lllm3090.catalog.speed_qualifier` -- the card and the backend are
    two independent reasons and they do not say the same thing.
    """
    if row.get("expected_tok_s") is None:
        return "speed not measured"
    label = f"~{row['expected_tok_s']} tok/s"
    if row.get("verified") and not (row.get("speed_applies") and brief):
        # The fallback follows `speed_applies`, and that matters: this can be
        # rendering a row fetched over HTTP from an older panel that sends no
        # `speed_note`, and defaulting to "measured" there would assert the
        # figure describes this machine at exactly the moment it does not.
        default = "measured" if row.get("speed_applies") else "other card"
        label += f" ({row.get('speed_note') or default})"
    return label


def window(count: int, index: int, rows: int) -> tuple[int, int]:
    """The slice of a list to draw so that ``index`` is inside it."""
    if rows <= 0 or count <= 0:
        return 0, 0
    start = min(max(0, index - rows // 2), max(0, count - rows))
    return start, min(count, start + rows)


def _columns(left: str, right: str, width: int) -> str:
    """``left`` flush left and ``right`` flush right, truncating the left."""
    if not right:
        return left[:width]
    room = width - len(right) - 1
    if room < 8:
        return (left + " " + right)[:width]
    return f"{left[:room]:<{room}} {right}"


def _header(snap: dict[str, Any], width: int) -> list[Line]:
    card = snap["card"]
    if not card["present"]:
        right = f"no GPU -- figures are a {card['reference']}'s"
    else:
        right = f"{card['name']} {card['vram_gb']} GB"
        if not card["measured"]:
            right += " (speeds from elsewhere)"
    return [
        Line(_columns(f"lllm3090 {snap['version']}", right, width), "head"),
        Line("-" * width, "dim"),
    ]


def _engine_lines(snap: dict[str, Any], width: int) -> list[Line]:
    e = snap["engine"]
    if not e["running"]:
        word, style = "stopped", "bad"
    elif e["answering"]:
        word, style = "running", "ok"
    else:
        word, style = "loading", "warn"
    model = e["model"] or ("loading..." if e["running"] else "-")
    # An older panel does not report its endpoint; the local default is a
    # better answer than a blank, since both are on the same loopback.
    endpoint = snap.get("endpoint") or config.ENGINE_URL
    lines = [Line(_columns(f"engine   {word:<9}{model}", endpoint, width), style)]
    d = snap.get("disk")
    # The other way a model fails to arrive: room on the card is no use without
    # room on the disk to put the 21 GB first.
    free = f"{d['free_gb']:.0f} GB free" if d else ""
    v = snap.get("vram")
    if v and v.get("total_mb"):
        used, total = v["used_mb"], v["total_mb"]
        figure = f"{used / 1024:.1f} / {total / 1024:.1f} GiB"
        # The bar gives back what the right-hand figure needs.
        span = min(34, max(10, width - 32 - (len(free) + 2 if free else 0)))
        lines.append(Line(_columns(
            f"VRAM     {bar(used / total, span)} {figure}", free, width
        )))
    else:
        lines.append(Line(_columns(
            "VRAM     not reported (no nvidia-smi on this machine)", free, width
        ), "dim"))
    return lines


def model_rows(snap: dict[str, Any]) -> list[dict[str, Any]]:
    """One list: the catalogue, plus any GGUF on disk it has never heard of.

    The same eight models used to appear twice -- once as INSTALLED and once as
    CATALOGUE -- which left the reader to do the join and made ``s`` mean two
    slightly different things depending on which copy the cursor happened to be
    on. There is one row per model now, and what it offers depends on the state
    that row is in.

    On-disk rows come first and catalogue order is kept inside each group, so
    the models that start without waiting are at the top. The sort is stable
    and reads only the snapshot, which means a finished download moves its row
    once, when the poller hands over a new one, and not under a held ``j``.
    """
    served = snap["engine"]["model"] if snap["engine"]["running"] else None
    active = {d["id"]: d for d in snap["downloads"]}
    rows = []
    for c in snap["catalog"]:
        d = active.get(c["id"])
        fetching = d and d["state"] in {"queued", "downloading"}
        rows.append(
            {
                **c,
                "on_disk": c["installed"],
                "running": c["name"] == served,
                "download": d if fetching else None,
                "failed": bool(d and d["state"] == "error"),
            }
        )
    catalogued = {c["name"] for c in snap["catalog"]}
    for m in snap["installed"]:
        if m["name"] in catalogued:
            continue
        # A GGUF nobody curated is still something you can start, so it belongs
        # in the list. Its window came from the snapshot, which got it from
        # launch_plan -- the same call the start itself will make.
        rows.append(
            {
                "id": None,
                "name": m["name"],
                "gb": m["gb"],
                "kind": m.get("kind", "gguf"),
                "vision": False,
                "params": "",
                "notes": "",
                "fits": m.get("fits", True),
                # Nothing is known about an uncatalogued GGUF's cache cost, so
                # there is no arithmetic behind a caution and none is shown.
                "status": catalog.STATUS_OK,
                "status_note": "",
                "max_ctx": m.get("max_ctx", catalog.UNKNOWN_MODEL_CTX),
                "parallel": m.get("parallel", config.DEFAULT_PARALLEL),
                "expected_tok_s": None,
                "verified": False,
                "speed_applies": True,
                "installed": True,
                "on_disk": True,
                "running": m["name"] == served,
                "download": None,
                "failed": False,
            }
        )
    rows.sort(key=lambda r: 0 if r["on_disk"] else 1)
    return rows


def _row_lines(
    rows: list[dict[str, Any]], ui: Ui, width: int, height: int
) -> list[Line]:
    """The merged list, scrolled so the cursor is on screen.

    The marker between the cursor and the name is the whole of what used to be
    two panes: ``*`` is the model this card is serving, ``+`` is one already on
    disk, and a space is one you would have to fetch first.
    """
    if not rows:
        return [Line("  nothing to show -- the catalogue did not load", "dim")][:height]
    out = []
    start, end = window(len(rows), ui.row, height)
    for i in range(start, end):
        c = rows[i]
        chosen = i == ui.row
        mark = "*" if c["running"] else ("+" if c["on_disk"] else " ")
        kind = c["kind"] + ("+vis" if c["vision"] else "")
        ctx = f"{fmt_ctx(c['max_ctx'])} x{c['parallel']}" if c["fits"] else "-"
        name_width = 24 if width >= 96 else 18
        base = (
            f" {'>' if chosen else ' '}{mark} {c['name']:<{name_width}} "
            f"{c['gb']:>5.1f}G {kind:<9} {ctx:>10}"
        )
        # The cursor keeps its highlight whatever the row's own state is: a
        # reversed line that also went red would say less, not more.
        style = "sel" if chosen else ""
        d = c["download"]
        if d:
            right = f"{bar(d['percent'] / 100, 10)} {d['percent']:>5.1f}%"
        elif c["failed"]:
            right, style = "failed", style or "bad"
        elif c["running"]:
            right, style = "running", style or "ok"
        # The card's verdict outranks "on disk", which would otherwise hide it:
        # being downloaded says nothing about whether this card can use it, and
        # on a box where everything is downloaded that branch swallowed every
        # caution. Nothing is lost by preferring the verdict -- the row's `mark`
        # already carries disk state.
        elif c.get("status") == catalog.STATUS_CAPABILITY:
            right, style = "old GPU", style or "dim"
        elif not c["fits"]:
            right, style = "too big", style or "dim"
        elif c.get("status") == catalog.STATUS_TIGHT:
            # Startable and fine for chat, so not dimmed like a refusal -- but
            # it cannot hold an agent's system prompt and the list must say so.
            right, style = "tight", style or "warn"
        elif c["on_disk"]:
            right = "on disk"
        else:
            right = ""
        # The speed goes in whole or not at all. Half of "~115 tok/s (other
        # card)" is a figure with its provenance cut off, which is precisely
        # the claim this project does not make.
        speed = speed_label(c, brief=width < 96)
        room = width - (len(right) + 1 if right else 0)
        left = base if len(base) + 2 + len(speed) > room else f"{base}  {speed}"
        out.append(Line(_columns(left, right, width), style))
    return out


def _log_lines(snap: dict[str, Any], width: int, rows: int) -> list[Line]:
    lines = snap.get("log") or []
    if not lines:
        return [Line("  (the engine log is empty)", "dim")][:rows]
    out = []
    for text in lines[-rows:]:
        low = text.lower()
        if "error" in low or "failed" in low:
            style = "bad"
        elif "warn" in low:
            style = "warn"
        else:
            style = "dim"
        out.append(Line(("  " + text)[:width], style))
    return out


def footer(snap: dict[str, Any], ui: Ui, width: int) -> list[Line]:
    """The message line and the key hints -- the only help this UI has room for."""
    last = snap.get("last") or {}
    if ui.busy:
        note, style = f"{ui.busy}...", "warn"
    elif ui.message:
        note, style = ui.message, "warn"
    elif snap.get("source") == "local":
        note = "no panel: status and start/stop are local, downloads need it (P)"
        style = "warn"
    elif last.get("action"):
        # What the panel last did, whoever asked it to. A start that failed
        # says why here, whether it was this UI or a browser that pressed it.
        verdict = "ok" if last.get("ok") else "FAILED"
        note = f"{verdict}: {last['action']} -- {last.get('detail', '')}"
        style = "dim" if last.get("ok") else "bad"
    else:
        # Nothing has happened yet, so the line explains the marker column --
        # the only thing on screen with no words of its own.
        note = f"* running  + on disk       {snap.get('models_dir', '')}"
        style = "dim"
    keys = "  ".join(f"{key} {what}" for key, what in KEYS)
    return [
        Line(note[:width], style),
        Line(keys[:width], "head"),
    ]


def render(snap: dict[str, Any], ui: Ui, width: int, height: int) -> list[Line]:
    """The whole screen, as at most ``height`` lines of at most ``width``.

    A curses window raises rather than clipping when something is written past
    its last column, so every line this returns is already inside the window.
    """
    ui.clamp(snap)
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        small = [
            f"window is {width}x{height}",
            f"lllm3090 tui needs {MIN_WIDTH}x{MIN_HEIGHT}",
            "resize, or q to quit",
        ]
        return [Line(text[: max(width, 0)], "warn") for text in small][
            : max(height, 0)
        ]

    # Two lists share what the fixed furniture leaves: two headers, the engine
    # block, a section rule each, and the two footer lines.
    spare = height - 7
    log_rows = max(3, spare // 3)
    list_rows = max(1, spare - log_rows)

    lines = _header(snap, width) + _engine_lines(snap, width)
    # Always focused: with one list there is nothing to switch to.
    lines.append(Line(_section("MODELS", True, width), "head"))
    lines += _row_lines(model_rows(snap), ui, width, list_rows)
    lines.append(Line(_section("ENGINE LOG", False, width), "head"))
    lines += _log_lines(snap, width, log_rows)
    lines += footer(snap, ui, width)
    return [Line(line.text[:width], line.style) for line in lines[:height]]


def _section(title: str, focused: bool, width: int) -> str:
    mark = "*" if focused else " "
    return f"{mark}{title} {'-' * max(0, width - len(title) - 2)}"[:width]


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def selected_row(snap: dict[str, Any], ui: Ui) -> dict[str, Any] | None:
    """The row the cursor is on, or None when there is no list to be on."""
    rows = model_rows(snap)
    return rows[ui.row] if rows else None


def handle_key(
    key: str, snap: dict[str, Any], ui: Ui, control: Control, submit
) -> bool:
    """Act on one keypress. Returns False when the user asked to quit.

    ``submit(label, action)`` runs ``action`` somewhere that is not the drawing
    loop: stopping an engine waits for the VRAM to come back, which is seconds,
    and a UI that stops repainting for seconds looks broken.
    """
    ui.clamp(snap)
    if key in {"q", "escape"}:
        return False
    ui.message = ""
    rows = model_rows(snap)
    row = rows[ui.row] if rows else None

    def move(delta: int) -> None:
        ui.row = min(max(0, ui.row + delta), max(0, len(rows) - 1))

    if key in {"down", "j"}:
        move(1)
    elif key in {"up", "k"}:
        move(-1)
    elif key == "npage":
        move(5)
    elif key == "ppage":
        move(-5)
    elif key == "s":
        if row is None:
            ui.message = "there is nothing here to start"
        elif not row["on_disk"]:
            # The row is in the list because every model is; that is not the
            # same as being able to start it.
            ui.message = f"{row['name']} is not downloaded -- press d"
        elif row["running"]:
            ui.message = f"{row['name']} is already running"
        else:
            submit(f"start {row['name']}", lambda: control.start(row["name"]))
    elif key == "x":
        if not snap["engine"]["running"]:
            ui.message = "the engine is not running"
        else:
            submit("stop the engine", control.stop)
    elif key == "d":
        if row is None:
            ui.message = "there is nothing here to download"
        elif row["on_disk"]:
            ui.message = f"{row['name']} is already downloaded"
        elif not row["fits"]:
            # Showing an entry that does not fit is not offering it: the panel
            # answers such a download with a 400, and 40 GB is an expensive way
            # to find that out.
            ui.message = f"{row['name']} does not fit this card"
        elif row.get("status") == catalog.STATUS_CAPABILITY:
            ui.message = row.get("status_note") or f"{row['name']} needs a newer GPU"
        elif snap.get("source") == "local":
            ui.message = "downloads need the panel -- press P to start it"
        else:
            submit(f"download {row['name']}", lambda: control.download(row["id"]))
    elif key == "c":
        if row is None or not row["download"]:
            ui.message = "nothing is downloading here"
        else:
            submit(f"cancel {row['name']}", lambda: control.cancel(row["id"]))
    elif key == "P":
        submit("start the panel", control.start_service)
    return True


# ---------------------------------------------------------------------------
# The curses driver
# ---------------------------------------------------------------------------


@dataclass
class _Shared:
    """What the poller and the keyboard loop pass between them."""

    snap: dict[str, Any] | None = None
    wake: threading.Event = field(default_factory=threading.Event)
    stop: threading.Event = field(default_factory=threading.Event)


def _poll(control: Control, shared: _Shared) -> None:
    while not shared.stop.is_set():
        try:
            shared.snap = control.snapshot()
        except Exception as e:  # a UI that dies on a bad read is worse than a stale one
            if shared.snap is None:
                shared.snap = state.snapshot() | {"source": "local", "log": [str(e)]}
        shared.wake.wait(REFRESH_SECONDS)
        shared.wake.clear()


def _key_name(curses: Any, ch: int) -> str:
    """One keypress as the name :func:`handle_key` knows it."""
    named = {
        curses.KEY_UP: "up",
        curses.KEY_DOWN: "down",
        curses.KEY_NPAGE: "npage",
        curses.KEY_PPAGE: "ppage",
        27: "escape",
    }
    if ch in named:
        return named[ch]
    if 0 <= ch < 0x110000:
        return chr(ch)
    return ""


def _styles(curses: Any) -> dict[str, int]:
    if not curses.has_colors():
        return {
            "head": curses.A_BOLD, "sel": curses.A_REVERSE, "ok": curses.A_BOLD,
            "warn": curses.A_BOLD, "bad": curses.A_BOLD, "dim": curses.A_NORMAL,
        }
    curses.start_color()
    try:
        curses.use_default_colors()
        background = -1
    except curses.error:
        background = curses.COLOR_BLACK
    for index, colour in enumerate(
        (curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED,
         curses.COLOR_CYAN, curses.COLOR_BLUE), start=1
    ):
        curses.init_pair(index, colour, background)
    return {
        "ok": curses.color_pair(1),
        "warn": curses.color_pair(2),
        "bad": curses.color_pair(3),
        "head": curses.color_pair(4) | curses.A_BOLD,
        "dim": curses.A_DIM,
        "sel": curses.A_REVERSE,
    }


def _draw(stdscr: Any, curses: Any, lines: list[Line], styles: dict[str, int]) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    for y, line in enumerate(lines[:height]):
        # Writing to the very last cell of the window scrolls it and raises,
        # even though the text fits, so the bottom line stops one short.
        text = line.text[: width - 1] if y == height - 1 else line.text[:width]
        try:
            stdscr.addstr(y, 0, text, styles.get(line.style, 0))
        except curses.error:
            pass
    stdscr.noutrefresh()
    curses.doupdate()


def _loop(stdscr: Any, curses: Any, control: Control) -> int:
    curses.curs_set(0)
    stdscr.timeout(200)
    styles = _styles(curses)
    ui = Ui()
    shared = _Shared()
    poller = threading.Thread(target=_poll, args=(control, shared), daemon=True)
    poller.start()

    def submit(label: str, action) -> None:
        if ui.busy:
            ui.message = f"still {ui.busy}"
            return
        ui.busy = label

        def work() -> None:
            try:
                result = action()
            except Exception as e:
                result = str(e)
            ui.busy = ""
            ui.message = f"{label}: {result}"
            shared.wake.set()

        threading.Thread(target=work, daemon=True).start()

    try:
        while True:
            snap = shared.snap
            height, width = stdscr.getmaxyx()
            if snap is None:
                _draw(stdscr, curses, [Line("reading the machine...", "dim")], styles)
            else:
                _draw(stdscr, curses, render(snap, ui, width, height), styles)
            ch = stdscr.getch()
            if ch == -1 or ch == curses.KEY_RESIZE:
                continue
            key = _key_name(curses, ch)
            # Quit is answered before anything else, and without waiting for a
            # first snapshot: a UI you cannot leave while it is talking to
            # something slow is a UI that has trapped you.
            if key in {"q", "escape"}:
                break
            if snap is None or not key:
                continue
            if not handle_key(key, snap, ui, control, submit):
                break
            shared.wake.set()
    finally:
        shared.stop.set()
        shared.wake.set()
    return 0


def run(url: str | None = None) -> int:
    """Draw the panel in this terminal until the user quits."""
    import curses

    control = Control(url)
    return curses.wrapper(_loop, curses, control)
