"""The terminal UI, which has to be right without anyone looking at it.

Every check here is on a pure function: the screen is rendered to a list of
strings and asserted on, and the keyboard is driven through
``tui.handle_key``. Nothing in this file opens a terminal, because nothing in
the parts of the UI that can be wrong needs one -- what curses does with these
strings is a much smaller surface than what decides them.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from lllm3090 import catalog, config, downloads, engine, panel, state, tui

MODELS = [
    {"name": "Qwen3.6-35B-A3B", "path": "/m/Qwen3.6-35B-A3B/w.gguf",
     "mmproj": None, "gb": 17.7},
    {"name": "Qwen3-8B", "path": "/m/Qwen3-8B/w.gguf", "mmproj": None, "gb": 5.0},
]

CATALOG = [
    {"id": "qwen3.6-35b-a3b", "name": "Qwen3.6-35B-A3B", "gb": 17.7, "kind": "moe",
     "params": "35B", "vision": False, "fits": True, "max_ctx": 212992,
     "parallel": 2, "installed": True, "expected_tok_s": 126, "verified": True,
     "speed_applies": True, "notes": ""},
    {"id": "qwen3-8b", "name": "Qwen3-8B", "gb": 5.0, "kind": "dense",
     "params": "8B", "vision": False, "fits": True, "max_ctx": 32768,
     "parallel": 4, "installed": True, "expected_tok_s": 115, "verified": True,
     "speed_applies": False, "notes": ""},
    {"id": "muse-glimmer-30b", "name": "Muse-Glimmer-30B", "gb": 17.9,
     "kind": "dense", "params": "30B", "vision": True, "fits": True,
     "max_ctx": 131072, "parallel": 3, "installed": False,
     "expected_tok_s": None, "verified": False, "speed_applies": True,
     "notes": ""},
    {"id": "too-big", "name": "Something-70B", "gb": 40.0, "kind": "dense",
     "params": "70B", "vision": False, "fits": False, "max_ctx": 0,
     "parallel": 2, "installed": False, "expected_tok_s": None,
     "verified": False, "speed_applies": True, "notes": ""},
]


@pytest.fixture
def snap() -> dict:
    """A machine with an engine up, two models down and a card that is known."""
    return {
        "version": "1.2.3",
        "card": {"name": "NVIDIA GeForce RTX 3090", "vram_gb": 24,
                 "measured": True, "present": True, "reference": "RTX 3090"},
        "engine": {"running": True, "pid": 999, "port": 1919,
                   "answering": True, "model": "Qwen3.6-35B-A3B"},
        "endpoint": "http://127.0.0.1:1919",
        "vram": {"used_mb": 19000, "total_mb": 24576},
        "installed": list(MODELS),
        "catalog": [dict(c) for c in CATALOG],
        "downloads": [],
        "models_dir": "/home/x/models",
        "busy": False,
        "last": {"action": None, "ok": None, "detail": ""},
        "source": "panel",
        "log": [f"line {i}" for i in range(50)],
    }


class Recorder:
    """A ``submit`` that keeps the action instead of running it on a thread."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def __call__(self, label, action) -> None:
        self.calls.append((label, action))

    @property
    def labels(self) -> list[str]:
        return [label for label, _ in self.calls]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [60, 72, 80, 100, 132, 200])
@pytest.mark.parametrize("height", [18, 24, 30, 43, 60])
def test_the_screen_never_reaches_past_the_edge_of_the_window(snap, width, height):
    """curses raises rather than clipping, so an over-long line is a crash.

    Writing past the last column of a curses window raises ``curses.error``
    and takes the whole UI down; there is no forgiving truncation to fall back
    on. Every line the renderer emits is therefore already inside the window,
    at every size the renderer claims to support.
    """
    lines = tui.render(snap, tui.Ui(), width, height)
    assert len(lines) <= height
    for line in lines:
        assert len(line.text) <= width, repr(line.text)


@pytest.mark.parametrize("size", [(20, 5), (59, 24), (80, 17), (1, 1), (0, 0)])
def test_a_window_too_small_to_use_says_so_rather_than_drawing_a_ruin(snap, size):
    """A cramped layout that half-works is worse than a sentence saying why.

    The renderer is also the thing that must not overflow, so the small-window
    path has the same obligation as the ordinary one.
    """
    width, height = size
    lines = tui.render(snap, tui.Ui(), width, height)
    assert len(lines) <= max(height, 0)
    for line in lines:
        assert len(line.text) <= max(width, 0)
    if width >= 20 and height >= 3:
        assert "needs" in " ".join(line.text for line in lines)


def test_a_model_nobody_has_benchmarked_is_never_given_a_number(snap):
    """The browser panel once rendered such an entry as ``~null tok/s``.

    An absent measurement is a fact about the catalogue, not a value to format.
    Printing the absence as a number reads as a broken UI, and printing it as a
    speed would be an invented one.
    """
    row = next(c for c in snap["catalog"] if c["expected_tok_s"] is None)
    assert tui.speed_label(row) == "speed not measured"
    assert "None" not in tui.speed_label(row)
    screen = "\n".join(line.text for line in tui.render(snap, tui.Ui(), 120, 40))
    assert "None" not in screen and "null" not in screen


def test_a_speed_is_never_shown_without_where_it_was_measured(snap):
    """A number cut off from its provenance is the one claim this project bans.

    Speeds are never scaled between cards, so a figure shown on a machine that
    did not produce it has to carry the words that say so. Narrowness is not an
    excuse: the short label may drop ``(measured)``, which says the ordinary
    thing, and a row too narrow for the whole qualifier drops the speed
    entirely rather than truncating it into a bare number.
    """
    elsewhere = next(c for c in snap["catalog"] if not c["speed_applies"])
    here = next(
        c for c in snap["catalog"] if c["speed_applies"] and c["expected_tok_s"]
    )
    assert "(other card)" in tui.speed_label(elsewhere, brief=True)
    assert "(other card)" in tui.speed_label(elsewhere, brief=False)
    assert "(measured)" in tui.speed_label(here, brief=False)
    assert "(measured)" not in tui.speed_label(here, brief=True)

    for width in (60, 72, 80, 100, 132, 200):
        rows = [
            line.text
            for line in tui.render(snap, tui.Ui(pane="catalog"), width, 40)
            if elsewhere["name"] in line.text
        ]
        for row in rows:
            assert "tok/s" not in row or "(other card)" in row, f"{width}: {row!r}"
    wide = "\n".join(
        line.text for line in tui.render(snap, tui.Ui(pane="catalog"), 132, 40)
    )
    assert "(other card)" in wide, "a window with room must show the qualifier"


def test_the_row_under_the_cursor_is_always_drawn(snap):
    """A cursor you cannot see is a UI that acts on something invisible.

    The list scrolls to follow the selection; the failure this prevents is
    pressing start with the highlight off the bottom of a short window.
    """
    for count in (0, 1, 5, 40):
        for index in range(max(count, 1)):
            for rows in (1, 3, 7):
                start, end = tui.window(count, index, rows)
                assert end - start <= rows
                if count:
                    assert start <= index < end or index >= count


def test_the_log_pane_shows_the_end_of_the_log(snap):
    """A load failure is at the bottom of the log, never at the top."""
    screen = [line.text for line in tui.render(snap, tui.Ui(), 100, 24)]
    assert any("line 49" in text for text in screen)
    assert not any("line 0 " in text for text in screen)


def test_a_download_in_flight_shows_its_progress(snap):
    """Progress is the only sign a 17 GB fetch is alive rather than wedged."""
    snap["downloads"] = [
        {"id": "muse-glimmer-30b", "name": "Muse-Glimmer-30B",
         "state": "downloading", "percent": 41.2, "rate_mib_s": 33.1, "detail": ""}
    ]
    screen = "\n".join(
        line.text for line in tui.render(snap, tui.Ui(pane="catalog"), 100, 30)
    )
    assert "41.2%" in screen
    assert "#" in screen


def test_a_machine_with_nothing_downloaded_still_draws(snap):
    """The state a fresh install is in, and the one with no list to scroll."""
    snap["installed"] = []
    snap["engine"] = {"running": False, "pid": None, "port": 1919,
                      "answering": False, "model": None}
    lines = tui.render(snap, tui.Ui(), 80, 24)
    assert len(lines) <= 24
    assert any("nothing downloaded" in line.text for line in lines)


def test_the_footer_says_when_the_panel_is_not_there(snap):
    """Without the panel some keys do nothing, and the user has to know which."""
    snap["source"] = "local"
    note = tui.footer(snap, tui.Ui(), 100)[0].text
    assert "panel" in note and "downloads" in note


def test_a_start_that_failed_says_why_whoever_pressed_it(snap):
    """The panel remembers its last action, and a browser may have caused it.

    A start that fails on KV allocation fails minutes after the keypress, so
    the reason has to be somewhere the next person to look will see it.
    """
    snap["last"] = {"action": "start Qwen3.6-35B-A3B", "ok": False,
                    "detail": "engine exited with 1: failed to allocate KV"}
    line = tui.footer(snap, tui.Ui(), 120)[0]
    assert "FAILED" in line.text and "allocate KV" in line.text
    assert line.style == "bad"


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def test_start_acts_on_the_row_the_cursor_is_on(snap):
    """Not on the first row, and not on whatever the engine last ran.

    The catalogue pane starts a model too, when that entry is downloaded --
    otherwise the obvious keypress on the obvious row would do nothing.
    """
    control = tui.Control()
    submit = Recorder()
    ui = tui.Ui(pane="models", model=1)
    tui.handle_key("s", snap, ui, control, submit)
    assert submit.labels == ["start Qwen3-8B"]

    submit = Recorder()
    ui = tui.Ui(pane="catalog", entry=0)
    tui.handle_key("s", snap, ui, control, submit)
    assert submit.labels == ["start Qwen3.6-35B-A3B"]


def test_a_model_that_does_not_fit_is_never_downloaded(snap):
    """The catalogue shows entries that do not fit so the reason is visible.

    Showing them is not offering them: the panel rejects such a download with a
    400, and the terminal UI must not spend 40 GB finding that out.
    """
    control = tui.Control()
    submit = Recorder()
    ui = tui.Ui(pane="catalog", entry=3)
    tui.handle_key("d", snap, ui, control, submit)
    assert submit.calls == []
    assert "does not fit" in ui.message


def test_stopping_an_engine_that_is_not_running_is_not_an_action(snap):
    snap["engine"]["running"] = False
    control, submit, ui = tui.Control(), Recorder(), tui.Ui()
    tui.handle_key("x", snap, ui, control, submit)
    assert submit.calls == []
    assert "not running" in ui.message


def test_every_key_the_footer_advertises_does_something(snap):
    """The footer is the only documentation on screen, so it must not drift.

    A key listed there that falls through the handler is a promise the UI does
    not keep, and nothing else in the suite would notice.
    """
    control = tui.Control()
    for key, what in tui.KEYS:
        ui, submit = tui.Ui(), Recorder()
        alive = tui.handle_key(key, snap, ui, control, submit)
        acted = submit.calls or ui.message or ui != tui.Ui()
        assert acted or not alive, f"{key!r} ({what}) does nothing"


def test_quitting_is_the_only_key_that_ends_the_loop(snap):
    control = tui.Control()
    assert tui.handle_key("q", snap, tui.Ui(), control, Recorder()) is False
    for key in ("s", "x", "d", "c", "tab", "up", "down", "P", "z"):
        assert tui.handle_key(key, snap, tui.Ui(), control, Recorder()) is True


# ---------------------------------------------------------------------------
# Talking to the machine
# ---------------------------------------------------------------------------


def test_a_download_is_never_started_without_the_panel(monkeypatch):
    """One ``.part`` file, one writer.

    Downloads resume from a part file, and the panel restarts any it finds. A
    second process appending to the same file would not produce a slow
    download but a corrupt one, so with no panel there is no download -- and
    the message has to name the thing that would fix it.
    """
    def explode(entry):
        raise AssertionError("the terminal UI must not download anything itself")

    monkeypatch.setattr(downloads, "start", explode)
    control = tui.Control("http://127.0.0.1:1")
    control.reachable = False
    message = control.download("qwen3-8b")
    assert "panel" in message


def test_starting_a_model_without_the_panel_uses_the_pool_the_panel_would(
    monkeypatch,
):
    """The same model from the console and from the browser is the same engine.

    The pool and the slot count are the whole of what makes a start succeed or
    fail on a 24 GB card, so they come from one place -- and the engine is
    stopped first, because there is one GPU.
    """
    launched: list[tuple] = []
    stopped: list[int] = []
    monkeypatch.setattr(catalog, "installed", lambda *a, **k: list(MODELS))
    monkeypatch.setattr(engine, "stop", lambda *a, **k: stopped.append(1) or (True, ""))
    monkeypatch.setattr(
        engine, "start",
        lambda *args: launched.append(args) or (True, "starting Qwen3-8B"),
    )
    control = tui.Control("http://127.0.0.1:1")
    control.reachable = False
    detail = control.start("Qwen3-8B")

    expected = catalog.launch_plan("Qwen3-8B")
    path, name, ctx, parallel, wait = launched[0][:5]
    assert stopped, "the engine must be stopped before another is started"
    assert (path, name) == ("/m/Qwen3-8B/w.gguf", "Qwen3-8B")
    assert (ctx, parallel) == (expected.pool, expected.parallel)
    assert wait == 0, "a blocking start would freeze the screen for minutes"
    assert detail == "starting Qwen3-8B"


def test_an_unknown_gguf_is_planned_conservatively():
    """Nothing is known about its KV cost, so nothing may be assumed.

    Guessing high produces an engine that loads, reports itself healthy and
    fails every request out of device memory.
    """
    p = catalog.launch_plan("Something-Nobody-Curated")
    assert p.per_session == catalog.UNKNOWN_MODEL_CTX
    assert p.parallel == config.DEFAULT_PARALLEL
    assert p.capped_by == "default"


def test_the_panel_and_the_terminal_ui_describe_the_same_machine():
    """One snapshot, two front ends. A field only one of them has is a bug.

    The terminal UI renders whatever ``/api/status`` returns, so if the panel's
    shape and the local one drift apart, half the screen goes blank against a
    running panel or against none -- whichever was not the one being tested.
    """
    from fastapi.testclient import TestClient

    from lllm3090 import panel

    served = TestClient(panel.app).get("/api/status").json()
    local = state.snapshot()
    assert set(served) - set(local) == {"busy", "last"}
    assert set(local) - set(served) == set()


# ---------------------------------------------------------------------------
# The curses driver
# ---------------------------------------------------------------------------


class FakeCurses:
    """Just enough of the curses module to drive ``_draw`` without a terminal."""

    error = type("error", (Exception,), {})
    KEY_UP, KEY_DOWN, KEY_NPAGE, KEY_PPAGE = 259, 258, 338, 339

    def doupdate(self) -> None:
        pass


class FakeWindow:
    """A window that refuses the last cell, exactly as a real one does."""

    def __init__(self, height: int, width: int) -> None:
        self.height, self.width = height, width
        self.painted: list[str] = []

    def getmaxyx(self) -> tuple[int, int]:
        return self.height, self.width

    def erase(self) -> None:
        self.painted.clear()

    def noutrefresh(self) -> None:
        pass

    def addstr(self, y: int, x: int, text: str, attr: int = 0) -> None:
        if y >= self.height or x + len(text) > self.width:
            raise FakeCurses.error("addstr() returned ERR")
        if y == self.height - 1 and x + len(text) >= self.width:
            raise FakeCurses.error("addstr() returned ERR")
        self.painted.append(text)


def test_the_bottom_right_cell_is_left_alone(snap):
    """Filling the last cell of a curses window scrolls it, and raises.

    The text fits and the write still fails, which is why the bottom line stops
    one column short instead of trusting the width.
    """
    window = FakeWindow(24, 80)
    tui._draw(window, FakeCurses(), tui.render(snap, tui.Ui(), 80, 24), {})
    assert window.painted, "nothing was drawn at all"
    assert len(window.painted[-1]) < 80


def test_a_key_is_named_before_it_is_acted_on():
    """The handler is written against key names, not against terminal codes."""
    curses = FakeCurses()
    assert tui._key_name(curses, curses.KEY_UP) == "up"
    assert tui._key_name(curses, 9) == "tab"
    assert tui._key_name(curses, ord("q")) == "q"
    assert tui._key_name(curses, ord("P")) == "P"


def test_the_curses_driver_paints_a_screen_and_quits(tmp_path, monkeypatch):
    """Everything else here asserts on strings. This one runs the real thing.

    A UI that renders perfectly into a list and cannot initialise ncurses is
    still a UI that does not work, so this drives ``lllm3090 tui`` on a
    pseudo-terminal with no panel, no models and no engine, and checks that it
    paints and then leaves when asked.
    """
    pty = pytest.importorskip("pty")
    pytest.importorskip("curses")
    import os
    import select
    import time

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(config, "ENGINE_LOG", tmp_path / "engine.log")
    monkeypatch.setattr(config, "ENGINE_PID", tmp_path / "engine.pid")
    # Port 9 is discard: nothing answers, so the UI falls back to this process.
    monkeypatch.setattr(config, "PANEL_URL", "http://127.0.0.1:9")

    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover -- the child never returns
        try:
            os.environ["TERM"] = "xterm"
            tui.run()
            os._exit(0)
        except BaseException:
            os._exit(9)

    painted, sent, gone, deadline = "", False, False, time.time() + 30
    try:
        while not gone and time.time() < deadline:
            if select.select([fd], [], [], 0.5)[0]:
                try:
                    chunk = os.read(fd, 65536)
                except OSError:  # the child closed the terminal and exited
                    chunk = b""
                gone = not chunk
                painted += chunk.decode("utf-8", "replace")
            if not sent and "INSTALLED" in painted:
                os.write(fd, b"q")
                sent = True
    finally:
        os.close(fd)
        if not gone:
            os.kill(pid, 9)
    status = os.waitpid(pid, 0)[1]

    if not sent:
        if not painted.strip():
            pytest.skip("curses could not start a terminal in this environment")
        pytest.fail(f"the UI painted nothing recognisable: {painted!r}")
    assert status == 0, "the UI did not leave cleanly when asked to quit"
    assert "lllm3090" in painted
    assert "CATALOGUE" in painted and "ENGINE LOG" in painted
    assert "nothing downloaded yet" in painted


def test_the_cursor_survives_a_list_that_shrinks_under_it(snap):
    """A model deleted from disk must not close the UI on the next keypress.

    The poller replaces the snapshot on its own thread, so the cursor can be
    left pointing past the end of the list it was placed in -- and every read
    of it after that is an IndexError out of `handle_key`, which is a UI that
    exits when you press a key.
    """
    control, submit = tui.Control(), Recorder()
    ui = tui.Ui(pane="models", model=1)
    shrunk = dict(snap, installed=[MODELS[0]])
    assert tui.handle_key("s", shrunk, ui, control, submit) is True
    assert ui.model == 0, "the cursor must be pulled back onto the shorter list"
    assert submit.labels == [f"start {MODELS[0]['name']}"], (
        "and the key must act on the row it was pulled back to"
    )

    empty = dict(snap, installed=[], catalog=[])
    ui = tui.Ui(pane="models", model=3, entry=2)
    assert tui.handle_key("s", empty, ui, tui.Control(), Recorder()) is True
    assert (ui.model, ui.entry) == (0, 0)
    assert tui.render(empty, ui, 100, 30), "an empty machine still draws"


def test_the_terminal_ui_is_refused_without_a_terminal_to_draw_on(monkeypatch):
    """Both streams, not just the one curses writes to.

    curses draws on stdout and reads the keyboard from stdin, so a piped stdin
    with a real stdout gets a screen that repaints and answers nothing.
    """
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from lllm3090 import cli

    # A stand-in for `sys`, because CliRunner substitutes the real streams for
    # its own during invoke -- patching those would be patching the runner.
    def streams(stdin: bool, stdout: bool) -> SimpleNamespace:
        return SimpleNamespace(
            stdin=SimpleNamespace(isatty=lambda: stdin),
            stdout=SimpleNamespace(isatty=lambda: stdout),
        )

    for stdin_tty, stdout_tty in ((False, True), (True, False), (False, False)):
        monkeypatch.setattr(cli, "sys", streams(stdin_tty, stdout_tty))
        result = CliRunner().invoke(cli.app, ["tui"])
        assert result.exit_code == 1, (stdin_tty, stdout_tty, result.output)
        assert "needs a terminal" in result.output


def test_a_slot_count_below_one_is_not_a_plan():
    """Zero is swallowed by the default and negative divides the pool backwards.

    Either way the engine is started with it, so it is refused where the plan
    is made rather than left to produce a Plan that looks like any other.
    """
    for bad in (0, -1):
        with pytest.raises(ValueError, match="at least 1"):
            catalog.launch_plan("Qwen3-8B", bad)
    assert catalog.launch_plan("Qwen3-8B", 1).parallel == 1
    assert catalog.launch_plan("Qwen3-8B").parallel == config.DEFAULT_PARALLEL


# ---------------------------------------------------------------------------
# The log
# ---------------------------------------------------------------------------


def test_the_log_tail_strips_the_redraws_llama_server_writes(tmp_path, monkeypatch):
    """llama.cpp writes progress by redrawing a line with carriage returns.

    Rendered literally, one load turns into hundreds of near-identical rows and
    the actual message scrolls away. Only the last frame of a redraw is a line.
    """
    log = tmp_path / "engine.log"
    log.write_text(
        "\x1b[32mload: ok\x1b[0m\n"
        "\n"
        "progress 1%\rprogress 50%\rprogress 100%\n"
        "srv    load_model: done\n"
    )
    monkeypatch.setattr(config, "ENGINE_LOG", log)
    assert engine.tail() == ["load: ok", "progress 100%", "srv    load_model: done"]
    assert engine.tail(1) == ["srv    load_model: done"]


def test_a_redraw_split_across_two_reads_is_still_one_line(tmp_path, monkeypatch):
    """The SSE stream must read the log the way `engine.tail` reads it.

    Universal newlines turn a progress redraw's \\r into \\n before anything
    can see it, so `engine.clean` gets one frame per update instead of one
    line, and the browser is sent the hundreds of rows the tail pane does not
    show. The two views of the same file have to agree.
    """
    log = tmp_path / "engine.log"
    log.write_text("")
    monkeypatch.setattr(config, "ENGINE_LOG", log)

    async def collect() -> list[str]:
        out: list[str] = []
        stream = panel._tail_stream()
        # The generator seeks to the end of what exists before it yields, so
        # the file is grown after it starts -- which is the case that matters.
        assert await anext(stream) == "retry: 3000\n\n"
        with log.open("a") as f:
            f.write("progress 1%\rprogress 50%\rprogress 100%\nsrv done\n")
        for _ in range(3):
            frame = await anext(stream)
            if frame.startswith("data: "):
                out.append(json.loads(frame[len("data: "):].strip()))
            if len(out) >= 2:
                break
        await stream.aclose()
        return out

    lines = asyncio.run(collect())
    assert lines[0] == "progress 100%", "a redraw is one line, at its last frame"
    assert lines[1] == "srv done"
    assert engine.tail(2) == ["progress 100%", "srv done"], (
        "and the polled tail must agree with the stream"
    )


def test_asking_for_no_lines_gets_no_lines(tmp_path, monkeypatch):
    """`readable[-0:]` is the whole log, which is the opposite of the request.

    The panel takes this count off a query string, so nothing between the
    caller and the slice makes zero impossible.
    """
    log = tmp_path / "engine.log"
    log.write_text("one\ntwo\nthree\n")
    monkeypatch.setattr(config, "ENGINE_LOG", log)
    assert engine.tail(3) == ["one", "two", "three"]
    assert engine.tail(0) == []
    assert engine.tail(-1) == []


def test_a_missing_log_is_not_an_error(tmp_path, monkeypatch):
    """Nothing has been started yet. That is the ordinary state after install."""
    monkeypatch.setattr(config, "ENGINE_LOG", tmp_path / "nothing.log")
    assert engine.tail() == []
