"""The window `lllm3090 claude` hands to Claude Code.

Claude Code believes whatever `CLAUDE_CODE_MAX_CONTEXT_TOKENS` says. If that
number is computed against a different VRAM budget than the one the engine was
started with, the agent silently caps itself below what the card is already
serving -- which is what happened on a text console: the engine took the
desktop reserve back and the agent did not.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from lllm3090 import catalog, cli

runner = CliRunner()


@pytest.fixture
def launched(monkeypatch):
    """A running engine serving a catalogue model, and a `claude` that records."""
    monkeypatch.setattr(
        cli.engine,
        "status",
        lambda: {"answering": True, "model": "Qwen3.8-27B", "running": True},
    )
    seen: dict = {}

    def fake_call(argv, env=None, **kw):
        seen["argv"] = argv
        seen["env"] = env
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    return seen


def _window(seen) -> int:
    return int(seen["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"])


def test_the_agent_is_told_the_window_the_engine_was_started_with(
    launched, monkeypatch
):
    """On a text console, the agent gets the console window -- not the desktop one.

    `launch_plan` is what sized the running engine. Anything else here is a
    second opinion, and the two were observed to differ by 39k tokens.
    """
    monkeypatch.setattr(cli.hardware, "graphical", lambda: False)
    result = runner.invoke(cli.app, ["claude"])
    assert result.exit_code == 0, result.output
    assert _window(launched) == catalog.launch_plan("Qwen3.8-27B").per_session


def test_a_desktop_still_gets_the_smaller_window(launched, monkeypatch):
    """The reserve is real when a compositor is holding it; do not hand it out."""
    monkeypatch.setattr(cli.hardware, "graphical", lambda: True)
    runner.invoke(cli.app, ["claude"])
    desktop_window = _window(launched)

    monkeypatch.setattr(cli.hardware, "graphical", lambda: False)
    runner.invoke(cli.app, ["claude"])
    assert _window(launched) > desktop_window, (
        "a text console frees the desktop reserve, so its window must be larger"
    )


def test_an_unknown_model_falls_back_rather_than_crashing(launched, monkeypatch):
    """A GGUF the catalogue never heard of gets the conservative fallback.

    32k is below the harness's own prompt, so the guard refuses it -- that is
    the guard working, and --force is the way past it.
    """
    monkeypatch.setattr(
        cli.engine,
        "status",
        lambda: {"answering": True, "model": "some-local.gguf", "running": True},
    )
    monkeypatch.setattr(cli.hardware, "graphical", lambda: False)
    refused = runner.invoke(cli.app, ["claude"])
    assert refused.exit_code == 1
    assert "prompt is too long" in refused.output

    result = runner.invoke(cli.app, ["claude", "--force"])
    assert result.exit_code == 0, result.output
    assert _window(launched) == catalog.UNKNOWN_MODEL_CTX


def test_the_models_offered_instead_are_sized_for_this_machine(launched, monkeypatch):
    """The alternatives list is advice, so it must quote the window you'd get.

    It reached for `plan`'s desktop default, so on a text console it advertised
    windows smaller than the ones it would then hand out.
    """
    monkeypatch.setattr(
        cli.engine,
        "status",
        lambda: {"answering": True, "model": "some-local.gguf", "running": True},
    )
    monkeypatch.setattr(cli.hardware, "graphical", lambda: False)
    console = runner.invoke(cli.app, ["claude"]).output
    headline = catalog.launch_plan("Qwen3.8-27B").per_session // 1024
    monkeypatch.setattr(cli.hardware, "graphical", lambda: True)
    desktop = runner.invoke(cli.app, ["claude"]).output

    assert f"Qwen3.8-27B ({headline}k)" in console
    assert console != desktop, "the same list on a console and a desktop is the bug"
