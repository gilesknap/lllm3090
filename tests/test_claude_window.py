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
    # `claude` on PATH, because these tests are about the window handed to it
    # and not about whether the machine running them has Claude Code installed.
    # Without this they pass on a developer's box and fail on every runner.
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
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


def test_print_env_prints_the_same_environment_it_would_launch_with(
    launched, monkeypatch
):
    """The point of the flag: check the mapping against a new Claude Code
    without starting a session to find out what it did with it."""
    result = runner.invoke(cli.app, ["claude", "--print-env"])
    assert result.exit_code == 0, result.output
    assert launched == {}, "--print-env must not launch anything"

    printed = dict(
        line.removeprefix("export ").split("=", 1)
        for line in result.stdout.splitlines()
        if line.startswith("export ")
    )
    window = catalog.launch_plan("Qwen3.8-27B").per_session
    assert printed == cli.claude_env("Qwen3.8-27B", window)
    assert f"unset {cli.CLAUDE_UNSET}" in result.stdout


def test_print_env_keeps_everything_but_the_environment_off_stdout(
    launched, monkeypatch
):
    """So that `eval "$(lllm3090 claude --print-env)"` is safe: a warning on
    stdout would be evaluated by the shell."""
    monkeypatch.setattr(
        cli.engine, "status",
        lambda: {"answering": True, "model": "Tiny-1B", "running": True},
    )
    monkeypatch.setattr(
        cli.catalog, "launch_plan",
        lambda *a, **k: catalog.Plan(
            pool=20000, per_session=20000, parallel=1, capped_by="vram"
        ),
    )
    result = runner.invoke(cli.app, ["claude", "--print-env", "--force"])
    assert result.exit_code == 0, result.output
    for line in result.stdout.splitlines():
        assert line.startswith(("export ", "unset ", "#")), line


def test_the_engine_being_down_is_reported_before_anything_is_printed(
    launched, monkeypatch
):
    monkeypatch.setattr(
        cli.engine, "status",
        lambda: {"answering": False, "model": None, "running": False},
    )
    result = runner.invoke(cli.app, ["claude", "--print-env"])
    assert result.exit_code == 1
    assert "export" not in result.stdout


def test_every_model_slot_points_at_the_local_model(launched):
    """`/model` inside the session must not fall back to the paid API."""
    env = cli.claude_env("Some-Model", 65536)
    assert {v for k, v in env.items() if k.endswith("_MODEL")} == {"Some-Model"}
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"
