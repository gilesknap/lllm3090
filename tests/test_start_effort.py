"""`lllm3090 start --effort`, from the console down to the launch arguments.

The level is the one launch argument a user types for its effect on the
*answers* rather than on the fit, and the engine takes it once, at startup —
there is no way to change it later from inside a session. That makes both
failure modes expensive: a typo that costs a reload, and a level accepted here
that the model's chat template will refuse on every request.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from lllm3090 import catalog, cli, engine, hardware

runner = CliRunner()

MODEL = "Qwen3.8-27B"
ENTRY = {"name": MODEL, "path": f"/models/{MODEL}/w.gguf", "mmproj": None, "gb": 15.4}


@pytest.fixture
def console(monkeypatch):
    """An installed model, a card with room, and an engine that records."""
    monkeypatch.setattr(catalog, "installed", lambda: [ENTRY])
    monkeypatch.setattr(hardware, "free_vram_mib", lambda: 24000)
    stopped: list[bool] = []
    monkeypatch.setattr(engine, "stop", lambda: stopped.append(True))
    launched: dict = {}

    def fake_start(*args):
        launched["args"] = args
        return True, "engine ready"

    monkeypatch.setattr(engine, "start", fake_start)
    return stopped, launched


def test_the_level_reaches_the_engine(console):
    stopped, launched = console
    result = runner.invoke(cli.app, ["start", MODEL, "--effort", "low"])
    assert result.exit_code == 0, result.output
    assert launched["args"][-1] == "low"
    assert stopped, "starting a model still replaces whatever was running"


def test_omitting_it_leaves_the_model_to_decide(console):
    _, launched = console
    result = runner.invoke(cli.app, ["start", MODEL])
    assert result.exit_code == 0, result.output
    assert launched["args"][-1] is None


def test_a_typo_is_refused_before_the_running_engine_is_stopped(console):
    """The check is worth doing early: `--effort hihg` would otherwise cost a
    working engine and the minutes it takes to load the weights again."""
    stopped, launched = console
    result = runner.invoke(cli.app, ["start", MODEL, "--effort", "hihg"])
    assert result.exit_code == 1
    assert "unknown effort 'hihg'" in result.output
    assert "xhigh" in result.output, "and says what would have worked"
    assert not stopped, "the engine that was running must still be running"
    assert not launched
