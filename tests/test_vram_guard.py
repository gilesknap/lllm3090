"""Every front end must refuse to start a plan quietly when the card is full.

The plan is computed against fixed reserves, which describe a machine at rest.
What the card actually has free is a measurement, and when the two disagree the
engine loads, reports itself healthy, and fails every request out of device
memory. The console and the panel start the same engine on the same card, so a
guard on one of them and not the other is not a guard at all.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from lllm3090 import catalog, cli, engine, hardware, panel

runner = CliRunner()

MODEL = "Qwen3.6-35B-A3B"
ENTRY = {"name": MODEL, "path": f"/models/{MODEL}/w.gguf", "mmproj": None, "gb": 17.7}


@pytest.fixture
def starved(monkeypatch):
    """An installed catalogue model, a stubbed engine, and a card with 1 GB free."""
    monkeypatch.setattr(catalog, "installed", lambda: [ENTRY])
    monkeypatch.setattr(hardware, "free_vram_mib", lambda: 1024)
    monkeypatch.setattr(engine, "stop", lambda: (True, "stopped"))
    monkeypatch.setattr(engine, "start", lambda *a, **k: (True, "engine started"))


def test_the_console_says_so(starved):
    result = runner.invoke(cli.app, ["start", MODEL])
    assert result.exit_code == 0, result.output
    assert "Warning:" in result.output
    assert "GB is free" in result.output


def test_the_panel_says_so(starved):
    """The panel starts the same engine, so it owes the same warning.

    It reports through JSON rather than a terminal, so the warning is both its
    own field -- for anything reading the API -- and part of the detail line
    the page shows, which otherwise reports an unqualified success.
    """
    with TestClient(panel.app) as client:
        body = client.post(f"/api/start?model={MODEL}").json()
    assert body["warning"], "the panel must not start a starved plan silently"
    assert "Warning:" in body["detail"]
    assert "engine started" in body["detail"], "the launch result is still reported"


def test_a_card_with_room_is_not_nagged(monkeypatch):
    """A warning on every start is a warning nobody reads."""
    monkeypatch.setattr(catalog, "installed", lambda: [ENTRY])
    monkeypatch.setattr(hardware, "free_vram_mib", lambda: 1024 * 1024)
    monkeypatch.setattr(engine, "stop", lambda: (True, "stopped"))
    monkeypatch.setattr(engine, "start", lambda *a, **k: (True, "started"))
    result = runner.invoke(cli.app, ["start", MODEL])
    assert result.exit_code == 0, result.output
    assert "Warning:" not in result.output


def test_no_card_is_not_reported_as_a_card_with_all_its_memory(monkeypatch):
    """With no GPU the capacity is borrowed, and nothing may be claimed for it.

    ``present=False`` and ``graphical()==False`` occur together in CI and in a
    container. Announcing that the full card is available there describes a
    card that is not in this machine.
    """
    borrowed = replace(hardware.reference(), present=False, measured=False)
    monkeypatch.setattr(cli.hardware, "detect", lambda: borrowed)
    monkeypatch.setattr(cli.hardware, "graphical", lambda: False)
    result = runner.invoke(cli.app, ["models"])
    assert result.exit_code == 0, result.output
    assert "none detected" in result.output
    assert "full card is available" not in result.output
