"""That the suite is looking at a temporary machine, not this one.

Without this, the fixture in ``conftest.py`` can stop working -- a constant
added to ``config`` and not redirected, a rename -- and nothing would say so.
The symptom would be what it was before the fixture existed: a suite that is
green on a fresh checkout and fails on a machine somebody has used, with the
failures landing in tests that have no connection to the cause.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lllm3090 import config

HOME = Path.home()

#: Every path constant a test could reach the real machine through.
REDIRECTED = [
    "MODELS_DIR", "STATE_DIR", "LLAMA_DIR", "ENGINES_DIR",
    "ENGINE_LOG", "ENGINE_PID", "ENGINE_CHOICE",
]


@pytest.mark.parametrize("name", REDIRECTED)
def test_no_test_can_see_the_real_directories(name):
    path = getattr(config, name)
    assert not path.is_relative_to(HOME), (
        f"config.{name} still points inside {HOME}; add it to the fixture in "
        "tests/conftest.py"
    )


def test_the_engine_files_moved_with_the_directory_that_holds_them():
    """They are computed from ``STATE_DIR`` at import, so redirecting the
    directory alone leaves all three behind in the real one -- and a test that
    wrote a pidfile would write it into the developer's running install."""
    for name in ("ENGINE_LOG", "ENGINE_PID", "ENGINE_CHOICE"):
        assert getattr(config, name).parent == config.STATE_DIR


def test_a_stored_engine_choice_cannot_leak_in():
    """The specific fault this all exists for: a choice made once in the panel
    used to reach ``chosen()``, which probes a real ``llama-server``."""
    assert not config.ENGINE_CHOICE.exists()
    assert not config.LLAMA_DIR_FROM_ENV
