"""Cut the suite off from the machine it happens to be running on.

Every path in :mod:`lllm3090.config` is read from the environment **at import
time**, so by the time a test runs the module constants already point at the
developer's real models, engines and state. Most of the suite never notices,
which is exactly the problem: the tests that do notice fail only on a machine
that has been *used*, and pass on a fresh CI runner.

The failure that prompted this was not subtle once found, and is worth writing
down because nothing about it announced itself. Choosing the CUDA engine from
the panel leaves ``{"dir": "b10715-cuda-sm86"}`` in the state directory;
``engines.chosen()`` then -- and *only* then, because with no stored choice it
returns ``None`` first -- calls ``choices()``, which probes ``llama-server
--list-devices`` through ``subprocess``. That collided with the fake ``Popen``
in ``test_engine_args.py`` and took 33 tests down with it, none of which had
anything to do with the change under test.

So the redirection is on by default rather than opt-in: inheriting the
developer's machine has to be something a test *asks* for. Two tests do ask --
they exist precisely to check the code against real weights and real binaries
-- and they say so with ``@pytest.mark.real_machine``.
"""

from __future__ import annotations

import pytest

from lllm3090 import config


@pytest.fixture(autouse=True)
def _machine_state_elsewhere(request, tmp_path, monkeypatch):
    """Point config's directories at an empty tmp dir, for every test.

    Except one marked ``real_machine``, which is opting into whatever this
    box actually has -- and which is expected to skip itself when the box has
    nothing, because CI is such a box.

    The directories are deliberately left *unmade*: absent is the state a
    fresh machine is in, and code that has to cope with a missing directory
    should be exercised doing so rather than handed one that exists.

    ``ENGINE_LOG``, ``ENGINE_PID`` and ``ENGINE_CHOICE`` are separate constants
    computed from ``STATE_DIR`` at import, so moving ``STATE_DIR`` alone would
    move the directory and leave the three files behind in the real one.
    """
    if request.node.get_closest_marker("real_machine"):
        return
    state = tmp_path / "state"
    monkeypatch.setattr(config, "STATE_DIR", state)
    monkeypatch.setattr(config, "ENGINE_LOG", state / "engine.log")
    monkeypatch.setattr(config, "ENGINE_PID", state / "engine.pid")
    monkeypatch.setattr(config, "ENGINE_CHOICE", state / "engine.json")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "LLAMA_DIR", tmp_path / "llama.cpp")
    monkeypatch.setattr(config, "ENGINES_DIR", tmp_path / "engines")
    # An override in the developer's shell would otherwise outrank every
    # stored choice inside the suite, so the engine-selection tests would
    # assert against a decision the environment had already made for them.
    monkeypatch.setattr(config, "LLAMA_DIR_FROM_ENV", False)
