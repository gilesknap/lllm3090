"""The shipped systemd unit must actually parse.

It did not. A botched edit left `TimeoutStopSec=15 is that it can leave
processes` as a live directive with half a sentence appended, and systemd
reported it on every single start:

    lllm3090-panel.service:25: Failed to parse TimeoutStopSec=15 is that it
    can leave processes, ignoring: Invalid argument

Nothing misbehaved, because a second, correct TimeoutStopSec further down
happened to parse -- which is exactly why it survived. Tests read the file
nobody was reading.
"""

from __future__ import annotations

import re
from importlib import resources

DIRECTIVE = re.compile(r"^([A-Za-z][A-Za-z0-9]*)=(.*)$")
#: Directives whose value is a systemd time span: a number and an optional unit.
TIME_SPANS = {"TimeoutStopSec", "TimeoutStartSec", "RestartSec", "TimeoutSec"}


def _sections() -> dict[str, list[tuple[str, str]]]:
    raw = (
        resources.files("lllm3090.data")
        .joinpath("lllm3090-panel.service")
        .read_text()
    )
    sections: dict[str, list[tuple[str, str]]] = {}
    current = ""
    for n, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, [])
            continue
        m = DIRECTIVE.match(line)
        assert m, f"line {n} is neither a comment nor Key=Value: {line!r}"
        sections[current].append((m.group(1), m.group(2)))
    return sections


def test_every_line_is_a_comment_or_a_directive():
    """Prose that has escaped its `#` is the whole bug this file guards."""
    assert _sections(), "unit file parsed to nothing"


def test_no_directive_is_given_twice_in_a_section():
    """A duplicate is how the broken one hid: the good copy masked the bad."""
    for section, directives in _sections().items():
        keys = [k for k, _ in directives]
        dupes = {k for k in keys if keys.count(k) > 1}
        assert not dupes, f"[{section}] sets {sorted(dupes)} more than once"


def test_time_spans_are_times():
    for section, directives in _sections().items():
        for key, value in directives:
            if key in TIME_SPANS:
                assert re.fullmatch(r"\d+(ms|s|min|m|h)?", value), (
                    f"[{section}] {key}={value!r} is not a systemd time span"
                )


def test_the_unit_keeps_the_settings_the_engine_depends_on():
    """KillMode=process is what stops a panel restart killing the engine."""
    service = dict(_sections()["Service"])
    assert service["KillMode"] == "process"
    assert service["Restart"] == "on-failure"
