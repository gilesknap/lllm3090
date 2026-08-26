"""The changelog, which nothing else checks.

A release renames `Unreleased` and adds two links at the foot of the file, by
hand, at the moment when attention is on the tag rather than on the prose.
These are the three ways that goes wrong and nobody notices until someone
clicks a link.
"""

from __future__ import annotations

import re
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"
TEXT = CHANGELOG.read_text()

#: `## [0.4.0] — 2026-08-25` or `## [Unreleased]`
HEADING = re.compile(
    r"^## \[([^\]]+)\](?: [-—] (\d{4}-\d{2}-\d{2}))?$", re.MULTILINE
)


def test_there_is_somewhere_to_put_the_next_change():
    """Without it the next change is either lost or filed under a shipped
    version, which is worse than lost."""
    assert TEXT.count("## [Unreleased]") == 1
    assert HEADING.search(TEXT).group(1) == "Unreleased", "and it is at the top"


def test_every_released_version_says_when():
    versions = [(name, date) for name, date in HEADING.findall(TEXT)
                if name != "Unreleased"]
    assert versions, "a changelog with no releases in it is a to-do list"
    assert all(date for _, date in versions), "a release without a date"
    assert [d for _, d in versions] == sorted((d for _, d in versions), reverse=True)


def test_every_heading_has_a_link_to_the_diff_it_describes():
    """The link definitions are the half of the release edit that is easiest
    to forget, and a bare `[0.4.0]` renders as literal brackets."""
    defined = set(re.findall(r"^\[([^\]]+)\]: https://", TEXT, re.MULTILINE))
    assert {name for name, _ in HEADING.findall(TEXT)} <= defined
