"""That every badge on a model row points at documentation that exists.

The panel's badges name concepts -- `MOE`, `MTP`, `TEMPLATE` -- and each one
links to the page that explains it. A heading anchor is generated from the
heading *text*, so rewording a heading silently breaks the link: the page still
loads, the browser lands at the top of it, and nothing anywhere reports a
failure. That is the kind of rot nobody finds by using the product, because the
person who follows the link does not know where they were supposed to arrive.

So the map lives in `config.BADGE_DOCS` rather than in the page, and this reads
it against the Markdown the docs are built from.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lllm3090 import config

DOCS = Path(__file__).resolve().parent.parent / "docs"

#: `## Heading text`, at any depth.
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.MULTILINE)


def slug(heading: str) -> str:
    """The anchor Sphinx gives a heading: lowercase, punctuation to hyphens.

    Reimplemented rather than taken from a build, because the build happens
    after this in CI and the point is to fail before it. If the two ever
    disagree the failure is loud, which is the right way round.
    """
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


@pytest.mark.parametrize("badge,target", sorted(config.BADGE_DOCS.items()))
def test_a_badge_links_to_a_page_that_is_here(badge, target):
    page, _, anchor = target.partition("#")
    source = DOCS / f"{page}.md"
    assert source.is_file(), f"the {badge} badge points at a page that is gone"
    if not anchor:
        return
    headings = {slug(h) for h in HEADING.findall(source.read_text())}
    assert anchor in headings, (
        f"the {badge} badge points at #{anchor}, and no heading in "
        f"{page}.md makes that anchor any more. Reword the link or the heading."
    )


def test_every_badge_the_panel_draws_has_somewhere_to_send_a_reader():
    """A badge with no entry renders as plain text rather than a link, which is
    a quiet way to lose one. These are the labels `rowHtml` can produce."""
    drawn = {"dense", "moe", "vision", "mtp", "template"}
    assert drawn <= set(config.BADGE_DOCS)


def test_the_kinds_in_the_catalogue_are_all_badges_that_link():
    """`kind` comes from models.yaml and is rendered as a badge unchanged, so a
    new kind arrives as an unlinked chip unless it is added here too."""
    from lllm3090 import catalog

    kinds = {m.kind for m in catalog.load_catalog()}
    unlinked = kinds - set(config.BADGE_DOCS)
    assert not unlinked, f"kinds with no page to send a reader to: {unlinked}"
