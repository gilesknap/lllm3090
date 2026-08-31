"""What the engine is told to guess with, and which backend each setting wins on.

Speculative decoding drafts the next few tokens and verifies them in one pass,
so accepted drafts cost a fraction of a forward pass. Which drafter, and how
many tokens it may draft, are two separate knobs -- and the second turned out
to matter more than the first.

**The verdicts invert between backends, which is why this is a table and not a
flag.** Verifying k drafted tokens *is* a batched forward pass, and Vulkan gets
nothing from a wider batch: 1026.9 tok/s at pp512 against 1014.0 at pp4096,
where CUDA goes 1217.4 to 1343.8. So a backend that does not reward wider
batches punishes wider drafts, and every drafter verdict measured on Vulkan was
measured on the backend least able to make drafting pay. On the dense 27B:

* ``ngram-cache`` is **1.41x** on CUDA copy-heavy work and **0.65x** on Vulkan
  prose.
* Draft width 7 wins copying on CUDA; on Vulkan it costs MTP 22% on prose.

A "go faster" switch that did not know which backend it was on would therefore
be a footgun aimed at the default install, which is Vulkan. Each profile here
records the backends it was *measured* to win on and is refused on the others,
with the number, rather than being quietly applied and quietly losing.

Two kinds of answer, deliberately different:

* :data:`DEFAULT` is what every start gets when nobody asked for anything. It
  **degrades**: a checkpoint with no multi-token prediction head simply gets no
  speculation, because the alternative is an engine that refuses to launch.
* A named profile is a **request**, and a request that cannot be honoured is
  refused rather than approximated. ``--profile copy`` is a measurement of
  ``draft-mtp,ngram-cache``; run against a checkpoint with no MTP head it would
  be a different configuration that nobody here has measured, and serving that
  under the same name is how a number stops meaning anything.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    """One measured speculation setting, and where it is allowed to be used."""

    name: str
    #: ``--spec-type``, comma-joined. Empty means no speculation at all.
    spec_types: tuple[str, ...]
    #: ``--spec-draft-n-max``. ``None`` leaves llama.cpp's own default of 3,
    #: which is what the engine has always passed -- and on Vulkan that is the
    #: right width, so "pass nothing" is a decision rather than an omission.
    draft_n_max: int | None
    #: Backends this was measured to win on. ``None`` means "anywhere": true
    #: only of the default, which is the floor rather than an optimisation.
    backends: frozenset[str] | None
    #: One line for ``--help`` and for the line printed at start.
    summary: str
    #: Why it is refused elsewhere, in measured numbers. Empty for the default.
    elsewhere: str = ""
    #: Whether the configuration is only meaningful with an MTP head present.
    needs_mtp: bool = False

    def allowed_on(self, backend: str) -> bool:
        return self.backends is None or backend in self.backends


class Unavailable(Exception):
    """This profile is not measured to win on the backend that is installed."""


#: What every start gets when nobody asked for anything.
#:
#: Multi-token prediction alone, at llama.cpp's default draft width. Measured
#: on the reference 3090: 34.9 -> 56.6 tok/s (1.62x) on the dense 27B, 130.5 ->
#: 171.8 (1.32x) on the sparse 35B. Nothing tried has displaced it on Vulkan --
#: DFlash2 is a wash for 1.1 GB of VRAM, and ngram is a loss on every workload.
DEFAULT = Profile(
    name="default",
    spec_types=("draft-mtp",),
    draft_n_max=None,
    backends=None,
    summary="multi-token prediction, at the engine's own draft width",
)

#: For a session that is mostly reproducing its input.
#:
#: Adds prompt-lookup drafting to the MTP head and widens the draft to 7. On
#: CUDA this takes the dense 27B's copy-heavy figure from 94.0 to 115.1 tok/s.
#: It is not free even there: it costs about 14% on prose, because n-gram
#: drafting can only pay when the output repeats a long stretch of the input.
#: Take it for a session spent editing a file back out, not for a conversation.
COPY = Profile(
    name="copy",
    spec_types=("draft-mtp", "ngram-cache"),
    draft_n_max=7,
    backends=frozenset({"cuda"}),
    summary="MTP plus prompt-lookup at draft width 7, for copy-heavy work",
    elsewhere=(
        "On Vulkan this configuration loses, measured on the dense 27B at b10715:\n"
        "  adding ngram-cache to MTP reads 0.84-0.90x against MTP alone,\n"
        "  widening the draft from 3 to 7 costs MTP a further 22% on prose,\n"
        "  and ngram-cache by itself is 0.65x.\n"
        "Vulkan gets nothing from a wider batch and verifying a draft is a "
        "batched pass,\nwhich is the mechanism behind all three."
    ),
    needs_mtp=True,
)

#: Everything ``--profile`` will accept, in the order ``--help`` should list it.
PROFILES: dict[str, Profile] = {p.name: p for p in (DEFAULT, COPY)}


def names() -> str:
    """The profile names, for a help string."""
    return ", ".join(PROFILES)


def resolve(name: str | None, backend: str) -> Profile:
    """The profile called ``name``, if this backend is one it wins on.

    ``None`` is the default profile, which is allowed everywhere. Anything else
    is a request, and :class:`Unavailable` carries the measurement that says no
    -- a refusal whose reason is a number is one the user can argue with.
    """
    if name is None:
        return DEFAULT
    profile = PROFILES.get(name)
    if profile is None:
        raise Unavailable(
            f"unknown speculation profile {name!r}. Known profiles: {names()}"
        )
    if not profile.allowed_on(backend):
        where = ", ".join(sorted(profile.backends or ()))
        raise Unavailable(
            f"--profile {profile.name} is measured on {where}, and the engine "
            f"installed here is {backend}.\n{profile.elsewhere}"
        )
    return profile
