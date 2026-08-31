"""Which speculation settings are allowed where, and what a refusal says.

The whole reason this table exists is that the verdicts invert between
backends: ngram drafting is 1.41x on CUDA copy-heavy work and 0.65x on Vulkan
prose. So the interesting tests are all refusals, and what they say -- a
refusal whose reason is a number is one the user can argue with, and a refusal
that just says "no" sends someone to the source to find out why.
"""

from __future__ import annotations

import pytest

from lllm3090 import speculation


def test_the_default_is_allowed_on_every_backend():
    """It is the floor rather than an optimisation: MTP alone is what has
    shipped since the head was detected automatically, on whatever backend."""
    for backend in ("vulkan", "cuda", "cpu"):
        assert speculation.resolve(None, backend) is speculation.DEFAULT


def test_the_copy_profile_is_allowed_on_the_backend_it_was_measured_on():
    assert speculation.resolve("copy", "cuda") is speculation.COPY


def test_the_copy_profile_is_refused_on_vulkan():
    """The default install is Vulkan, so a backend-blind "go faster" switch
    would be a footgun aimed at almost everybody who has one."""
    with pytest.raises(speculation.Unavailable) as exc:
        speculation.resolve("copy", "vulkan")
    assert "vulkan" in str(exc.value)


def test_a_refusal_carries_the_measurement_that_says_no():
    """Without the numbers this is an opinion, and the next person to read it
    will try it anyway -- on the backend where it loses."""
    with pytest.raises(speculation.Unavailable) as exc:
        speculation.resolve("copy", "vulkan")
    said = str(exc.value)
    assert "0.84-0.90x" in said, "what stacking ngram on MTP actually costs"
    assert "22%" in said, "what widening the draft actually costs"
    assert "wider batch" in said, "and the mechanism, so it generalises"


def test_an_unmeasured_backend_is_a_refusal_not_a_shrug():
    """A CPU build is not a backend any of this was measured on. Applying a
    CUDA-measured setting there is a guess wearing a measurement's name."""
    with pytest.raises(speculation.Unavailable):
        speculation.resolve("copy", "cpu")


def test_an_unknown_name_lists_the_ones_that_exist():
    with pytest.raises(speculation.Unavailable) as exc:
        speculation.resolve("fast", "cuda")
    assert "default" in str(exc.value) and "copy" in str(exc.value)


def test_every_profile_that_is_refused_somewhere_says_why():
    """A profile added later with no `elsewhere` would raise a refusal with a
    blank second line, which reads as a bug rather than as a reason."""
    for profile in speculation.PROFILES.values():
        if profile.backends is not None:
            assert profile.elsewhere, f"{profile.name} refuses without a reason"


def test_every_profile_is_summarised_for_the_help_text():
    for profile in speculation.PROFILES.values():
        assert profile.summary
    assert "copy" in speculation.names()
