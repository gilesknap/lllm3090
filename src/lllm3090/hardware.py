"""Which GPU is in the machine, and what may be claimed for it.

The catalogue makes two kinds of claim and they travel differently. Whether a
model *fits*, and what context it leaves, is arithmetic over capacity -- it
holds on any card once you know how much memory it has. How *fast* it runs is a
measurement, true of the card it was taken on and nowhere else.

So a profile carries capacity (used) and bandwidth (recorded, never used to
scale a measured speed).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from importlib import resources

import yaml

from . import config


@dataclass(frozen=True)
class Profile:
    """A GPU this project knows how to reason about."""

    id: str
    name: str
    compute_capability: str
    vram_mib: int
    bandwidth_gbs: int
    #: True for the one card the catalogue's speeds were measured on.
    measured: bool = False
    notes: str = ""
    #: True when this was synthesised from an unrecognised GPU rather than
    #: matched against the shipped list.
    detected: bool = False
    #: False when no GPU could be found at all and the capacity is borrowed so
    #: the catalogue can still be inspected. Nothing about such a profile
    #: describes a card in this machine.
    present: bool = True

    def usable_vram_mib(self, desktop: bool = True) -> int:
        """VRAM available for weights plus KV cache."""
        budget = self.vram_mib - config.WORKSPACE_RESERVE_MIB
        if desktop:
            budget -= config.DESKTOP_RESERVE_MIB
        return budget


def load_profiles() -> list[Profile]:
    raw = resources.files("lllm3090.data").joinpath("profiles.yaml").read_text()
    return [Profile(**p) for p in yaml.safe_load(raw)["profiles"]]


def _normalise(name: str) -> str:
    """A GPU name reduced to what can be compared across nvidia-smi versions."""
    return "".join(c for c in name.lower() if c.isalnum())


def _smi(query: str) -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        return subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0].strip()
    except Exception:
        return None


def detect() -> Profile:
    """The profile for the GPU in this machine.

    An unrecognised card gets a profile built from what ``nvidia-smi`` reports,
    marked ``detected`` and never ``measured``: its capacity is known, so fit
    and context are computed correctly, while nothing claims its speed.

    With no GPU at all -- CI, a container -- capacity is borrowed from the
    reference profile so the catalogue can still be inspected, but the profile
    is marked ``present=False`` and never ``measured``: there is no card here,
    so nothing may be said about how fast one would be.
    """
    name = _smi("name")
    cap = _smi("compute_cap")
    total = _smi("memory.total")
    profiles = load_profiles()

    if name is None or total is None:
        ref = next(p for p in profiles if p.id == config.REFERENCE_PROFILE)
        return Profile(
            id="no-gpu",
            name="no GPU detected",
            compute_capability="unknown",
            vram_mib=ref.vram_mib,
            bandwidth_gbs=0,
            measured=False,
            notes=(
                "nvidia-smi reported no GPU. Capacity is borrowed from the "
                f"{ref.name} so the catalogue can be inspected; none of these "
                "figures describes a card in this machine."
            ),
            present=False,
        )

    vram = int(total)
    for p in profiles:
        # The name is what identifies the card. A 3090 Ti carries the same 24
        # GiB and the same compute capability as a 3090 and is not the card the
        # speeds were taken on, so capacity alone must not claim a match.
        if _normalise(p.name) != _normalise(name):
            continue
        same_size = abs(p.vram_mib - vram) <= 512
        if same_size and (cap is None or p.compute_capability == cap):
            return p

    return Profile(
        id="detected",
        name=name,
        compute_capability=cap or "unknown",
        vram_mib=vram,
        # Unknown: recorded honestly rather than guessed, and unused anyway.
        bandwidth_gbs=0,
        measured=False,
        notes="Not a profile shipped with lllm3090. Fit is computed from the "
              "reported memory; speeds in the catalogue were measured elsewhere.",
        detected=True,
    )


def reference() -> Profile:
    """The profile the catalogue's speeds were measured on."""
    return next(p for p in load_profiles() if p.measured)
