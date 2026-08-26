"""Checks that answer "will this machine run the stack" before it is asked to.

Each check returns (ok, message). They are deliberately specific: this project
targets one GPU and one OS family, and a vague pass on the wrong hardware would
invalidate every figure in the model catalogue.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from . import config, hardware


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


def check_os() -> tuple[bool, str]:
    release = Path("/etc/os-release")
    if not release.exists():
        return False, (
            "no /etc/os-release; this installer targets Debian and derivatives"
        )
    fields = dict(
        line.split("=", 1)
        for line in release.read_text().splitlines()
        if "=" in line
    )
    name = fields.get("PRETTY_NAME", "").strip('"')
    ids = f"{fields.get('ID', '')} {fields.get('ID_LIKE', '')}"
    if "debian" not in ids:
        return False, f"{name} is not Debian-family (need apt)"
    if not shutil.which("apt-get"):
        return False, f"{name} has no apt-get"
    return True, name


def check_gpu() -> tuple[bool, str]:
    """Identify the card, and say whether the catalogue's numbers were taken on it.

    An unrecognised GPU is not an error. Fit and context are computed from the
    memory it reports, so they stay correct; only the speeds are unverified,
    and the message says so rather than implying the whole thing is wrong.
    """
    profile = hardware.detect()
    if not profile.present:
        return False, "nvidia-smi not found or no GPU visible"

    summary = (
        f"{profile.name} ({profile.vram_mib} MiB, "
        f"compute {profile.compute_capability})"
    )
    if profile.measured:
        return True, summary
    if profile.detected:
        return True, (
            f"{summary} -- not a profile shipped with lllm3090. Fit is computed "
            "from the memory it reports; the catalogue's speeds were measured on "
            f"a {hardware.reference().name}. Run 'lllm3090 bench' to contribute "
            "real numbers for this card."
        )
    return True, (
        f"{summary} -- fit computed for this card; speeds were measured on a "
        f"{hardware.reference().name}"
    )


def check_driver() -> tuple[bool, str]:
    version = _smi("driver_version")
    if version is None:
        return False, "could not read the NVIDIA driver version"
    try:
        major = int(version.split(".")[0])
    except ValueError:
        return False, f"unparseable driver version {version!r}"
    if major < config.MIN_DRIVER_VERSION:
        return False, (
            f"driver {version} is below the minimum {config.MIN_DRIVER_VERSION}"
        )
    return True, f"driver {version}"


def check_vulkan() -> tuple[bool, str]:
    icd = Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
    if not icd.exists():
        return False, (
            "NVIDIA Vulkan ICD missing. Install the driver's Vulkan support "
            "(apt install libvulkan1 nvidia-driver-libs) -- the engine uses Vulkan, "
            "not CUDA, so this is what talks to the card."
        )
    return True, f"Vulkan ICD at {icd}"


def check_engine() -> tuple[bool, str]:
    binary = config.LLAMA_DIR / "llama-server"
    if not binary.exists():
        return False, (
            f"llama-server not installed at {binary} "
            "(run: lllm3090 install-engine)"
        )
    return True, str(binary)


def check_models_dir() -> tuple[bool, str]:
    d = config.MODELS_DIR
    if not d.exists():
        return True, f"{d} (will be created on first download)"
    free_gb = shutil.disk_usage(d).free / 1e9
    if free_gb < 20:
        return False, f"{d} has only {free_gb:.0f} GB free; models are 5-21 GB each"
    return True, f"{d} ({free_gb:.0f} GB free)"


def check_linger() -> tuple[bool, str]:
    """Whether the panel survives the user logging out.

    The panel is a *user* unit, so it lives inside ``user@UID.service`` and the
    user manager stops when the last session ends -- taking the panel, and the
    engine in its cgroup, with it. ``loginctl enable-linger`` is what keeps it
    running without a session.

    This matters most in exactly the case the headless how-to describes:
    isolating to ``multi-user.target`` ends the graphical session, and without
    lingering the panel goes with it.
    """
    if not shutil.which("loginctl"):
        return True, "no loginctl; not a systemd-logind system"
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    try:
        out = subprocess.run(
            ["loginctl", "show-user", user, "-p", "Linger", "--value"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip()
    except Exception as e:
        return True, f"could not ask loginctl ({e}); assuming it is fine"
    if out == "yes":
        return True, "lingering enabled; the panel survives logout"
    return False, (
        "lingering is off, so the panel stops when your last session ends -- "
        f"including when you switch to a text console. Fix with: "
        f"sudo loginctl enable-linger {user}"
    )


CHECKS = [
    ("os", check_os),
    ("gpu", check_gpu),
    ("driver", check_driver),
    ("vulkan", check_vulkan),
    ("engine", check_engine),
    ("models dir", check_models_dir),
    ("linger", check_linger),
]


def run_all() -> list[tuple[str, bool, str]]:
    return [(name, *fn()) for name, fn in CHECKS]
