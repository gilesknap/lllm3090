"""Checks that answer "will this machine run the stack" before it is asked to.

Each check returns (ok, message). They are deliberately specific: this project
targets one GPU and one OS family, and a vague pass on the wrong hardware would
invalidate every figure in the model catalogue.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import config


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
    name = _smi("name")
    if name is None:
        return False, "nvidia-smi not found or no GPU visible"
    cap = _smi("compute_cap")
    total = _smi("memory.total")
    if cap != config.TARGET_COMPUTE_CAPABILITY:
        return False, (
            f"{name} is compute capability {cap}; this project is scoped to "
            f"{config.TARGET_COMPUTE_CAPABILITY} (RTX 3090). It may work, but every "
            "size and speed figure in the model catalogue would be wrong."
        )
    if total and int(total) < config.TARGET_VRAM_MIB * 0.95:
        return False, f"{name} has {total} MiB; the catalogue assumes 24 GB"
    return True, f"{name} ({total} MiB, compute {cap})"


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
            "(run: llm3090 install-engine)"
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


CHECKS = [
    ("os", check_os),
    ("gpu", check_gpu),
    ("driver", check_driver),
    ("vulkan", check_vulkan),
    ("engine", check_engine),
    ("models dir", check_models_dir),
]


def run_all() -> list[tuple[str, bool, str]]:
    return [(name, *fn()) for name, fn in CHECKS]
