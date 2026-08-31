"""Which physical disk backs a path, and whether a faster one is going spare.

Where the checkpoints live is not a detail. A cold load reads the whole file
before the engine answers, and on the reference machine the same 18.2 GB
checkpoint takes **62 s** from a SATA SSD and **15-26 s** from a Gen4 NVMe --
with a floor of about 10 s that is dequantisation and the VRAM upload and that
no disk can help with.

That gap is invisible from inside the program: ``~/models`` looks the same on
either, ``df`` reports the same free space, and the symptom is "switching models
feels slow" with nothing in the log. So it is checked at setup, once, when
there is still a decision to make.

Everything here reads ``/sys`` rather than shelling out to ``lsblk`` or ``findmnt``,
so it works in a container that has neither.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

#: Where the kernel maps device numbers to devices.
SYS_DEV_BLOCK = Path("/sys/dev/block")

#: Where whole disks are listed.
SYS_BLOCK = Path("/sys/block")


def nearest_existing(path: Path | str) -> Path | None:
    """The closest ancestor of ``path`` that exists, or ``path`` itself.

    A directory that has not been created yet still has a filesystem: it is
    whichever one holds its parent. Without this the check is useless in the
    only situation it was written for -- a fresh install, where ``~/models``
    does not exist, every question about it answers "cannot tell", and the
    warning stays silent on exactly the machine that needed it.
    """
    here = Path(path).expanduser().absolute()
    while not here.exists():
        if here.parent == here:
            return None
        here = here.parent
    return here


def backing_disk(path: Path | str, sys_root: Path | None = None) -> str | None:
    """Kernel name of the whole disk behind ``path`` -- ``nvme0n1``, ``sda``.

    A path that does not exist yet is classified by the nearest ancestor that
    does, because that is the filesystem it will be created on.

    ``None`` when there is no block device to name: a network mount, a tmpfs,
    or a path with no existing ancestor at all. Callers must treat that as
    "cannot tell" rather than as "slow", because refusing to proceed on a
    filesystem this cannot classify would break every container and NFS home
    directory.
    """
    dev_block = (sys_root or Path("/sys")) / "dev/block"
    existing = nearest_existing(path)
    if existing is None:
        return None
    try:
        st = os.stat(existing)
    except OSError:
        return None
    node = dev_block / f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}"
    try:
        resolved = node.resolve(strict=True)
    except OSError:
        return None
    # A partition's sysfs directory carries a `partition` file and sits inside
    # its disk's; a whole disk has neither.
    if (resolved / "partition").exists():
        resolved = resolved.parent
    # Device mapper (LVM, LUKS) names itself dm-N and lists what it is built
    # from. One level is enough for an ordinary single-disk volume group;
    # anything striped across several disks has no single answer and is left
    # alone rather than guessed at.
    if resolved.name.startswith("dm-"):
        slaves = sorted((resolved / "slaves").iterdir()) if (
            resolved / "slaves"
        ).is_dir() else []
        if len(slaves) != 1:
            return None
        resolved = slaves[0].resolve()
        if (resolved / "partition").exists():
            resolved = resolved.parent
    return resolved.name


def is_nvme(disk: str | None) -> bool:
    """Whether a disk name is an NVMe device."""
    return bool(disk) and str(disk).startswith("nvme")


def nvme_disks(sys_root: Path | None = None) -> list[str]:
    """Every NVMe whole-disk this machine has."""
    block = (sys_root or Path("/sys")) / "block"
    if not block.is_dir():
        return []
    return sorted(d.name for d in block.iterdir() if d.name.startswith("nvme"))


def writable_mount_on(disk: str, sys_root: Path | None = None) -> Path | None:
    """A directory the user can write to that is backed by ``disk``.

    Used only to make the advice actionable -- naming a device is no help if
    the reader then has to work out where it is mounted. Returns the first
    candidate that is on the right disk, preferring one already writable.
    """
    candidates = [Path.home(), Path("/srv"), Path("/opt"), Path("/var"), Path("/")]
    fallback = None
    for base in candidates:
        if backing_disk(base, sys_root) != disk:
            continue
        if os.access(base, os.W_OK):
            return base
        fallback = fallback or base
    return fallback


def free_gb(path: Path | str) -> float:
    """Free space on the filesystem that holds -- or will hold -- ``path``."""
    existing = nearest_existing(path)
    if existing is None:
        return 0.0
    try:
        return shutil.disk_usage(existing).free / 1e9
    except OSError:
        return 0.0


def slow_disk_warning(models_dir: Path, sys_root: Path | None = None) -> str | None:
    """Why this models directory is on the wrong disk, or ``None`` if it is fine.

    Fine means any of: it is already on an NVMe, the machine has no NVMe to
    move to, or the filesystem cannot be classified at all. Only a machine that
    *has* a faster disk and is not using it has a problem worth stopping for.
    """
    nvmes = nvme_disks(sys_root)
    if not nvmes:
        return None
    disk = backing_disk(models_dir, sys_root)
    if disk is None or is_nvme(disk):
        return None

    disks = ", ".join(f"/dev/{n}" for n in nvmes)
    has = "an NVMe" if len(nvmes) == 1 else "NVMe disks"
    cost = (
        f"{models_dir} is on /dev/{disk}, but this machine has {has}: {disks}\n"
        "\n"
        "A cold model load reads the whole checkpoint before the engine "
        "answers. Measured on\n"
        "the reference machine with the same 18.2 GB file: 62 s from a SATA SSD "
        "against 15-26 s\n"
        "from a Gen4 NVMe. You will pay that on every model switch.\n"
    )

    target = None
    for nvme in nvmes:
        mount = writable_mount_on(nvme, sys_root)
        if mount is not None:
            target = mount / "models"
            break

    if target is None:
        move = (
            "Mount the NVMe somewhere you can write, then:\n"
            "    lllm3090 setup --model-folder <that path>/models\n"
        )
    else:
        move = (
            "To move it:\n"
            f"    lllm3090 setup --model-folder {target}\n"
            "\n"
            "If that directory's parent needs root, setup will say so and give "
            "you the one command\n"
            "it needs.\n"
        )

    keep = (
        "\nTo keep the models where they are, say so explicitly:\n"
        f"    lllm3090 setup --model-folder {models_dir}"
    )
    return cost + "\n" + move + keep
