"""Which disk backs the models directory, and whether that is worth stopping for.

The probing reads `/sys`, so the tests build a fake one. A CI runner's real
disks are not this machine's, and a check that only passes on the author's
hardware is the thing this file exists to avoid.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest
import typer

from lllm3090 import cli, config, storage


def fake_sys(tmp_path: Path, disks: dict[str, list[str]]) -> Path:
    """A `/sys` with these whole disks, each holding these partitions.

    `{"nvme0n1": ["nvme0n1p2"], "sda": ["sda2"]}` gives the layout of the
    reference machine.
    """
    root = tmp_path / "sys"
    for disk, parts in disks.items():
        (root / "block" / disk).mkdir(parents=True)
        for part in parts:
            p = root / "block" / disk / part
            p.mkdir()
            (p / "partition").write_text("2\n")
    return root


def link(sys_root: Path, major: int, minor: int, target: Path) -> None:
    """Point /sys/dev/block/M:m at a device directory, as the kernel does."""
    dev = sys_root / "dev/block"
    dev.mkdir(parents=True, exist_ok=True)
    (dev / f"{major}:{minor}").symlink_to(target)


def on_device(target: Path, major: int, minor: int):
    """Patch `os.stat` for one path only.

    Patching it wholesale is tempting and wrong: `backing_disk` asks whether a
    `partition` file exists, `Path.exists()` is `os.stat`, and a blanket mock
    makes every such question answer yes -- so a whole disk is mistaken for a
    partition and the test passes for the wrong reason. This one lies about the
    device number of a single path and tells the truth about everything else.
    """
    real = os.stat

    def fake(path, *args, **kwargs):
        if str(path) == str(target):
            return os.stat_result((0o40755, 0, os.makedev(major, minor)) + (0,) * 7)
        return real(path, *args, **kwargs)

    return mock.patch.object(os, "stat", side_effect=fake)


def test_a_partition_resolves_to_its_whole_disk(tmp_path):
    """`/dev/nvme0n1p2` is not a disk you can shop for; `nvme0n1` is."""
    root = fake_sys(tmp_path, {"nvme0n1": ["nvme0n1p2"]})
    link(root, 259, 2, root / "block/nvme0n1/nvme0n1p2")
    target = tmp_path / "models"
    target.mkdir()
    with on_device(target, 259, 2):
        assert storage.backing_disk(target, root) == "nvme0n1"


def test_a_whole_disk_is_returned_unchanged(tmp_path):
    root = fake_sys(tmp_path, {"sda": []})
    link(root, 8, 0, root / "block/sda")
    with on_device(tmp_path, 8, 0):
        assert storage.backing_disk(tmp_path, root) == "sda"


def test_what_cannot_be_classified_is_none_rather_than_slow(tmp_path):
    """NFS, tmpfs and overlayfs have no block device, and must not be refused.

    Treating "cannot tell" as "slow" would make setup fail inside every
    container and on every network home directory, for a performance hint.
    """
    root = fake_sys(tmp_path, {"nvme0n1": []})
    with on_device(tmp_path, 0, 42):
        assert storage.backing_disk(tmp_path, root) is None
    assert storage.backing_disk("/does/not/exist", root) is None


def test_no_nvme_on_the_machine_is_never_a_complaint(tmp_path):
    """Nothing to move to means nothing to say."""
    root = fake_sys(tmp_path, {"sda": ["sda2"]})
    link(root, 8, 2, root / "block/sda/sda2")
    with on_device(tmp_path, 8, 2):
        assert storage.slow_disk_warning(tmp_path, root) is None


def test_an_nvme_going_spare_is_worth_stopping_for(tmp_path):
    root = fake_sys(tmp_path, {"nvme0n1": ["nvme0n1p2"], "sda": ["sda2"]})
    link(root, 8, 2, root / "block/sda/sda2")
    with on_device(tmp_path, 8, 2):
        warning = storage.slow_disk_warning(tmp_path, root)
    assert warning is not None
    assert "/dev/sda" in warning and "/dev/nvme0n1" in warning
    # It has to be actionable, not merely correct.
    assert "--model-folder" in warning
    assert "62 s" in warning, "say what it costs, or nobody acts on it"


def test_already_on_an_nvme_says_nothing(tmp_path):
    root = fake_sys(tmp_path, {"nvme0n1": ["nvme0n1p2"]})
    link(root, 259, 2, root / "block/nvme0n1/nvme0n1p2")
    with on_device(tmp_path, 259, 2):
        assert storage.slow_disk_warning(tmp_path, root) is None


# ---------------------------------------------------------------------------
# What setup does with the answer
# ---------------------------------------------------------------------------


def test_an_explicit_folder_is_honoured_on_any_disk(tmp_path):
    """The check exists to stop an unconsidered default, not to overrule a choice.

    Someone with a 20 TB array and a 500 GB NVMe may want the array, and this
    must not argue with them every time they re-run setup.
    """
    chosen = tmp_path / "elsewhere"
    with mock.patch.object(config, "MODELS_DIR", chosen), \
         mock.patch.object(storage, "slow_disk_warning") as warn:
        cli._configure_models_folder(str(chosen))
    warn.assert_not_called()
    assert chosen.is_dir()


def test_a_chosen_folder_is_reached_through_the_default_path(tmp_path):
    """Everything else reads config.MODELS_DIR, so the choice becomes a symlink.

    No environment variable to export, no edit to the service unit, and nothing
    to remember at the next upgrade.
    """
    default, chosen = tmp_path / "models", tmp_path / "fast" / "models"
    with mock.patch.object(config, "MODELS_DIR", default):
        cli._configure_models_folder(str(chosen))
    assert default.is_symlink()
    assert default.resolve() == chosen.resolve()


def test_a_default_that_already_holds_models_is_not_moved_silently(tmp_path):
    """Setup will not shift 180 GB behind your back, nor delete it."""
    default, chosen = tmp_path / "models", tmp_path / "fast"
    default.mkdir()
    (default / "Some-Model").mkdir()
    with mock.patch.object(config, "MODELS_DIR", default), \
         pytest.raises(typer.Exit):
        cli._configure_models_folder(str(chosen))
    assert (default / "Some-Model").is_dir(), "it must still be there"
    assert not default.is_symlink()


def test_an_empty_default_is_replaced_without_complaint(tmp_path):
    """A fresh install has an empty ~/models, and that is not an obstacle."""
    default, chosen = tmp_path / "models", tmp_path / "fast"
    default.mkdir()
    with mock.patch.object(config, "MODELS_DIR", default):
        cli._configure_models_folder(str(chosen))
    assert default.is_symlink()


def test_rerunning_with_the_same_folder_is_a_no_op(tmp_path):
    """`setup` is the repair command; it has to be safe to run twice."""
    default, chosen = tmp_path / "models", tmp_path / "fast"
    with mock.patch.object(config, "MODELS_DIR", default):
        cli._configure_models_folder(str(chosen))
        cli._configure_models_folder(str(chosen))
    assert default.is_symlink() and default.resolve() == chosen.resolve()


def test_a_slow_default_stops_setup_with_an_explanation(tmp_path, capsys):
    """The failure has to carry the remedy, or it is just an obstacle."""
    with mock.patch.object(config, "MODELS_DIR", tmp_path / "models"), \
         mock.patch.object(
             storage, "slow_disk_warning",
             return_value=(
                 "it is on a spinning disk\n"
                 "    lllm3090 setup --model-folder /fast"
             ),
         ), pytest.raises(typer.Exit) as exit_info:
        cli._configure_models_folder(None)
    assert exit_info.value.exit_code == 1
    out = capsys.readouterr().out
    assert "not on an NVMe" in out
    assert "--model-folder" in out, "a refusal without a remedy is not actionable"
