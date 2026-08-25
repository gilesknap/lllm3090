"""Downloads are threads, so they die with the panel. They must come back."""

from __future__ import annotations

import pytest

from lllm3090 import config, downloads

ENTRY = {
    "id": "demo",
    "name": "Demo-Model",
    "repo": "someone/Demo-GGUF",
    "file": "Demo-Q4_K_S.gguf",
}


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    downloads._downloads.clear()
    return tmp_path


def _part(models_dir, size: int = 1024):
    d = models_dir / ENTRY["name"]
    d.mkdir(parents=True, exist_ok=True)
    part = d / (ENTRY["file"] + ".part")
    part.write_bytes(b"x" * size)
    return part


def test_a_part_file_is_resumed(models_dir, monkeypatch):
    started: list[dict] = []
    monkeypatch.setattr(downloads, "start", lambda e: started.append(e))
    _part(models_dir)
    assert downloads.resume_interrupted([ENTRY]) == ["demo"]
    assert started == [ENTRY]


def test_a_completed_download_is_left_alone(models_dir, monkeypatch):
    """The finished file exists; resuming it would re-download 17 GB."""
    started: list[dict] = []
    monkeypatch.setattr(downloads, "start", lambda e: started.append(e))
    d = models_dir / ENTRY["name"]
    d.mkdir(parents=True)
    (d / ENTRY["file"]).write_bytes(b"complete")
    _part(models_dir)  # a stale part file alongside it must not trigger a fetch
    assert downloads.resume_interrupted([ENTRY]) == []
    assert started == []


def test_nothing_on_disk_means_nothing_to_do(models_dir, monkeypatch):
    started: list[dict] = []
    monkeypatch.setattr(downloads, "start", lambda e: started.append(e))
    assert downloads.resume_interrupted([ENTRY]) == []
    assert started == []


def test_an_empty_part_file_is_not_resumed(models_dir, monkeypatch):
    """Zero bytes is a failed start, not progress worth continuing."""
    started: list[dict] = []
    monkeypatch.setattr(downloads, "start", lambda e: started.append(e))
    _part(models_dir, size=0)
    assert downloads.resume_interrupted([ENTRY]) == []
    assert started == []
