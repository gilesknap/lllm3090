"""Background model downloads with real progress.

Deliberately a plain HTTP range-resumable GET rather than the HuggingFace
client: it gives exact byte counts for the panel's progress bar, resumes a part
file after a cancel or a crash, and keeps a 15 GB download out of the request
handler that started it.
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import config

CHUNK = 4 * 1024 * 1024
USER_AGENT = "llm3090/0.1 (+https://github.com/gilesknap/llm3090)"


@dataclass
class Download:
    """State of one in-flight or finished download."""

    id: str
    name: str
    repo: str
    file: str
    target: Path
    total: int = 0
    done: int = 0
    state: str = "queued"  # queued | downloading | complete | error | cancelled
    detail: str = ""
    started: float = field(default_factory=time.time)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def percent(self) -> float:
        return round(100.0 * self.done / self.total, 1) if self.total else 0.0

    @property
    def rate_mib_s(self) -> float:
        elapsed = max(time.time() - self.started, 0.001)
        return round(self.done / elapsed / (1024 * 1024), 1)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "file": self.file,
            "state": self.state,
            "detail": self.detail,
            "percent": self.percent,
            "done_gb": round(self.done / 1e9, 2),
            "total_gb": round(self.total / 1e9, 2),
            "rate_mib_s": self.rate_mib_s,
        }


#: Live and recently finished downloads, keyed by catalogue id.
_downloads: dict[str, Download] = {}
_lock = threading.Lock()


def url_for(repo: str, file: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{file}"


def _run(dl: Download) -> None:
    part = dl.target.with_suffix(dl.target.suffix + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    try:
        dl.state = "downloading"
        resume = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url_for(dl.repo, dl.file))
        req.add_header("User-Agent", USER_AGENT)
        if resume:
            req.add_header("Range", f"bytes={resume}-")
        with urllib.request.urlopen(req, timeout=60) as r:
            length = int(r.headers.get("Content-Length") or 0)
            dl.total = resume + length
            dl.done = resume
            mode = "ab" if resume else "wb"
            with part.open(mode) as f:
                while True:
                    if dl._cancel.is_set():
                        dl.state = "cancelled"
                        dl.detail = "cancelled; part file kept for resume"
                        return
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    dl.done += len(chunk)
        part.rename(dl.target)
        dl.state = "complete"
        dl.detail = f"saved to {dl.target}"
    except urllib.error.HTTPError as e:
        dl.state = "error"
        dl.detail = f"HTTP {e.code} for {url_for(dl.repo, dl.file)}"
    except Exception as e:  # noqa: BLE001 - surfaced to the user verbatim
        dl.state = "error"
        dl.detail = str(e)


def start(entry: dict) -> Download:
    """Begin downloading a catalogue entry. Returns immediately."""
    with _lock:
        existing = _downloads.get(entry["id"])
        if existing and existing.state in {"queued", "downloading"}:
            return existing
        target = config.MODELS_DIR / entry["name"] / entry["file"]
        dl = Download(
            id=entry["id"],
            name=entry["name"],
            repo=entry["repo"],
            file=entry["file"],
            target=target,
        )
        _downloads[entry["id"]] = dl
    threading.Thread(target=_run, args=(dl,), daemon=True).start()
    return dl


def cancel(model_id: str) -> bool:
    dl = _downloads.get(model_id)
    if dl and dl.state in {"queued", "downloading"}:
        dl._cancel.set()
        return True
    return False


def all_downloads() -> list[dict]:
    return [d.as_dict() for d in _downloads.values()]
