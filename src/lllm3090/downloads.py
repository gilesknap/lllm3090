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
USER_AGENT = "lllm3090/0.1 (+https://github.com/gilesknap/lllm3090)"


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
    #: Files that belong with the weights -- a multimodal projector. Fetched
    #: after them, into the same directory, under the same cancel flag.
    extras: list[str] = field(default_factory=list)
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


def _size_of(repo: str, file: str) -> int:
    """Content-Length without fetching the body, so the bar knows the whole job."""
    req = urllib.request.Request(url_for(repo, file), method="HEAD")
    req.add_header("User-Agent", USER_AGENT)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def _fetch(dl: Download, file: str, target: Path, base: int) -> bool:
    """One file, resuming a part file if there is one. ``base`` is bytes already
    finished in this download, so progress runs across the whole job."""
    part = target.with_suffix(target.suffix + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    resume = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url_for(dl.repo, file))
    req.add_header("User-Agent", USER_AGENT)
    if resume:
        req.add_header("Range", f"bytes={resume}-")
    with urllib.request.urlopen(req, timeout=60) as r:
        dl.done = base + resume
        mode = "ab" if resume else "wb"
        with part.open(mode) as f:
            while True:
                if dl._cancel.is_set():
                    dl.state = "cancelled"
                    dl.detail = "cancelled; part file kept for resume"
                    return False
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                dl.done += len(chunk)
    part.rename(target)
    return True


def _run(dl: Download) -> None:
    jobs = [(dl.file, dl.target)] + [
        (e, dl.target.parent / e) for e in dl.extras
    ]
    try:
        dl.state = "downloading"
        # Size the whole job up front: a bar that resets when the projector
        # starts reads as a fault rather than as the second of two files.
        dl.total = sum(_size_of(dl.repo, f) for f, _ in jobs)
        base = 0
        for file, target in jobs:
            dl.file = file
            if target.exists():
                base += target.stat().st_size
                dl.done = base
                continue
            if not _fetch(dl, file, target, base):
                return
            base = dl.done
        dl.state = "complete"
        dl.detail = f"saved to {dl.target.parent}"
    except urllib.error.HTTPError as e:
        dl.state = "error"
        dl.detail = f"HTTP {e.code} for {url_for(dl.repo, dl.file)}"
    except Exception as e:
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
            extras=[entry["mmproj"]] if entry.get("mmproj") else [],
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


def resume_interrupted(catalog_entries: list[dict]) -> list[str]:
    """Restart downloads left half-finished by a panel restart.

    Downloads are threads, so stopping the panel -- which systemd does on every
    upgrade -- abandons them. The part file survives and the range request
    resumes from it, but nothing was restarting them, so an interrupted download
    sat at 83% until someone noticed and pressed the button again.

    Returns the ids resumed, for logging.
    """
    def interrupted(path: Path) -> bool:
        """A part file with bytes in it, for a file that has not since landed."""
        part = path.with_name(path.name + ".part")
        return not path.exists() and part.exists() and part.stat().st_size > 0

    resumed = []
    for entry in catalog_entries:
        d = config.MODELS_DIR / entry["name"]
        wanted = [entry["file"]] + ([entry["mmproj"]] if entry.get("mmproj") else [])
        # A model whose weights are complete but whose projector was never
        # fetched still serves text -- it just does not see -- so that is not a
        # download to start behind someone's back. Only interruptions resume.
        if not any(interrupted(d / f) for f in wanted):
            continue
        start(entry)
        resumed.append(entry["id"])
    return resumed
