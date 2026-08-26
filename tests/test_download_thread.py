"""The download thread, over a real socket.

Every other download test mocks `downloads.start`, which means the code that
actually moves the bytes -- the range request, the resume arithmetic, the
progress counters, the cancel flag, the two-file job -- has never run under
test. It is also the code most expensive to get wrong: it is what fetches
17 GB, and its failure mode is a progress bar that stops.

So this serves the file from a `ThreadingHTTPServer` on loopback instead.
Nothing here reaches the network: `url_for` is pointed at that server, which
speaks the two things HuggingFace speaks that this code depends on -- a HEAD
that answers `Content-Length`, and a GET that honours `Range` with a 206.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lllm3090 import config, downloads

WEIGHTS = bytes(range(256)) * 40          # 10240 bytes, and every byte checkable
PROJECTOR = b"mmproj" * 100

ENTRY = {
    "id": "demo",
    "name": "Demo-Model",
    "repo": "someone/Demo-GGUF",
    "file": "Demo-Q4_K_S.gguf",
}


class Files(BaseHTTPRequestHandler):
    """Serves `server.files`, with the range support the resume depends on."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):        # a test suite is not a web server log
        pass

    def _body(self) -> bytes | None:
        name = self.path.rsplit("/", 1)[-1]
        return self.server.files.get(name)

    def do_HEAD(self):
        body = self._body()
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

    def do_GET(self):
        body = self._body()
        if body is None:
            self.send_error(404)
            return
        rng = self.headers.get("Range")
        self.server.ranges.append(rng)
        start = 0
        if rng:
            start = int(rng.removeprefix("bytes=").split("-")[0])
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}"
            )
        else:
            self.send_response(200)
        chunk = body[start:]
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Files)
    srv.files = {ENTRY["file"]: WEIGHTS, "mmproj-F16.gguf": PROJECTOR}
    srv.ranges = []
    # poll_interval: shutdown() waits for it, and the default 0.5s is half a
    # second per test spent watching a socket nobody is using.
    threading.Thread(
        target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    ).start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def fetching(server, tmp_path, monkeypatch):
    """A models directory of our own, and a HuggingFace that is on loopback."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    host, port = server.server_address
    monkeypatch.setattr(
        downloads, "url_for", lambda repo, file: f"http://{host}:{port}/{repo}/{file}"
    )
    downloads._downloads.clear()
    yield tmp_path
    downloads._downloads.clear()


def finished(dl: downloads.Download, timeout: float = 10.0) -> downloads.Download:
    """Wait for the thread to reach a state it will not leave."""
    deadline = time.monotonic() + timeout
    while dl.state in {"queued", "downloading"} and time.monotonic() < deadline:
        time.sleep(0.01)
    assert dl.state not in {"queued", "downloading"}, "download never settled"
    return dl


def test_a_download_lands_the_whole_file(fetching):
    dl = finished(downloads.start(ENTRY))
    assert dl.state == "complete"
    target = fetching / ENTRY["name"] / ENTRY["file"]
    assert target.read_bytes() == WEIGHTS
    assert not target.with_suffix(target.suffix + ".part").exists()
    assert dl.total == len(WEIGHTS)
    assert dl.done == len(WEIGHTS)
    assert dl.percent == 100.0
    assert ENTRY["name"] in dl.detail


def test_a_part_file_is_resumed_from_where_it_stopped(fetching, server):
    """The bytes already on disk are not fetched again -- the whole reason the
    part file is kept when a download is cancelled or the panel restarts."""
    d = fetching / ENTRY["name"]
    d.mkdir(parents=True)
    (d / (ENTRY["file"] + ".part")).write_bytes(WEIGHTS[:4096])

    dl = finished(downloads.start(ENTRY))
    assert dl.state == "complete"
    assert (d / ENTRY["file"]).read_bytes() == WEIGHTS
    assert server.ranges == ["bytes=4096-"], "it must ask for the rest, not the lot"
    assert dl.done == len(WEIGHTS), "progress counts the resumed bytes too"


def test_a_part_file_that_is_already_whole_still_completes(fetching, server):
    """The server has nothing left to send; the rename is what was missing."""
    d = fetching / ENTRY["name"]
    d.mkdir(parents=True)
    (d / (ENTRY["file"] + ".part")).write_bytes(WEIGHTS)
    dl = finished(downloads.start(ENTRY))
    assert dl.state == "complete"
    assert (d / ENTRY["file"]).read_bytes() == WEIGHTS


def test_a_projector_is_fetched_into_the_same_directory_under_one_total(fetching):
    """One bar for the job. A total that resets when the second file starts
    reads as a fault rather than as the second of two files."""
    dl = finished(downloads.start({**ENTRY, "mmproj": "mmproj-F16.gguf"}))
    assert dl.state == "complete"
    d = fetching / ENTRY["name"]
    assert (d / ENTRY["file"]).read_bytes() == WEIGHTS
    assert (d / "mmproj-F16.gguf").read_bytes() == PROJECTOR
    assert dl.total == len(WEIGHTS) + len(PROJECTOR)
    assert dl.done == dl.total


def test_a_file_already_on_disk_is_counted_not_refetched(fetching, server):
    """A model whose weights landed but whose projector never did."""
    d = fetching / ENTRY["name"]
    d.mkdir(parents=True)
    (d / ENTRY["file"]).write_bytes(WEIGHTS)
    dl = finished(downloads.start({**ENTRY, "mmproj": "mmproj-F16.gguf"}))
    assert dl.state == "complete"
    assert (d / "mmproj-F16.gguf").read_bytes() == PROJECTOR
    assert dl.done == len(WEIGHTS) + len(PROJECTOR)
    assert server.ranges == [None], "only the projector should have been fetched"


def test_a_cancel_stops_the_thread_and_keeps_the_part_file(fetching, monkeypatch):
    """Cancel has to leave the bytes behind, or resuming is a re-download."""
    monkeypatch.setattr(downloads, "CHUNK", 512)
    dl = downloads.Download(
        id=ENTRY["id"], name=ENTRY["name"], repo=ENTRY["repo"], file=ENTRY["file"],
        target=fetching / ENTRY["name"] / ENTRY["file"],
    )
    downloads._downloads[ENTRY["id"]] = dl
    dl._cancel.set()                      # cancelled before the first chunk lands
    downloads._run(dl)
    assert dl.state == "cancelled"
    assert "resume" in dl.detail
    assert not dl.target.exists()
    assert dl.target.with_suffix(dl.target.suffix + ".part").exists()


def test_a_missing_file_is_reported_as_the_http_error_it_was(fetching):
    dl = finished(downloads.start({**ENTRY, "file": "not-there.gguf"}))
    assert dl.state == "error"
    assert "HTTP 404" in dl.detail
    assert "not-there.gguf" in dl.detail


def test_a_second_click_joins_the_download_already_running(fetching, monkeypatch):
    """Two clicks are one download, not two threads writing one part file."""
    monkeypatch.setattr(downloads, "CHUNK", 256)
    first = downloads.start(ENTRY)
    second = downloads.start(ENTRY)
    assert second is first
    finished(first)
    assert (fetching / ENTRY["name"] / ENTRY["file"]).read_bytes() == WEIGHTS
