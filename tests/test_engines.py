"""What a build has to prove before it is allowed to run or be measured.

The fetch is the one place this project takes bytes off the internet and then
executes them, so the interesting cases are all refusals. None of these touch
the network: the release API and the download are both faked, because a test
that needs GitHub to be reachable stops being run.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import urllib.error

import pytest

from lllm3090 import config, engines

PINNED = "b10628"
CANDIDATE = "b10715"


def _archive(payload: bytes = b"#!/bin/sh\n") -> bytes:
    """A tarball shaped like the real one: one directory holding llama-server."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("build/bin/llama-server")
        info.size = len(payload)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


@pytest.fixture
def upstream(monkeypatch, tmp_path):
    """A fake release API and download, wired to the same bytes by default."""

    state = {"archive": _archive(), "published": None, "api_fails": False}
    state["published"] = hashlib.sha256(state["archive"]).hexdigest()

    def fake_published(build, timeout=30.0):
        if state["api_fails"]:
            raise engines.BuildError("rate limited")
        return state["published"]

    def fake_retrieve(url, filename):
        with open(filename, "wb") as f:
            f.write(state["archive"])

    monkeypatch.setattr(engines, "published_digest", fake_published)
    monkeypatch.setattr(engines.urllib.request, "urlretrieve", fake_retrieve)
    monkeypatch.setattr(config, "ENGINES_DIR", tmp_path / "engines")
    return state


def test_a_verified_build_unpacks_and_reports_what_it_got(upstream, tmp_path):
    target = tmp_path / "out"
    digest = engines.fetch(CANDIDATE, target)
    assert (target / "llama-server").exists()
    assert digest == upstream["published"]
    # Returned so a candidate that wins can be promoted by committing it.
    assert len(digest) == 64


def test_a_replaced_asset_is_refused_before_it_is_downloaded(upstream, tmp_path):
    """The one failure a published digest cannot report on its own.

    If the bytes behind a tag are swapped, the API reports the new digest quite
    happily. Only the digest recorded in this repository disagrees -- so when
    the two differ the fetch must stop rather than trust the fresher number.
    """
    upstream["published"] = "b" * 64
    with pytest.raises(engines.BuildError, match="have been replaced"):
        engines.fetch(PINNED, tmp_path / "out", expect="a" * 64)
    assert not (tmp_path / "out").exists()


def test_corrupted_bytes_are_refused(upstream, tmp_path):
    """Digest agreed in advance, different bytes arrived."""
    upstream["archive"] = _archive(b"not the same file at all\n")
    with pytest.raises(engines.BuildError, match="Checksum mismatch"):
        engines.fetch(CANDIDATE, tmp_path / "out")


def test_a_recorded_digest_survives_an_unreachable_api(upstream, tmp_path):
    """The release API is rate limited unauthenticated, and an install that
    fails because someone else used the quota is a bad install. The recorded
    digest is the stronger check anyway, so it stands alone."""
    upstream["api_fails"] = True
    digest = engines.fetch(
        PINNED, tmp_path / "out", expect=hashlib.sha256(upstream["archive"]).hexdigest()
    )
    assert (tmp_path / "out" / "llama-server").exists()
    assert digest == upstream["published"]


def test_an_unpinned_build_may_not_skip_verification(upstream, tmp_path):
    """Without a recorded digest the API is the only check there is, so losing
    it is fatal. An unverified build is not worth measuring against."""
    upstream["api_fails"] = True
    with pytest.raises(engines.BuildError, match="rate limited"):
        engines.fetch(CANDIDATE, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_benchmark_build_never_lands_on_the_installed_one(monkeypatch, tmp_path):
    """The whole point of the separate directory: deciding whether to move the
    pin means running the candidate and the incumbent, which is impossible if
    fetching one replaces the other."""
    monkeypatch.setattr(config, "ENGINES_DIR", tmp_path / "engines")
    monkeypatch.setattr(config, "LLAMA_DIR", tmp_path / "llama.cpp")
    for build in (PINNED, CANDIDATE):
        assert engines.bench_dir(build) != config.LLAMA_DIR
        assert config.LLAMA_DIR not in engines.bench_dir(build).parents
    assert engines.bench_dir(PINNED) != engines.bench_dir(CANDIDATE)


def test_the_asset_name_follows_the_tag():
    assert engines.asset(CANDIDATE) == "llama-b10715-bin-ubuntu-vulkan-x64.tar.gz"
    assert engines.url(CANDIDATE).endswith(f"/{CANDIDATE}/{engines.asset(CANDIDATE)}")


class _Response:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _api(monkeypatch, body: bytes):
    monkeypatch.setattr(
        engines.urllib.request, "urlopen", lambda req, timeout=30.0: _Response(body)
    )


def test_the_published_digest_is_read_for_the_right_asset(monkeypatch):
    _api(monkeypatch, (
        b'{"assets": ['
        b'{"name": "llama-b10715-bin-win-vulkan-x64.zip", "digest": "sha256:'
        + b"f" * 64 + b'"},'
        b'{"name": "llama-b10715-bin-ubuntu-vulkan-x64.tar.gz", "digest": "sha256:'
        + b"a" * 64 + b'"}]}'
    ))
    assert engines.published_digest(CANDIDATE) == "a" * 64


def test_a_digest_that_is_not_there_is_an_error_not_a_shrug(monkeypatch):
    """Returning None here would land as "verified against nothing"."""
    _api(monkeypatch, b'{"assets": [{"name": "other.tar.gz", "digest": "sha256:x"}]}')
    with pytest.raises(engines.BuildError, match="publishes no asset"):
        engines.published_digest(CANDIDATE)


def test_an_unreachable_api_is_reported_as_such(monkeypatch):
    def boom(req, timeout=30.0):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(engines.urllib.request, "urlopen", boom)
    with pytest.raises(engines.BuildError, match="could not read the release"):
        engines.published_digest(CANDIDATE)
