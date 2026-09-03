"""What a build has to prove before it is allowed to run or be measured.

The fetch is the one place this project takes bytes off the internet and then
executes them, so the interesting cases are all refusals. None of these touch
the network: the release API and the download are both faked, because a test
that needs GitHub to be reachable stops being run.
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tarfile
import urllib.error
from pathlib import Path

import pytest

from lllm3090 import config, engines

PINNED = "b10715"
CANDIDATE = "b10800"


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
    assert engines.asset(CANDIDATE) == "llama-b10800-bin-ubuntu-vulkan-x64.tar.gz"
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
        b'{"name": "llama-b10800-bin-win-vulkan-x64.zip", "digest": "sha256:'
        + b"f" * 64 + b'"},'
        b'{"name": "llama-b10800-bin-ubuntu-vulkan-x64.tar.gz", "digest": "sha256:'
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


# ---------------------------------------------------------------------------
# Which backend a build was compiled against
# ---------------------------------------------------------------------------

VULKAN_DEVICES = """\
Available devices:
  Vulkan0: NVIDIA GeForce RTX 3090 (24822 MiB, 20233 MiB free)
"""

CUDA_DEVICES = """\
Available devices:
  CUDA0: NVIDIA GeForce RTX 3090 (24125 MiB, 19511 MiB free)
"""


@pytest.fixture
def probe(monkeypatch, tmp_path):
    """An engine directory whose binary answers --list-devices with what you say."""
    engines._BACKENDS.clear()
    directory = tmp_path / "engine"
    directory.mkdir()
    (directory / "llama-server").write_text("#!/bin/sh\n")
    calls: list[list[str]] = []

    class Result:
        stdout = ""
        stderr = ""
        returncode = 0

    def answer(text: str, returncode: int = 0) -> None:
        def fake_run(argv, **kw):
            calls.append(argv)
            out = Result()
            out.stdout = text
            out.returncode = returncode
            return out

        monkeypatch.setattr(engines.subprocess, "run", fake_run)

    answer.directory = directory
    answer.calls = calls
    return answer


def test_a_vulkan_build_says_so(probe):
    probe(VULKAN_DEVICES)
    assert engines.backend(probe.directory) == "vulkan"


def test_a_cuda_build_says_so(probe):
    probe(CUDA_DEVICES)
    assert engines.backend(probe.directory) == "cuda"


def test_a_second_card_does_not_become_a_second_backend(probe):
    """CUDA0 and CUDA1 are two devices and one backend. Matching the printed
    name whole would fail to recognise either."""
    probe(
        "Available devices:\n"
        "  CUDA0: NVIDIA GeForce RTX 3090 (24125 MiB, 19511 MiB free)\n"
        "  CUDA1: NVIDIA GeForce RTX 3090 (24125 MiB, 24000 MiB free)\n"
    )
    assert engines.backend(probe.directory) == "cuda"


def test_a_build_with_no_accelerator_is_cpu(probe):
    probe("Available devices:\n")
    assert engines.backend(probe.directory) == "cpu"


def test_an_engine_that_is_not_installed_is_not_a_crash(tmp_path):
    """`lllm3090 models` runs before `install-engine` on a fresh machine, and
    it asks the planner, which asks this.

    `unknown`, not `cpu`: there is no binary to have a backend, and the two
    answers are read differently downstream."""
    engines._BACKENDS.clear()
    assert engines.backend(tmp_path / "nothing-here") == "unknown"


def test_the_probe_happens_once_per_binary(probe):
    """`catalog.fit` runs once per catalogue entry per `lllm3090 models`. A
    subprocess apiece would be felt."""
    probe(CUDA_DEVICES)
    for _ in range(5):
        assert engines.backend(probe.directory) == "cuda"
    assert len(probe.calls) == 1


def test_a_replaced_binary_is_asked_again(probe):
    """`install-engine --force` swaps the binary underneath a panel that has
    been up for days. An answer cached from the old one would outlive it."""
    probe(VULKAN_DEVICES)
    assert engines.backend(probe.directory) == "vulkan"
    binary = probe.directory / "llama-server"
    os.utime(binary, (0, 0))
    probe(CUDA_DEVICES)
    assert engines.backend(probe.directory) == "cuda"


def test_a_probe_that_could_not_be_run_is_not_remembered(probe, monkeypatch):
    """A timeout under load is a fact about the moment. Caching it would make
    one busy minute permanent, and the answer wrong for the rest of the day."""
    def boom(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 60)

    monkeypatch.setattr(engines.subprocess, "run", boom)
    assert engines.backend(probe.directory) == "unknown"
    probe(CUDA_DEVICES)
    assert engines.backend(probe.directory) == "cuda"


def test_a_probe_that_exits_non_zero_is_unknown_and_is_not_remembered(probe):
    """The failure this module exists to prevent, reached the other way.

    A build whose backend fails to initialise prints its error and exits
    non-zero, and the device list is then empty -- which parses as `cpu`. Left
    uninspected that is wrong once; *cached*, a CUDA engine is `cpu` for the
    life of a panel process, `catalog.fit` drops its KV factor and fixed
    overhead, and the plan promises a window the card cannot hold.
    """
    probe("ggml_cuda_init: failed to initialise CUDA: no device\n", returncode=1)
    assert engines.backend(probe.directory) == "unknown"
    probe(CUDA_DEVICES)
    assert engines.backend(probe.directory) == "cuda", "the failure was remembered"


def test_a_cpu_build_is_an_answer_not_an_absence(probe):
    """`cpu` and `unknown` must not be the same value.

    A CPU-only engine cannot produce the catalogue's GPU speeds on any card,
    so it is not waved through the way an unasked engine is.
    """
    probe("Available devices:\n")
    assert engines.backend(probe.directory) == "cpu"
    assert engines.CPU != engines.UNKNOWN


@pytest.mark.parametrize("build,want", [
    ("b10715", "vulkan"),
    ("b10715-cuda-sm86", "cuda"),
])
def test_the_real_engines_on_this_box_identify_themselves(build, want):
    """The only test here that runs a real llama-server.

    Everything above proves the parsing; this proves the parsing is of the
    right thing. It is the one claim that cannot be made against a fixture --
    that `--list-devices` is where the distinction is actually visible -- and
    it is checkable because this machine has both builds on disk. Skipped
    everywhere else, including CI.
    """
    engines._BACKENDS.clear()
    directory = engines.bench_dir(build)
    if not (directory / "llama-server").exists():
        pytest.skip(f"{build} is not installed on this machine")
    assert engines.backend(directory) == want


# ---------------------------------------------------------------------------
# Building a CUDA engine
# ---------------------------------------------------------------------------


def test_a_toolkit_too_old_to_build_this_is_not_usable():
    """Ubuntu 26.04 ships 13.1, and it cannot compile this against glibc 2.43:
    it declares rsqrt/rsqrtf without an exception specifier where glibc
    declares them noexcept(true), so nvcc refuses every file including
    <math.h>. Offering it would waste a download and fail at the first file."""
    assert not engines.Toolkit(Path("/usr/local/cuda-13.1/bin/nvcc"), (13, 1)).usable
    assert engines.Toolkit(Path("/usr/local/cuda-13.3/bin/nvcc"), (13, 3)).usable
    assert engines.Toolkit(Path("/usr/local/cuda-14.0/bin/nvcc"), (14, 0)).usable


def test_the_newest_usable_toolkit_is_the_one_chosen(monkeypatch):
    """A machine can have several. /usr/local/cuda is an alternatives symlink
    and points at whichever was configured last, so it is not the answer."""
    monkeypatch.setattr(engines, "toolkits", lambda: [
        engines.Toolkit(Path("/usr/local/cuda-14.0/bin/nvcc"), (14, 0)),
        engines.Toolkit(Path("/usr/local/cuda-13.3/bin/nvcc"), (13, 3)),
        engines.Toolkit(Path("/usr/local/cuda-13.1/bin/nvcc"), (13, 1)),
    ])
    assert engines.toolkit().version == (14, 0)


def test_only_an_old_toolkit_is_the_same_as_none(monkeypatch):
    """Answering "yes there is one" here starts a build that cannot finish."""
    monkeypatch.setattr(engines, "toolkits", lambda: [
        engines.Toolkit(Path("/usr/local/cuda-13.1/bin/nvcc"), (13, 1)),
    ])
    assert engines.toolkit() is None


@pytest.mark.parametrize("capability,want", [
    ("8.6", "86"), ("12.0", "120"), ("7.5", "75"),
    ("unknown", None), ("", None), ("8", None),
])
def test_an_architecture_is_derived_from_the_card_or_not_at_all(capability, want):
    assert engines.sm(capability) == want


def test_the_directory_names_the_architecture_it_was_built_for(monkeypatch, tmp_path):
    """The binary is compiled sm_<arch>-real and will not run on another
    architecture at all. Named after the tag alone, a card swap would be an
    engine that dies at load for no stated reason; named this way the absence
    is legible."""
    monkeypatch.setattr(config, "ENGINES_DIR", tmp_path)
    assert engines.cuda_dir("b10715", "86").name == "b10715-cuda-sm86"
    assert engines.cuda_dir("b10715", "120").name == "b10715-cuda-sm120"


def _built(root, name):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "llama-server").write_text("#!/bin/sh\n")
    return directory


def test_a_pin_move_orphans_a_cuda_build_and_something_has_to_notice(
    monkeypatch, tmp_path
):
    """The most likely way this feature rots. A CUDA engine is tied to the
    commit it was compiled from, and moving LLAMA_BUILD does not move it, so
    an upgrade silently leaves a CUDA user on an older engine than everything
    in the catalogue was measured against."""
    monkeypatch.setattr(config, "ENGINES_DIR", tmp_path)
    _built(tmp_path, "b10715-cuda-sm86")
    monkeypatch.setattr(engines, "LLAMA_BUILD", "b10715")
    assert engines.stale_cuda_builds() == []
    monkeypatch.setattr(engines, "LLAMA_BUILD", "b10900")
    assert [p.name for p in engines.stale_cuda_builds()] == ["b10715-cuda-sm86"]


def test_a_downloaded_build_is_not_mistaken_for_a_compiled_one(monkeypatch, tmp_path):
    """`fetch-engine` puts tags in the same directory. Only the compiled ones
    can go stale in the way above -- a downloaded one is re-fetched by tag."""
    monkeypatch.setattr(config, "ENGINES_DIR", tmp_path)
    _built(tmp_path, "b10628")
    _built(tmp_path, "b10715-cuda-sm86")
    assert [p.name for p in engines.cuda_builds()] == ["b10715-cuda-sm86"]


def test_a_half_built_directory_is_not_left_where_an_engine_goes(
    monkeypatch, tmp_path
):
    """A directory holding some of an engine is indistinguishable from one
    holding all of it, and would be launched as though it were complete."""
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(engines.shutil, "which", lambda tool: f"/usr/bin/{tool}")

    class Silent:
        stdout = iter(())

        def wait(self):
            return 0

    monkeypatch.setattr(engines.subprocess, "Popen", lambda *a, **kw: Silent())
    target = tmp_path / "engine"
    with pytest.raises(engines.BuildError, match="no llama-server"):
        engines.build_cuda(target, "86", engines.Toolkit(Path("nvcc"), (13, 3)))
    assert not target.exists()


def test_a_build_without_the_tools_to_do_it_fails_before_the_clone(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(engines.shutil, "which", lambda tool: None)
    with pytest.raises(engines.BuildError, match="cmake|git"):
        engines.build_cuda(
            tmp_path / "engine", "86", engines.Toolkit(Path("nvcc"), (13, 3))
        )


def test_what_cuda_costs_in_window_is_derived_from_the_two_ceilings():
    """So the figure `setup` quotes and the figure a planner would use cannot
    drift apart, and re-measuring is one edit."""
    assert engines.CUDA_CEILING < engines.VULKAN_CEILING
    assert engines.cuda_window_cost() == 14


def test_the_real_toolkits_on_this_box_are_ranked_correctly():
    """The only test here that runs a real nvcc.

    This machine has 13.1 and 13.3 installed and nvcc is on neither PATH nor
    reliably at /usr/local/cuda -- which is the exact shape the search exists
    for. Skipped everywhere else, CI included.
    """
    found = engines.toolkits()
    if not found:
        pytest.skip("no CUDA toolkit on this machine")
    assert found == sorted(found, key=lambda t: t.version, reverse=True)
    chosen = engines.toolkit()
    if chosen is not None:
        assert chosen.version >= engines.MIN_CUDA
        assert chosen.nvcc.exists()
