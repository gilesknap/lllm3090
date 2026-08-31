"""The llama.cpp build this project runs, and the ones it measures against.

llama.cpp tags almost every commit that passes CI and attaches prebuilt
binaries to each tag, so a build number is a complete identifier: tag, commit
and asset all follow from it. Nothing here tracks a branch and nothing resolves
"latest" -- that would silently change the thing every figure in the catalogue
was measured against.

**Two digests, and they are not the same claim.** ``LLAMA_SHA256`` below is
recorded in the repository: reviewed in a diff, outside the serving party's
control, and therefore the only one that can notice a tag whose bytes changed
after the fact. The digest the release API publishes cannot -- it would move
with them -- but it does say what GitHub is serving right now, which is what
lets a build nobody has pinned yet be verified as strictly as one that is.

So a fetch verifies against both where both exist, and promoting a candidate
build to the pin is the act of committing the digest it printed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from . import config

#: The build the product runs.
LLAMA_BUILD = "b10715"

#: Its digest. See the module docstring for why this is written down rather
#: than only asked for at fetch time.
LLAMA_SHA256 = "1246c764f630f1cfc7c0921353a9719603c5d9ccfa7ced621bac216ffd9b2d87"

#: Vulkan rather than CUDA, for the reason in docs/tutorials/installation.md:
#: upstream publishes no CUDA archive for Linux, only for Windows.
FLAVOUR = "bin-ubuntu-vulkan-x64"

RELEASES = "https://github.com/ggml-org/llama.cpp/releases/download"
API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags"
USER_AGENT = "lllm3090/0.1 (+https://github.com/gilesknap/lllm3090)"


class BuildError(RuntimeError):
    """A build could not be fetched, or could not be trusted once it was."""


def asset(build: str) -> str:
    return f"llama-{build}-{FLAVOUR}.tar.gz"


def url(build: str) -> str:
    return f"{RELEASES}/{build}/{asset(build)}"


def bench_dir(build: str) -> Path:
    """Where a build lives when it is here to be measured rather than to be run.

    Kept apart from ``config.LLAMA_DIR`` so that benchmarking a candidate never
    disturbs the engine the panel launches -- which matters because deciding
    whether to move the pin means running the candidate *and* the incumbent.
    """
    return config.ENGINES_DIR / build


def published_digest(build: str, timeout: float = 30.0) -> str:
    """The sha256 GitHub publishes for this build's asset, read before fetching.

    Available for any tag, which is what makes "always verify" achievable for a
    candidate build and not only for the pinned one.
    """
    req = urllib.request.Request(
        f"{API}/{build}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            release = json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise BuildError(f"could not read the release for {build}: {exc}") from exc

    want = asset(build)
    for entry in release.get("assets", []):
        if entry.get("name") == want:
            digest = entry.get("digest") or ""
            if not digest.startswith("sha256:"):
                raise BuildError(f"{want} is published without a sha256 digest")
            return digest.split(":", 1)[1]
    raise BuildError(f"{build} publishes no asset named {want}")


def fetch(build: str, target: Path, expect: str | None = None) -> str:
    """Download, verify and unpack one build. Returns the digest of what landed.

    ``expect`` is a digest recorded in this repository. When it is given and the
    API disagrees with it, the fetch stops: the asset behind the tag has been
    replaced, which is the one thing a published digest can never tell you on
    its own. When it is given and the API cannot be reached -- it is rate
    limited unauthenticated -- the recorded digest stands alone and the fetch
    continues, because it is the stronger of the two anyway. Without it, an
    unreachable API is fatal: an unverified build is not worth measuring.
    """
    try:
        published: str | None = published_digest(build)
    except BuildError:
        if expect is None:
            raise
        published = None

    if expect and published and expect != published:
        raise BuildError(
            f"{asset(build)} no longer matches the digest recorded for it.\n"
            f"  recorded  {expect}\n  published {published}\n"
            "The bytes behind this tag have been replaced since it was pinned."
        )

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / asset(build)
        try:
            urllib.request.urlretrieve(url(build), archive)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise BuildError(f"could not download {asset(build)}: {exc}") from exc

        got = hashlib.sha256(archive.read_bytes()).hexdigest()
        against = expect or published
        if got != against:
            raise BuildError(
                f"Checksum mismatch for {asset(build)}!\n"
                f"  expected {against}\n  got      {got}"
            )

        extract_to = Path(tmp) / "x"
        with tarfile.open(archive) as tar:
            tar.extractall(extract_to, filter="data")
        # The archive contains a single build directory; flatten it.
        inner = next(p for p in extract_to.rglob("llama-server")).parent
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(inner, target)

    for binary in ("llama-server", "llama-cli", "llama-bench"):
        path = target / binary
        if path.exists():
            path.chmod(0o755)
    return got


# ---------------------------------------------------------------------------
# Which backend a build was compiled against
# ---------------------------------------------------------------------------
# Nothing above this line cares -- a tag identifies an archive, and while there
# was only one archive that was the whole story. A locally compiled CUDA engine
# has no tag and no digest, so the only thing that can identify it is the
# binary itself, and the only thing that can ask the binary is the binary.

#: The device-name prefixes llama.cpp prints, and the backend each one means.
#:
#: ``--list-devices`` names devices, not backends: a Vulkan build reports
#: ``Vulkan0``, a CUDA build ``CUDA0``. That is the only place the distinction
#: is visible from outside -- both ship a binary called ``llama-server``,
#: neither writes a backend banner to its log on this build, and ``--version``
#: reports the commit, which for two builds of the same commit is the same.
DEVICE_BACKENDS = {
    "cuda": "cuda",
    "vulkan": "vulkan",
    "rocm": "rocm",
    "sycl": "sycl",
    "metal": "metal",
}

#: What a build reports when it found no accelerator at all -- and what an
#: engine that could not be asked is treated as. That is deliberate: ``cpu`` is
#: the backend with no measured envelope of its own and no special flags, so
#: falling back to it withholds a claim rather than making a wrong one.
CPU = "cpu"

#: Answers to :func:`backend`, keyed the way :func:`lllm3090.engine.supports`
#: keys its own -- by path *and* mtime, because ``install-engine --force``
#: replaces a binary underneath a panel that has been up for days, and an
#: answer cached from the old one would outlive it.
_BACKENDS: dict[tuple[str, float], str] = {}


def _backend_from_devices(text: str) -> str:
    """The backend named by a ``--list-devices`` listing.

    Lines look like ``  CUDA0: NVIDIA GeForce RTX 3090 (24125 MiB, ...)``. The
    device index is stripped rather than matched, because a two-card machine
    reports ``CUDA0`` and ``CUDA1`` and both mean the same backend.
    """
    for line in text.splitlines():
        head = line.strip().split(":", 1)[0]
        name = head.rstrip("0123456789").lower()
        if name in DEVICE_BACKENDS:
            return DEVICE_BACKENDS[name]
    return CPU


def backend(directory: Path | None = None) -> str:
    """Which compute backend the engine in ``directory`` was built against.

    Answers ``"cuda"``, ``"vulkan"`` or ``"cpu"`` -- and names ROCm, SYCL or
    Metal if it ever meets one. This is not cosmetic. The KV cache costs about
    10% more per token on CUDA than on Vulkan and the backend carries its own
    fixed overhead, so a planner that does not know which one it is planning
    for over-promises context on the more expensive of the two; and the
    speculation settings that win on one lose on the other. A package with two
    backends on disk and no way to tell them apart gets both wrong.

    Cached, because ``catalog.fit`` runs once per catalogue entry per
    ``lllm3090 models`` and a subprocess apiece would be felt.
    """
    directory = directory or config.LLAMA_DIR
    binary = directory / "llama-server"
    try:
        key = (str(binary), binary.stat().st_mtime)
    except OSError:
        # No engine there at all. Not an error -- ``models`` runs happily
        # before ``install-engine`` on a fresh machine -- but nothing about a
        # binary that is not there can be claimed.
        return CPU
    if key in _BACKENDS:
        return _BACKENDS[key]
    try:
        out = subprocess.run(
            [str(binary), "--list-devices"],
            capture_output=True, text=True, timeout=60, check=False,
            # The build's own shared objects, not the system's. Two engines sit
            # on this disk and both ship a libggml; without this the one that
            # happens to be found first would answer for the other.
            env=dict(os.environ, LD_LIBRARY_PATH=str(directory)),
        )
        text = out.stdout + out.stderr
    except (OSError, subprocess.SubprocessError):
        # Not cached: a timeout under load is a fact about the moment, and
        # remembering it would make one busy minute permanent.
        return CPU
    _BACKENDS[key] = _backend_from_devices(text)
    return _BACKENDS[key]
