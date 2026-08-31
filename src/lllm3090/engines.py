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
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from . import config

#: The build the product runs.
LLAMA_BUILD = "b10628"

#: Its digest. See the module docstring for why this is written down rather
#: than only asked for at fetch time.
LLAMA_SHA256 = "c64b6d5820ea6dc3227495e2c30c397fb73c24158291cfb7ef99892a708605a6"

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
