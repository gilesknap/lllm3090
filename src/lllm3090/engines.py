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
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
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

#: What a build reports when it found no accelerator at all.
#:
#: A real answer about a real binary, and therefore *not* what an engine that
#: could not be asked is called -- see :data:`UNKNOWN`. The two were one value
#: until a CPU-only build would have had the catalogue label Vulkan GPU speeds
#: "measured" on an engine incapable of producing them.
CPU = "cpu"

#: What an engine that could not be asked is called -- nothing installed yet,
#: a probe that timed out, a probe that exited non-zero.
#:
#: Distinct from :data:`CPU` because the two carry opposite claims. ``cpu`` is
#: a fact about a binary -- no accelerator, so no measured speed here applies.
#: ``unknown`` is the *absence* of a fact, and it has to stay the benign
#: fallback: ``lllm3090 models`` runs before ``install-engine`` on a fresh
#: machine, and qualifying every row there would be noise about a backend that
#: is about to be Vulkan.
#:
#: Never cached, for the same reason a timeout is not: it is a statement about
#: the moment, and the binary is still whatever it was compiled as.
UNKNOWN = "unknown"

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
    directory = directory or active_dir()
    binary = directory / "llama-server"
    try:
        key = (str(binary), binary.stat().st_mtime)
    except OSError:
        # No engine there at all. Not an error -- ``models`` runs happily
        # before ``install-engine`` on a fresh machine -- but nothing about a
        # binary that is not there can be claimed.
        return UNKNOWN
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
        if out.returncode != 0:
            # A probe that ran and failed says nothing about how the binary
            # was compiled -- and left uninspected it said the wrong thing.
            # ``--list-devices`` on a build whose backend fails to initialise
            # prints its error and exits non-zero; the device list is then
            # empty, which parses as ``cpu``, which would be *cached* for the
            # life of a panel process. From there ``catalog.fit`` drops this
            # backend's KV factor and fixed overhead and promises a window the
            # card cannot hold. Same class as the timeout below, so same
            # answer: refuse to remember it.
            return UNKNOWN
        text = out.stdout + out.stderr
    except (OSError, subprocess.SubprocessError):
        # Not cached: a timeout under load is a fact about the moment, and
        # remembering it would make one busy minute permanent.
        return UNKNOWN
    _BACKENDS[key] = _backend_from_devices(text)
    return _BACKENDS[key]


# ---------------------------------------------------------------------------
# Building a CUDA engine, because nobody publishes one
# ---------------------------------------------------------------------------
# Everything above this line downloads a build and proves it is the build it
# claims to be. None of that is available here: as of b10715 llama.cpp
# publishes CUDA archives for Windows only, so on Linux a CUDA engine is
# something this machine compiles or does not have.
#
# That is a weaker promise and it is deliberately kept visible. A downloaded
# build is a tag and a digest recorded in this repository; a compiled one
# reports "build 1, commit <sha>" -- a shallow clone has no tag history, so
# only the commit is real and nobody attests to the binary. Which is why
# nothing here ever switches the active engine: the user makes that promise,
# not `setup`.

#: The longest pool of the dense 27B each backend will actually hold, found by
#: loading until it dies, with a desktop session running. Vulkan dies at 204800
#: and CUDA at 176128.
#:
#: These are what makes "CUDA costs context" a number rather than an
#: impression: it is ~10% more KV per token plus ~230 MiB more fixed overhead,
#: and it lands as about a seventh of the window. Recorded here, together, so
#: that the figure `setup` quotes and the figure the planner would use cannot
#: drift apart -- and so re-measuring is one edit.
VULKAN_CEILING = 200704
CUDA_CEILING = 172032


def cuda_window_cost() -> int:
    """What choosing CUDA costs in window, as a whole percent."""
    return round(100 * (VULKAN_CEILING - CUDA_CEILING) / VULKAN_CEILING)


#: The oldest CUDA that can compile this against a current glibc.
#:
#: Not a preference. Ubuntu 26.04 ships 13.1 in multiverse, and it declares
#: ``rsqrt``/``rsqrtf`` without an exception specifier while glibc 2.43
#: declares them ``noexcept(true)``; ``nvcc`` then refuses to compile anything
#: that includes ``<math.h>``, which is everything. 13.3 tests for glibc >= 2.42
#: and matches it. So the toolkit has to come from NVIDIA's own repository, and
#: a machine with both installed must not be allowed to pick the older one --
#: which is exactly what ``/usr/local/cuda`` does when alternatives points
#: there.
MIN_CUDA = (13, 3)

#: Where llama.cpp is cloned from to build it. The same project the pinned
#: archives come from, checked out at the same tag, so a locally built engine
#: is the pinned build compiled differently rather than a different engine.
SOURCE_REPO = "https://github.com/ggml-org/llama.cpp.git"

#: What is worth compiling. The server is the product; bench and cli are what
#: make a locally built engine measurable against a downloaded one, which is
#: the only way anyone can check the claim that it is faster.
BUILD_TARGETS = ("llama-server", "llama-bench", "llama-cli")

#: What to tell someone who has no usable toolkit.
#:
#: Printed rather than executed. This is 4-6 GB of packages from a third-party
#: repository, and installing that behind someone's back -- during a command
#: whose job is to get a Vulkan engine working -- is not a reasonable reading
#: of "setup".
TOOLKIT_INSTRUCTIONS = """\
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/\
x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get install -y cuda-toolkit-13-3

Not `apt install nvidia-cuda-toolkit`. The distribution's 13.1 cannot compile
this against glibc 2.43 -- it declares rsqrt/rsqrtf without an exception
specifier where glibc declares them noexcept(true), and nvcc then refuses
every file that includes <math.h>."""


@dataclass(frozen=True)
class Toolkit:
    """An ``nvcc`` on this machine, and whether it is new enough to be used."""

    nvcc: Path
    version: tuple[int, ...]

    @property
    def text(self) -> str:
        return ".".join(str(part) for part in self.version)

    @property
    def usable(self) -> bool:
        return self.version >= MIN_CUDA


def _nvcc_version(nvcc: Path) -> tuple[int, ...] | None:
    """The release ``nvcc`` reports, as comparable numbers, or None."""
    try:
        out = subprocess.run(
            [str(nvcc), "--version"],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    found = re.search(r"release (\d+)\.(\d+)", out)
    return tuple(int(g) for g in found.groups()) if found else None


def toolkits() -> list[Toolkit]:
    """Every CUDA toolkit findable on this machine, newest first.

    Both the ``PATH`` and ``/usr/local`` are searched, and ``/usr/local/cuda``
    is deliberately *not* trusted as the answer: it is a symlink managed by
    alternatives, and on a machine with 13.1 and 13.3 installed it points at
    whichever one was configured last. Picking the newest that works is the
    only reading of "a CUDA toolkit is present" that cannot pick the one this
    cannot be built with.
    """
    found: dict[Path, Toolkit] = {}
    candidates = []
    on_path = shutil.which("nvcc")
    if on_path:
        candidates.append(Path(on_path))
    candidates += sorted(Path("/usr/local").glob("cuda*/bin/nvcc"))
    for nvcc in candidates:
        real = nvcc.resolve()
        if real in found:
            continue
        version = _nvcc_version(real)
        if version is not None:
            found[real] = Toolkit(real, version)
    return sorted(found.values(), key=lambda t: t.version, reverse=True)


def toolkit() -> Toolkit | None:
    """The newest toolkit that can actually compile this, or None."""
    return next((t for t in toolkits() if t.usable), None)


def sm(compute_capability: str) -> str | None:
    """A compute capability as CUDA spells an architecture: ``8.6`` -> ``86``.

    ``None`` for a card that would not report one. Derived from what
    ``nvidia-smi`` says rather than typed anywhere, because a hand-written
    architecture is a number that goes stale in a machine the moment its card
    is replaced -- and the binary it produces does not fail, it runs wrong.
    """
    parts = compute_capability.split(".")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        return None
    return f"{int(parts[0])}{int(parts[1])}"


def cuda_dir(build: str, arch: str) -> Path:
    """Where a CUDA build for one architecture lives.

    The architecture is in the name on purpose. The build is compiled
    ``sm_<arch>-real``, which will not run on a different architecture at all,
    so naming the directory after the tag alone would let a card swap turn into
    an engine that dies at load for no stated reason. Named this way, the
    absence is legible: the directory for the new card simply is not there.
    """
    return config.ENGINES_DIR / f"{build}-cuda-sm{arch}"


def cuda_builds() -> list[Path]:
    """Every locally compiled CUDA engine on this machine."""
    if not config.ENGINES_DIR.is_dir():
        return []
    return sorted(
        d for d in config.ENGINES_DIR.iterdir()
        if d.is_dir() and "-cuda-sm" in d.name and (d / "llama-server").exists()
    )


def stale_cuda_builds(build: str | None = None) -> list[Path]:
    """Locally compiled engines that are no longer the pinned build.

    The most likely way this feature rots. A CUDA engine is tied to the commit
    it was compiled from, and moving ``LLAMA_BUILD`` does not move it -- so a
    user who chose CUDA is silently left on an older engine than the one every
    figure in the catalogue was measured against, with nothing to say so.
    """
    build = build or LLAMA_BUILD
    return [d for d in cuda_builds() if not d.name.startswith(f"{build}-cuda-sm")]


def build_cuda(
    target: Path,
    arch: str,
    kit: Toolkit,
    build: str | None = None,
    log: Path | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    """Compile a CUDA llama-server for one architecture. Returns ``target``.

    Checked out at the pinned tag, so this is the shipped build compiled
    differently rather than a different engine. Built into a temporary
    directory and copied into place only once the binary exists, because a
    half-populated engine directory is indistinguishable from a complete one
    and would be launched as though it were.
    """
    build = build or LLAMA_BUILD
    log = log or (config.STATE_DIR / "build-cuda.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    for tool in ("git", "cmake"):
        if not shutil.which(tool):
            raise BuildError(f"{tool} is not installed; it is needed to build CUDA")

    def run(argv: list[str], cwd: Path | None = None) -> None:
        with log.open("a") as sink:
            sink.write(f"\n$ {' '.join(argv)}\n")
            sink.flush()
            proc = subprocess.Popen(
                argv, cwd=cwd, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sink.write(line)
                # cmake's own percentage, and nothing else. A compile prints
                # tens of thousands of lines and none of the others answer the
                # only question anyone has while waiting.
                if on_progress and re.match(r"\[\s*\d+%\]", line):
                    on_progress(line.rstrip())
            if proc.wait() != 0:
                raise BuildError(
                    f"{argv[0]} failed while building CUDA. The full output is "
                    f"in {log}"
                )

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "llama.cpp"
        run([
            "git", "clone", "--depth", "1", "--branch", build,
            SOURCE_REPO, str(source),
        ])
        run([
            "cmake", "-S", str(source), "-B", str(source / "build"),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA=ON",
            # -real, not -virtual: no PTX is embedded, so the binary runs on
            # this architecture and refuses every other one rather than
            # silently JIT-compiling itself onto a card it was not measured on.
            f"-DCMAKE_CUDA_ARCHITECTURES={arch}-real",
            f"-DCMAKE_CUDA_COMPILER={kit.nvcc}",
            # Nothing here downloads models through the engine, and libcurl's
            # headers are one more thing that has to be installed to build.
            "-DLLAMA_CURL=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
        ])
        run([
            "cmake", "--build", str(source / "build"), "--config", "Release",
            "-j", str(os.cpu_count() or 4), "--target", *BUILD_TARGETS,
        ])
        binaries = source / "build" / "bin"
        if not (binaries / "llama-server").exists():
            raise BuildError(
                f"the build finished but produced no llama-server. See {log}"
            )
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Everything, not just the binaries: the engine is launched with
        # LD_LIBRARY_PATH pointing at this directory, and libggml-cuda.so lives
        # beside llama-server exactly as it does in a downloaded archive.
        shutil.copytree(binaries, target)
    for binary in BUILD_TARGETS:
        path = target / binary
        if path.exists():
            path.chmod(0o755)
    return target


# ---------------------------------------------------------------------------
# Which engine actually serves
# ---------------------------------------------------------------------------
# Two backends can sit on this disk at once and they are not interchangeable:
# CUDA is worth 1.55-1.60x on the dense 27B and costs about a seventh of the
# window, and the speculation settings that win on one lose on the other. So
# which one serves is a choice, and until now the only way to express it was an
# environment variable -- which reaches a shell and therefore the CLI, but not
# a panel started by systemd minutes after boot.


@dataclass(frozen=True)
class Choice:
    """One engine this machine could serve with."""

    path: Path
    backend: str
    #: True for the engine ``install-engine`` manages. The others are compiled.
    installed: bool
    #: Compiled against a build that is no longer the pinned one. Offered
    #: anyway -- it works -- but every catalogue figure was measured on the pin.
    stale: bool

    @property
    def id(self) -> str:
        """What a front end sends back to choose this one.

        The directory name, never a path. The panel is a loopback HTTP server
        that starts processes, and accepting a path from it would let a
        request name any binary on the disk; a name is looked up in the list
        this function produced, so nothing outside it can be selected.
        """
        return self.path.name


def choices() -> list[Choice]:
    """Every engine on this machine that could serve, best-known first.

    The managed install first because it is what the catalogue's speeds were
    measured on and what a fresh machine has.
    """
    stale = {d.name for d in stale_cuda_builds()}
    found: list[Choice] = []
    if (config.LLAMA_DIR / "llama-server").exists():
        found.append(Choice(
            path=config.LLAMA_DIR,
            backend=backend(config.LLAMA_DIR),
            installed=True,
            stale=False,
        ))
    for d in cuda_builds():
        found.append(Choice(
            path=d, backend=backend(d), installed=False, stale=d.name in stale,
        ))
    return found


def chosen() -> Path | None:
    """The engine directory recorded as chosen, or None.

    None covers every way of not having an answer -- never chosen, the file
    corrupted, the directory deleted since. A stored choice that no longer
    names a runnable engine is *not* an error to raise at the caller: the
    engine it named can be removed by a ``build-cuda --force`` that failed
    halfway, and the right behaviour then is to fall back to the install that
    is still there rather than to refuse to start anything.
    """
    try:
        name = json.loads(config.ENGINE_CHOICE.read_text())["dir"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return next((c.path for c in choices() if c.id == name), None)


def active_dir() -> Path:
    """The engine directory that will actually serve the next start.

    Precedence, highest first:

    * ``LLLM3090_LLAMA_DIR``. An explicit override from the environment, and
      the documented way to run an engine this code knows nothing about --
      a build under test, a distribution's own. It outranks a stored choice
      because it is the more specific statement, and because a variable set in
      a shell is visibly about *this* invocation.
    * What :func:`select` recorded.
    * The managed install.
    """
    if config.LLAMA_DIR_FROM_ENV:
        return config.LLAMA_DIR
    return chosen() or config.LLAMA_DIR


def select(choice_id: str | None) -> Path:
    """Record which engine serves from now on. Returns the directory chosen.

    ``None`` clears the choice, which is how a front end says "back to the
    managed install" without having to know its name.

    Nothing is started, stopped or copied. A running engine is a process that
    already has its binary open and keeps serving from it; this decides what
    the *next* start loads, and the front ends say so rather than implying a
    switch that has not happened.
    """
    config.ENGINE_CHOICE.parent.mkdir(parents=True, exist_ok=True)
    if choice_id is None:
        config.ENGINE_CHOICE.unlink(missing_ok=True)
        return config.LLAMA_DIR
    match = next((c for c in choices() if c.id == choice_id), None)
    if match is None:
        raise ValueError(f"no engine here called {choice_id!r}")
    if match.installed:
        # Recording the managed install by name would pin it to a path that
        # `install-engine` is entitled to move. Clearing says the same thing
        # and keeps saying it afterwards.
        config.ENGINE_CHOICE.unlink(missing_ok=True)
        return match.path
    config.ENGINE_CHOICE.write_text(json.dumps({"dir": match.id}) + "\n")
    return match.path
