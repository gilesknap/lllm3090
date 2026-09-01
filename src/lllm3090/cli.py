"""The ``lllm3090`` command."""

from __future__ import annotations

import getpass
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path

import typer

from . import catalog, config, engine, engines, hardware, preflight, storage
from ._version import __version__

# The pin, the digest recorded for it and the fetch itself live in
# `lllm3090.engines`, because a benchmark build has to be fetched the same way
# without going through the CLI. Re-exported: `cli.LLAMA_BUILD` was the name
# before there was more than one build.
LLAMA_BUILD = engines.LLAMA_BUILD
LLAMA_SHA256 = engines.LLAMA_SHA256

app = typer.Typer(add_completion=False, help=__doc__)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Print the version and exit.",
    ),
) -> None:
    """Local LLM serving for a single RTX 3090."""


def _echo_check(name: str, ok: bool, message: str) -> None:
    mark = "[ ok ]" if ok else "[FAIL]"
    typer.echo(f"  {mark} {name:<12} {message}")


@app.command()
def doctor() -> None:
    """Check this machine can run the stack, and say precisely what is missing."""
    typer.echo(f"lllm3090 {__version__}")
    results = preflight.run_all()
    for name, ok, message in results:
        _echo_check(name, ok, message)
    failures = [n for n, ok, _ in results if not ok]
    if failures:
        typer.echo(f"\n{len(failures)} check(s) failed: {', '.join(failures)}")
        raise typer.Exit(1)
    typer.echo("\nAll checks passed.")


@app.command("install-engine")
def install_engine(
    force: bool = typer.Option(False, help="Reinstall if present."),
) -> None:
    """Fetch and verify the pinned llama.cpp build."""
    target = config.LLAMA_DIR
    if (target / "llama-server").exists() and not force:
        typer.echo(f"Engine already installed at {target} (use --force to replace)")
        return

    typer.echo(f"Downloading {engines.asset(LLAMA_BUILD)} ...")
    try:
        digest = engines.fetch(LLAMA_BUILD, target, expect=LLAMA_SHA256)
    except engines.BuildError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(f"Checksum verified: {digest}")
    typer.echo(f"Engine installed at {target}")


@app.command("fetch-engine")
def fetch_engine(
    build: str = typer.Option(
        LLAMA_BUILD, help="Upstream llama.cpp tag, e.g. b10715."),
    force: bool = typer.Option(False, help="Re-fetch if already present."),
) -> None:
    """Fetch a build to measure against, without touching the installed engine.

    Deciding whether to move the pin means running a candidate and the incumbent
    on the same machine, so a benchmark build is kept beside the install rather
    than replacing it. The digest printed here is what gets committed if the
    candidate wins and becomes the pin.
    """
    target = engines.bench_dir(build)
    if (target / "llama-server").exists() and not force:
        typer.echo(f"{build} already at {target} (use --force to re-fetch)")
        return

    typer.echo(f"Downloading {engines.asset(build)} ...")
    expect = LLAMA_SHA256 if build == LLAMA_BUILD else None
    try:
        digest = engines.fetch(build, target, expect=expect)
    except engines.BuildError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(f"Checksum verified: {digest}")
    if expect is None:
        typer.echo("Verified against the digest GitHub publishes for this tag.")
        typer.echo("Commit it as LLAMA_SHA256 if this build becomes the pin.")
    typer.echo(f"{build} fetched to {target}")


@app.command()
def models() -> None:
    """List the curated catalogue and what is already downloaded."""
    profile = hardware.detect()
    desktop = hardware.graphical()
    if not profile.present:
        typer.echo("Card: none detected (nvidia-smi found no GPU)")
        typer.echo(
            f"Context below is computed against a {hardware.reference().name}'s "
            f"{profile.vram_mib // 1024} GB so the catalogue can be read; it "
            "describes no card in this machine, and neither do the speeds."
        )
    else:
        typer.echo(f"Card: {profile.name} ({profile.vram_mib // 1024} GB)")
        if not profile.measured:
            typer.echo(
                f"Context is computed for this card. Speeds were measured on a "
                f"{hardware.reference().name} and are shown for reference only."
            )
    # Said only once a card has been established. With no GPU the capacity above
    # is borrowed from the reference profile, and "the full card is available"
    # would be describing a card that is not in this machine.
    if profile.present:
        if desktop:
            typer.echo(
                "Context is computed with a desktop session's VRAM held back. "
                "See 'headless' in the docs\nfor how to get it back -- it is "
                "worth up to three times the cache on some models."
            )
        else:
            typer.echo(
                "No desktop session: the full card is available for context."
            )
    typer.echo("")
    header = f"{'MODEL':<24}{'SIZE':>8}{'KIND':>10}{'CONTEXT':>12}  {'STATE':<16}SPEED"
    typer.echo(header)
    rows = catalog.catalog_for_panel()
    cautions: list[str] = []
    for row in rows:
        # Two independent facts: whether it is on the disk, and what the card
        # makes of it. Collapsing them loses one, and it was the caution that
        # got lost -- an installed model reading "installed" while leaving less
        # context than an agent's own prompt is the case this column exists for.
        verdict = {
            catalog.STATUS_TOO_BIG: "too big",
            catalog.STATUS_TIGHT: "tight",
            catalog.STATUS_CAPABILITY: "old GPU",
        }.get(row["status"], "")
        if row["installed"]:
            state = f"{verdict}, on disk" if verdict else "installed"
        else:
            state = verdict or "fits"
        if row["status_note"]:
            cautions.append(f"  {row['name']}: {row['status_note']}")
        speed = f"~{row['expected_tok_s']} tok/s" if row["expected_tok_s"] else "-"
        if row["verified"]:
            speed += " (measured)" if row["speed_applies"] else " (other card)"
        ctx = (
            f"{row['max_ctx'] // 1024}k x{row['parallel']}" if row["fits"] else "-"
        )
        kind = row["kind"] + ("+vis" if row["vision"] else "")
        typer.echo(
            f"{row['name']:<24}{row['gb']:>7.1f}G{kind:>10}{ctx:>12}  "
            f"{state:<16}{speed}"
        )
    if cautions:
        typer.echo("")
        for line in cautions:
            typer.echo(line)


def _profile_named(gpu: str | None) -> hardware.Profile:
    """The profile to price against: a named card, or the one in this machine."""
    if gpu is None:
        return hardware.detect()
    profiles = hardware.load_profiles()
    match = next((p for p in profiles if p.id == gpu), None)
    if match is None:
        ids = ", ".join(p.id for p in profiles)
        typer.echo(f"Unknown GPU {gpu!r}. Known profiles: {ids}")
        raise typer.Exit(1)
    return match


@app.command()
def sweep(
    limit: int = typer.Option(
        100, help="How many of the most-downloaded GGUF repos to price."
    ),
    gpu: str | None = typer.Option(
        None, help="Price against a named profile instead of the detected card."
    ),
    show_yaml: bool = typer.Option(
        False, "--yaml", help="Emit catalogue entries for the survivors."
    ),
    show_skipped: bool = typer.Option(
        False, "--skipped", help="List repos that could not be priced, and why."
    ),
) -> None:
    """Survey published GGUF models and price them against this card.

    Downloads no weights: each candidate costs one config.json, from which the
    KV cost per token is derived and then run through the same arithmetic the
    panel uses. What comes out is a shortlist worth spending disk on, and --
    just as usefully -- a list of models that fit the card and still cannot
    hold an agent's system prompt.
    """
    from . import sweep as sweep_mod

    profile = _profile_named(gpu)
    desktop = hardware.graphical()
    typer.echo(
        f"Pricing against {profile.name} ({profile.vram_mib // 1024} GB)"
        + ("" if gpu is None else " [--gpu override]")
        + (", desktop session held back" if desktop else ", headless")
    )
    typer.echo(f"Surveying the top {limit} GGUF repositories...\n")

    keep, reject, skipped = sweep_mod.survey(limit, profile, desktop)

    typer.echo(f"{'CANDIDATE':<34}{'SIZE':>8}{'KIND':>7}{'KV':>7}  CONTEXT")
    for r in keep:
        c = r.candidate
        typer.echo(
            f"{c.name[:33]:<34}{c.size_gb:>7.1f}G{c.kind:>7}"
            f"{c.kv_kib_per_token:>6g}K  {r.plan.summary}"
        )
    if not keep:
        typer.echo("  (nothing new clears the bar)")

    if reject:
        typer.echo(f"\nPriced and rejected ({len(reject)}):")
        # Only said when something was rejected for that reason. A rejection
        # list that is all "too big" does not need the agent floor explained,
        # and saying "these fit the card" over an entry that does not fit is
        # the kind of small untruth that makes the rest look careless.
        if any(r.status == catalog.STATUS_TIGHT for r in reject):
            typer.echo(f"  {sweep_mod.agent_floor_note()}")
        typer.echo("")
        for r in reject:
            typer.echo(f"  {r.candidate.name[:33]:<34}{r.note}")

    if show_skipped and skipped:
        typer.echo(f"\nNot priced ({len(skipped)}):")
        for repo, why in skipped:
            typer.echo(f"  {repo[:44]:<46}{why}")
    elif skipped:
        typer.echo(f"\n{len(skipped)} repos could not be priced (--skipped to see why)")

    if show_yaml and keep:
        typer.echo("\n--- paste into src/lllm3090/data/models.yaml ---\n")
        typer.echo(sweep_mod.to_yaml(keep, profile))
        typer.echo(
            "\n--- speeds are deliberately absent: run 'lllm3090 bench <model>'\n"
            "--- after downloading, then set expected_tok_s and verified: true"
        )


@app.command()
def bench(
    model: str = typer.Argument(..., help="An installed model to benchmark."),
) -> None:
    """Benchmark a model with llama-bench and print a profile contribution.

    The catalogue's speeds are measurements, never extrapolations, so a card
    other than the one they were taken on has no numbers until somebody runs
    this on it. The output is meant to be pasted into an issue.
    """
    entry = next((m for m in catalog.installed() if m["name"] == model), None)
    if entry is None:
        typer.echo(f"{model!r} is not installed. Try: lllm3090 models")
        raise typer.Exit(1)
    binary = config.LLAMA_DIR / "llama-bench"
    if not binary.exists():
        typer.echo(f"llama-bench not found at {binary}; run 'lllm3090 setup'")
        raise typer.Exit(1)

    profile = hardware.detect()
    typer.echo(f"Benchmarking {model} on {profile.name} -- this takes a few minutes.\n")
    env = dict(os.environ, LD_LIBRARY_PATH=str(config.LLAMA_DIR))
    result = subprocess.run(
        [str(binary), "-m", entry["path"], "-ngl", "999", "-p", "512", "-n", "128"],
        env=env, capture_output=True, text=True, check=False,
    )
    typer.echo(result.stdout or result.stderr)
    if result.returncode != 0:
        raise typer.Exit(1)

    typer.echo("\n--- paste this into an issue at")
    typer.echo("--- https://github.com/gilesknap/lllm3090/issues\n")
    known = profile.present and not profile.detected
    typer.echo(f"""  - id: {profile.id if known else "CHOOSE-AN-ID"}
    name: {profile.name}
    compute_capability: "{profile.compute_capability}"
    vram_mib: {profile.vram_mib}
    bandwidth_gbs: {profile.bandwidth_gbs or "FILL-IN"}
    measured: false
    notes: >-
      Benchmarked with lllm3090 bench on {model}. Paste the table above so the
      tg (token generation) figure can be checked against the catalogue.""")


@app.command()
def status() -> None:
    """Show what the engine is doing."""
    state = engine.status()
    if not state["running"]:
        typer.echo("Engine: stopped")
        raise typer.Exit(0)
    typer.echo(
        f"Engine: running (pid {state['pid']}) on port {state['port']}\n"
        f"Model:  {state['model'] or 'loading...'}\n"
        f"Ready:  {'yes' if state['answering'] else 'not yet'}"
    )


@app.command()
def start(
    model: str = typer.Argument(..., help="Directory name under the models dir."),
    ctx: int = typer.Option(None, help="Total KV pool. Default: computed to fit."),
    parallel: int = typer.Option(
        None, min=1,
        help="Conversations sharing the pool. Default: 2, for subagents.",
    ),
    effort: str = typer.Option(
        None,
        help="Reasoning level for the whole session: "
             + ", ".join(engine.REASONING_EFFORTS) + ". Default: the model's.",
    ),
) -> None:
    """Start the engine on an installed model."""
    # Checked before the stop, deliberately: a typo is a typo, and taking a
    # working engine down to report one costs a reload of the weights.
    if effort is not None and effort not in engine.REASONING_EFFORTS:
        typer.echo(
            f"unknown effort {effort!r}; one of: "
            f"{', '.join(engine.REASONING_EFFORTS)}"
        )
        raise typer.Exit(1)
    entry = next((m for m in catalog.installed() if m["name"] == model), None)
    if entry is None:
        typer.echo(f"{model!r} is not installed. Try: lllm3090 models")
        raise typer.Exit(1)
    known = next((m for m in catalog.load_catalog() if m.name == model), None)
    if ctx is None:
        p = catalog.launch_plan(model, parallel)
        ctx, parallel = p.pool, p.parallel
        typer.echo(f"Context plan: {p.summary}")
    else:
        parallel = parallel or config.DEFAULT_PARALLEL
    # The plan is computed against a fixed reserve; this is the measurement.
    # Sizing on a text console and then starting under a desktop is the case
    # that loads, reports itself healthy, and fails every request out of memory.
    engine.stop()
    warning = catalog.free_vram_warning(known, ctx)
    if warning:
        typer.echo(warning)
    template = known.chat_template if known else None
    ok, detail = engine.start(
        entry["path"], model, ctx, parallel, 300, template,
        entry.get("mmproj"), effort,
    )
    typer.echo(detail)
    raise typer.Exit(0 if ok else 1)


@app.command()
def stop() -> None:
    """Stop the engine and free the VRAM."""
    typer.echo(engine.stop()[1])


APT_PACKAGES = ["libvulkan1"]


def _apt_missing() -> list[str]:
    missing = []
    for pkg in APT_PACKAGES:
        result = subprocess.run(
            ["dpkg", "-s", pkg], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            missing.append(pkg)
    return missing


def _whoami() -> str:
    """The current user's name, without a terminal to ask for one.

    ``os.getlogin`` reads the controlling terminal and raises ``OSError``
    without one -- in a service, a pipeline, or an agent session. It would do so
    while printing the command that recovers from a permission error, which is
    the worst possible moment to raise instead of speaking.
    """
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER") or os.environ.get("LOGNAME") or "$USER"


def _checkpoints_under(path: Path) -> Path | None:
    """Where ``path`` leads, if anything is stored there, else ``None``.

    Follows a symlink deliberately. ``~/models`` is a symlink on any machine
    that has already been through ``--model-folder``, and repointing one that
    leads to 180 GB does not delete anything -- it makes it invisible, while the
    disk stays full. That is a worse failure than deleting, because nothing
    reports it.
    """
    if not path.exists():
        return None
    resolved = path.resolve()
    if not resolved.is_dir():
        return None
    return resolved if any(resolved.iterdir()) else None


def _configure_models_folder(requested: str | None) -> None:
    """Settle where checkpoints live, before anything is downloaded into it.

    Asked here because it is the last moment the answer is cheap. Once there
    are 180 GB in the wrong place, changing it means copying them.

    An explicit ``--model-folder`` is honoured whatever disk it lands on -- the
    check exists to stop an unconsidered default, not to overrule a decision.
    """
    default = config.MODELS_DIR
    if requested is None:
        warning = storage.slow_disk_warning(default)
        if warning is None:
            disk = storage.backing_disk(default)
            where = f" on /dev/{disk}" if disk else ""
            _echo_check(
                "models dir", True,
                f"{default}{where} ({storage.free_gb(default):.0f} GB free)",
            )
            return
        _echo_check("models dir", False, f"{default} is not on an NVMe")
        typer.echo("\n" + warning)
        raise typer.Exit(1)

    # Absolute from here on: symlink_to() stores what it is given, so a
    # relative --model-folder would be resolved against ~/ and
    # `--model-folder models` would point ~/models at itself.
    target = Path(requested).expanduser().absolute()
    if not target.exists():
        try:
            target.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            typer.echo(
                f"\nCannot create {target}: its parent belongs to another user.\n"
                f"Create it once, then re-run this command:\n\n"
                f"    sudo mkdir -p {target} && sudo chown {_whoami()} {target}\n"
            )
            raise typer.Exit(1) from None
    if not target.is_dir():
        typer.echo(f"\n{target} exists and is not a directory.")
        raise typer.Exit(1)
    if not os.access(target, os.W_OK):
        typer.echo(
            f"\n{target} exists but you cannot write to it. Give it to yourself:\n\n"
            f"    sudo chown {_whoami()} {target}\n"
        )
        raise typer.Exit(1)

    disk = storage.backing_disk(target)
    where = f" on /dev/{disk}" if disk else ""
    already_there = (
        default.exists() and target.resolve() == default.resolve()
    ) or target == default
    if already_there:
        _echo_check("models dir", True, f"{target}{where}")
        return

    # Everything else in the project reads config.MODELS_DIR, which defaults to
    # ~/models. A symlink there means the choice needs no environment variable,
    # no edit to the service unit, and nothing to remember on the next upgrade.
    #
    # Whatever is there now, it is only safe to replace when it holds nothing.
    # A symlink counts: repointing one that leads to 180 GB of checkpoints does
    # not delete them, but it does make them invisible to every part of this
    # program, which is worse -- the disk stays full and the models are gone.
    holding = _checkpoints_under(default)
    if holding is not None:
        typer.echo(
            f"\n{default} already leads to checkpoints in {holding}, and setup "
            f"will not move them for you.\nCopy them, verify, then swap the "
            f"symlink in:\n\n"
            f"    rsync -a --info=progress2 {holding}/ {target}/\n"
            f"    rsync -acn {holding}/ {target}/     # must print nothing\n"
            f"    rm {default} && ln -s {target} {default}\n"
            if default.is_symlink() else
            f"\n{default} already holds checkpoints, and setup will not move "
            f"them for you.\nCopy them, verify, then swap the symlink in:\n\n"
            f"    rsync -a --info=progress2 {holding}/ {target}/\n"
            f"    rsync -acn {holding}/ {target}/     # must print nothing\n"
            f"    mv {default} {default}.old && ln -s {target} {default}\n"
        )
        raise typer.Exit(1)
    if default.is_symlink():
        default.unlink()
    elif default.is_dir():
        default.rmdir()
    default.parent.mkdir(parents=True, exist_ok=True)
    default.symlink_to(target.resolve())
    _echo_check("models dir", True, f"{default} -> {target}{where}")


@app.command()
def setup(
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not prompt."),
    service: bool = typer.Option(True, help="Install and start the user service."),
    model_folder: str | None = typer.Option(
        None, "--model-folder",
        help="Where checkpoints live. Symlinked from ~/models, so nothing else "
             "needs to know. Honoured whatever disk it is on.",
    ),
) -> None:
    """Prepare this machine: system packages, engine, and the panel service.

    Everything ``uv tool install`` cannot do for itself. Safe to re-run -- each
    step is skipped when it is already done.
    """
    typer.echo(f"lllm3090 {__version__}\n")

    # 1. Hardware first: nothing else is worth doing on the wrong card.
    for name, check in (("os", preflight.check_os), ("gpu", preflight.check_gpu),
                        ("driver", preflight.check_driver)):
        ok, message = check()
        _echo_check(name, ok, message)
        if ok:
            continue
        if name == "gpu" and not yes:
            typer.echo(
                "\nEvery size and speed figure in the model catalogue was measured or\n"
                "derived for a 24 GB Ampere card. On other hardware the software runs\n"
                "and the numbers are wrong."
            )
            if not typer.confirm("Continue anyway?", default=False):
                raise typer.Exit(1)
        else:
            raise typer.Exit(1)

    # 2. Where the checkpoints go, before anything is downloaded into it.
    #    Asked here because it is the last moment the answer is cheap: once
    #    there are 180 GB in the wrong place, changing it means copying them.
    typer.echo("")
    _configure_models_folder(model_folder)

    # 3. System packages. Only what the engine actually needs -- uv brings its
    #    own Python, so there is no python3-venv to install.
    missing = _apt_missing()
    if missing:
        typer.echo(f"\nNeeds apt packages: {' '.join(missing)}")
        if not yes and not typer.confirm("Install them with sudo?", default=True):
            raise typer.Exit(1)
        sudo = [] if os.geteuid() == 0 else ["sudo"]
        env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
        subprocess.run([*sudo, "apt-get", "update", "-qq"], check=True, env=env)
        subprocess.run(
            [*sudo, "apt-get", "install", "-y", *missing], check=True, env=env
        )

    ok, message = preflight.check_vulkan()
    _echo_check("vulkan", ok, message)
    if not ok:
        raise typer.Exit(1)

    # 4. The engine.
    typer.echo("")
    install_engine(force=False)

    # 5. The panel.
    if service:
        typer.echo("")
        install_service(enable=True)

    typer.echo("\nReady. Open http://127.0.0.1:8080")
    if catalog.installed():
        typer.echo("Models already present: " + ", ".join(
            m["name"] for m in catalog.installed()
        ))
    else:
        typer.echo("Nothing has been downloaded yet -- choose a model in the panel.")
        typer.echo("Qwen3-8B (5 GB) confirms it works; Qwen3.6-35B-A3B (17.7 GB)")
        typer.echo("is the one to keep -- 126 tok/s and the longest context here,")
        typer.echo("212k per conversation. Only gpt-oss-20b decodes faster, at 160,")
        typer.echo("and it has less room to work in.")


@app.command("install-service")
def install_service(
    enable: bool = typer.Option(True, help="Enable and start it once written."),
) -> None:
    """Write the systemd user unit for the panel, and start it.

    Lives here rather than in the installer so that installing from PyPI --
    where there is no checkout to copy a unit file out of -- works identically.
    """
    unit = (
        resources.files("lllm3090.data")
        .joinpath("lllm3090-panel.service")
        .read_text()
    )
    # The unit needs an absolute path to this interpreter's console script.
    unit = unit.replace("@BIN@", str(pathlib.Path(sys.argv[0]).resolve()))
    target = pathlib.Path.home() / ".config/systemd/user/lllm3090-panel.service"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(unit)
    typer.echo(f"Wrote {target}")

    if not enable:
        return
    if not shutil.which("systemctl"):
        typer.echo("No systemctl here; start the panel with: lllm3090 panel")
        return
    unit_name = "lllm3090-panel.service"
    # `restart` rather than `enable --now`: after an upgrade the running panel is
    # executing code that was swapped out from under it and will read the new
    # data files with the old classes, which fails per-request without the
    # process ever exiting -- so Restart=on-failure does not save it. Restart
    # also starts it when stopped, so this covers a first install too.
    for args in (["daemon-reload"], ["enable", unit_name], ["restart", unit_name]):
        result = subprocess.run(
            ["systemctl", "--user", *args], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            typer.echo(
                f"systemctl --user {' '.join(args)} failed: {result.stderr.strip()}"
            )
            typer.echo("Start the panel manually with: lllm3090 panel")
            return
    typer.echo("Panel enabled and started on http://127.0.0.1:8080")


@app.command()
def panel(
    port: int = typer.Option(config.PANEL_PORT, help="Port to bind on loopback."),
) -> None:
    """Run the control panel."""
    import uvicorn

    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run("lllm3090.panel:app", host="127.0.0.1", port=port, log_level="warning")


@app.command()
def tui(
    url: str = typer.Option(
        None, help="Panel to drive. Default: the local one, if it is running."
    ),
) -> None:
    """The control panel on a text console, for a machine with no browser.

    Drives the panel over HTTP when it is running, and falls back to this
    process for everything that has a local answer -- which is all of it except
    downloading, since that is state the panel owns.
    """
    from .tui import run as run_tui

    # Both streams: curses draws on stdout but reads the keyboard from stdin,
    # so a piped stdin gets a screen nobody can drive.
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        typer.echo("lllm3090 tui needs a terminal. For a script, try: lllm3090 status")
        raise typer.Exit(1)
    raise typer.Exit(run_tui(url))


def claude_env(model: str, window: int, slots: int | None = None) -> dict[str, str]:
    """The environment Claude Code is launched with, in one place.

    Claude Code's variables are not a versioned contract: a release can add
    one, rename one, or start reading one that is currently ignored. Nothing
    can be done about that from here -- moving these strings into a data file
    would change where they are written, not whether they still match -- so
    what this does instead is make the whole mapping one value that can be
    printed, diffed and tested. ``lllm3090 claude --print-env`` prints exactly
    this, which is how you check it against a new Claude Code without running
    a session to find out.

    All three model slots point at the local model deliberately: switching
    with ``/model`` inside that session then stays local rather than falling
    back to the paid API.

    ``slots`` is what the engine says it can hold at once, from
    :func:`lllm3090.engine.served_slots`. ``None`` means it would not say,
    and the pool this project starts by default is assumed.
    """
    return {
        "ANTHROPIC_BASE_URL": config.ENGINE_URL,
        "ANTHROPIC_AUTH_TOKEN": "local",
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        # The PER-CONVERSATION window, not the pool the slots share.
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(window),
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32768",
        # One slot is the parent's, so the rest are what a fan-out has to fit
        # into. Claude Code's own default is 20, which against a two-slot pool
        # is a promise of twenty concurrent conversations where there is room
        # for two. Overshooting is not refused by llama.cpp -- it queues, and
        # each subagent prefills into whichever slot it lands in -- so without
        # this the limit is discovered as the model being slow.
        "CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS": str(
            max(1, (slots or config.DEFAULT_PARALLEL) - 1)
        ),
    }


#: Removed rather than blanked when Claude Code is launched. An empty
#: ANTHROPIC_API_KEY still counts as set, which makes Claude Code disable its
#: claude.ai connectors and say so on every launch.
CLAUDE_UNSET = "ANTHROPIC_API_KEY"


@app.command()
def claude(
    ctx: typer.Context,
    force: bool = typer.Option(
        False, "--force", help="Launch even if the window is too small."
    ),
    print_env: bool = typer.Option(
        False, "--print-env",
        help="Print the environment instead of launching, for eval or another harness.",
    ),
) -> None:
    """Launch Claude Code against the local engine.

    Sets Anthropic environment variables for one subprocess only -- nothing is
    written to ~/.claude/settings.json, so a plain ``claude`` elsewhere still
    reaches Anthropic on your normal account.
    """
    # With --print-env the only thing on stdout is the environment, so it can
    # be eval'd. Everything the user reads goes to stderr instead.
    def say(message: str) -> None:
        typer.echo(message, err=print_env)

    state = engine.status()
    if not state["answering"]:
        say(f"Nothing is answering on {config.ENGINE_URL}. Start a model first.")
        raise typer.Exit(1)
    model = state["model"] or "local"
    # Claude Code must be told the PER-CONVERSATION window, not the pool. Tell it
    # the pool and it will happily fill the whole thing, leaving nothing for the
    # subagents that share it. Ask launch_plan rather than plan: it is what sized
    # the running engine, so this number and the engine's cannot disagree about
    # whether a desktop is holding VRAM. Computing it here with plan's defaults
    # capped the agent at the desktop window while the engine served the larger
    # console one.
    window = str(catalog.launch_plan(model).per_session)
    # A window that cannot hold the harness's own prompt is not a small window,
    # it is a broken session: the first message fails with "prompt is too long".
    # Say so here rather than let the user discover it inside Claude Code.
    if int(window) <= config.AGENT_PROMPT_FLOOR and not force:
        desktop = hardware.graphical()
        alternatives = [
            f"{m.name} ({catalog.plan(m, desktop=desktop).per_session // 1024}k)"
            for m in catalog.load_catalog()
            if catalog.plan(m, desktop=desktop).per_session
            > config.AGENT_PROMPT_FLOOR * 1.5
        ]
        say(
            f"{model} serves {int(window) // 1024}k per conversation, but Claude "
            f"Code sends about {config.AGENT_PROMPT_FLOOR // 1000}k tokens of "
            "system prompt and tool definitions on every turn.\n"
            "The first message would fail with 'prompt is too long'.\n"
        )
        if alternatives:
            say("Models with room to work: " + ", ".join(alternatives))
        say("\nUse --force to try anyway.")
        raise typer.Exit(1)
    if int(window) < config.AGENT_PROMPT_FLOOR * 1.5:
        say(
            f"Warning: {int(window) // 1024}k per conversation leaves little room "
            f"after Claude Code's ~{config.AGENT_PROMPT_FLOOR // 1000}k system "
            "prompt. Expect frequent compaction."
        )

    slots = engine.served_slots()
    if slots == 1:
        # The plan now fills one conversation to the model's ceiling before
        # opening a second, so a single-slot engine is the normal outcome on a
        # VRAM-bound model -- and it is exactly the shape that has no room for
        # a subagent. Said here rather than at start, because this is the one
        # command that knows a fan-out is about to be attempted.
        say(
            "Note: this engine has one slot, so subagents run one at a time and "
            "each one evicts the parent's cached prefix -- the next parent turn "
            "then pays a full cold prefill.\nThat is the trade for a full-length "
            f"conversation. For fan-out instead, restart with: lllm3090 start "
            f"{model} --parallel {config.DEFAULT_PARALLEL}"
        )
    settings = claude_env(model, int(window), slots)
    if print_env:
        # Shell-shaped so it can drive a harness this command does not know
        # about: `eval "$(lllm3090 claude --print-env)"`, then run that tool.
        say(f"# Claude Code -> {model} @ {config.ENGINE_URL}")
        for key, value in settings.items():
            typer.echo(f"export {key}={shlex.quote(value)}")
        typer.echo(f"unset {CLAUDE_UNSET}")
        return

    typer.echo(f"Claude Code -> {model} @ {config.ENGINE_URL} (context {window})")
    env = dict(os.environ, **settings)
    env.pop(CLAUDE_UNSET, None)
    if not shutil.which("claude"):
        typer.echo("claude is not on PATH; install Claude Code first.")
        raise typer.Exit(1)
    sys.exit(subprocess.call(["claude", *ctx.args], env=env))


claude.__doc__ = claude.__doc__  # keep the help text after decoration


def main() -> None:
    app()


if __name__ == "__main__":
    main()
