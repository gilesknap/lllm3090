"""The ``lllm3090`` command."""

from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from importlib import resources
from pathlib import Path

import typer

from . import catalog, config, engine, hardware, preflight
from ._version import __version__

# The engine build is pinned, not tracked. "Latest" would silently change the
# thing every figure in the model catalogue was measured against.
LLAMA_BUILD = "b10628"
LLAMA_ASSET = f"llama-{LLAMA_BUILD}-bin-ubuntu-vulkan-x64.tar.gz"
LLAMA_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/{LLAMA_ASSET}"
)
LLAMA_SHA256 = "c64b6d5820ea6dc3227495e2c30c397fb73c24158291cfb7ef99892a708605a6"

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

    typer.echo(f"Downloading {LLAMA_ASSET} ...")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / LLAMA_ASSET
        urllib.request.urlretrieve(LLAMA_URL, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != LLAMA_SHA256:
            typer.echo(
                f"Checksum mismatch!\n"
                f"  expected {LLAMA_SHA256}\n  got      {digest}"
            )
            raise typer.Exit(1)
        typer.echo("Checksum verified.")
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
    typer.echo(f"Engine installed at {target}")


@app.command()
def models() -> None:
    """List the curated catalogue and what is already downloaded."""
    profile = hardware.detect()
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
    typer.echo("")
    header = f"{'MODEL':<24}{'SIZE':>8}{'KIND':>10}{'CONTEXT':>12}  {'STATE':<12}SPEED"
    typer.echo(header)
    for row in catalog.catalog_for_panel():
        state = (
            "installed" if row["installed"]
            else ("fits" if row["fits"] else "too big")
        )
        speed = f"~{row['expected_tok_s']} tok/s" if row["expected_tok_s"] else "-"
        if row["verified"]:
            speed += " (measured)" if row["speed_applies"] else " (other card)"
        ctx = (
            f"{row['max_ctx'] // 1024}k x{row['parallel']}" if row["fits"] else "-"
        )
        kind = row["kind"] + ("+vis" if row["vision"] else "")
        typer.echo(
            f"{row['name']:<24}{row['gb']:>7.1f}G{kind:>10}{ctx:>12}  "
            f"{state:<12}{speed}"
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
        None, help="Conversations sharing the pool. Default: 2, for subagents."
    ),
) -> None:
    """Start the engine on an installed model."""
    entry = next((m for m in catalog.installed() if m["name"] == model), None)
    if entry is None:
        typer.echo(f"{model!r} is not installed. Try: lllm3090 models")
        raise typer.Exit(1)
    parallel = parallel or config.DEFAULT_PARALLEL
    known = next((m for m in catalog.load_catalog() if m.name == model), None)
    if ctx is None:
        if known is not None:
            p = catalog.plan(known, parallel)
            ctx, parallel = p.pool, p.parallel
            typer.echo(f"Context plan: {p.summary}")
        else:
            ctx = 32768 * parallel
    engine.stop()
    template = known.chat_template if known else None
    ok, detail = engine.start(
        entry["path"], model, ctx, parallel, 300, template, entry.get("mmproj")
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


@app.command()
def setup(
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not prompt."),
    service: bool = typer.Option(True, help="Install and start the user service."),
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

    # 2. System packages. Only what the engine actually needs -- uv brings its
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

    # 3. The engine.
    typer.echo("")
    install_engine(force=False)

    # 4. The panel.
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
def claude(
    ctx: typer.Context,
    force: bool = typer.Option(
        False, "--force", help="Launch even if the window is too small."
    ),
) -> None:
    """Launch Claude Code against the local engine.

    Sets Anthropic environment variables for one subprocess only -- nothing is
    written to ~/.claude/settings.json, so a plain ``claude`` elsewhere still
    reaches Anthropic on your normal account.
    """
    state = engine.status()
    if not state["answering"]:
        typer.echo(f"Nothing is answering on {config.ENGINE_URL}. Start a model first.")
        raise typer.Exit(1)
    model = state["model"] or "local"
    known = next((m for m in catalog.load_catalog() if m.name == model), None)
    # Claude Code must be told the PER-CONVERSATION window, not the pool. Tell it
    # the pool and it will happily fill the whole thing, leaving nothing for the
    # subagents that share it.
    window = str(catalog.plan(known).per_session if known else 32768)
    # A window that cannot hold the harness's own prompt is not a small window,
    # it is a broken session: the first message fails with "prompt is too long".
    # Say so here rather than let the user discover it inside Claude Code.
    if int(window) <= config.AGENT_PROMPT_FLOOR and not force:
        alternatives = [
            f"{m.name} ({catalog.plan(m).per_session // 1024}k)"
            for m in catalog.load_catalog()
            if catalog.plan(m).per_session > config.AGENT_PROMPT_FLOOR * 1.5
        ]
        typer.echo(
            f"{model} serves {int(window) // 1024}k per conversation, but Claude "
            f"Code sends about {config.AGENT_PROMPT_FLOOR // 1000}k tokens of "
            "system prompt and tool definitions on every turn.\n"
            "The first message would fail with 'prompt is too long'.\n"
        )
        if alternatives:
            typer.echo("Models with room to work: " + ", ".join(alternatives))
        typer.echo("\nUse --force to try anyway.")
        raise typer.Exit(1)
    if int(window) < config.AGENT_PROMPT_FLOOR * 1.5:
        typer.echo(
            f"Warning: {int(window) // 1024}k per conversation leaves little room "
            f"after Claude Code's ~{config.AGENT_PROMPT_FLOOR // 1000}k system "
            "prompt. Expect frequent compaction."
        )

    typer.echo(f"Claude Code -> {model} @ {config.ENGINE_URL} (context {window})")
    env = dict(
        os.environ,
        ANTHROPIC_BASE_URL=config.ENGINE_URL,
        ANTHROPIC_AUTH_TOKEN="local",
        ANTHROPIC_MODEL=model,
        ANTHROPIC_DEFAULT_OPUS_MODEL=model,
        ANTHROPIC_DEFAULT_SONNET_MODEL=model,
        ANTHROPIC_DEFAULT_HAIKU_MODEL=model,
        CLAUDE_CODE_SUBAGENT_MODEL=model,
        CLAUDE_CODE_MAX_CONTEXT_TOKENS=window,
        CLAUDE_CODE_MAX_OUTPUT_TOKENS="32768",
    )
    # Remove rather than blank it: an empty ANTHROPIC_API_KEY still counts as
    # set, which makes Claude Code disable its claude.ai connectors and say so
    # on every launch. The auth token above is what this endpoint uses.
    env.pop("ANTHROPIC_API_KEY", None)
    if not shutil.which("claude"):
        typer.echo("claude is not on PATH; install Claude Code first.")
        raise typer.Exit(1)
    sys.exit(subprocess.call(["claude", *ctx.args], env=env))


claude.__doc__ = claude.__doc__  # keep the help text after decoration


def main() -> None:
    app()


if __name__ == "__main__":
    main()
