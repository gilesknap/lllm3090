"""The ``lllm3090`` command."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import typer

from . import catalog, config, engine, preflight
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
    header = f"{'MODEL':<24}{'SIZE':>8}{'KIND':>7}{'CONTEXT':>12}  {'STATE':<12}SPEED"
    typer.echo(header)
    for row in catalog.catalog_for_panel():
        state = (
            "installed" if row["installed"]
            else ("fits" if row["fits"] else "too big")
        )
        speed = f"~{row['expected_tok_s']} tok/s" if row["expected_tok_s"] else "-"
        if row["verified"]:
            speed += " (measured)"
        ctx = (
            f"{row['max_ctx'] // 1024}k x{row['parallel']}" if row["fits"] else "-"
        )
        typer.echo(
            f"{row['name']:<24}{row['gb']:>7.1f}G{row['kind']:>7}{ctx:>12}  "
            f"{state:<12}{speed}"
        )


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
    if ctx is None:
        known = next((m for m in catalog.load_catalog() if m.name == model), None)
        if known is not None:
            p = catalog.plan(known, parallel)
            ctx, parallel = p.pool, p.parallel
            typer.echo(f"Context plan: {p.summary}")
        else:
            ctx = 32768 * parallel
    engine.stop()
    ok, detail = engine.start(entry["path"], model, ctx, parallel)
    typer.echo(detail)
    raise typer.Exit(0 if ok else 1)


@app.command()
def stop() -> None:
    """Stop the engine and free the VRAM."""
    typer.echo(engine.stop()[1])


@app.command()
def panel(
    port: int = typer.Option(config.PANEL_PORT, help="Port to bind on loopback."),
) -> None:
    """Run the control panel."""
    import uvicorn

    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run("lllm3090.panel:app", host="127.0.0.1", port=port, log_level="warning")


@app.command()
def claude(ctx: typer.Context) -> None:
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
    typer.echo(f"Claude Code -> {model} @ {config.ENGINE_URL} (context {window})")
    env = dict(
        os.environ,
        ANTHROPIC_BASE_URL=config.ENGINE_URL,
        ANTHROPIC_AUTH_TOKEN="local",
        ANTHROPIC_API_KEY="",
        ANTHROPIC_MODEL=model,
        ANTHROPIC_DEFAULT_OPUS_MODEL=model,
        ANTHROPIC_DEFAULT_SONNET_MODEL=model,
        ANTHROPIC_DEFAULT_HAIKU_MODEL=model,
        CLAUDE_CODE_SUBAGENT_MODEL=model,
        CLAUDE_CODE_MAX_CONTEXT_TOKENS=window,
        CLAUDE_CODE_MAX_OUTPUT_TOKENS="32768",
    )
    if not shutil.which("claude"):
        typer.echo("claude is not on PATH; install Claude Code first.")
        raise typer.Exit(1)
    sys.exit(subprocess.call(["claude", *ctx.args], env=env))


claude.__doc__ = claude.__doc__  # keep the help text after decoration


def main() -> None:
    app()


if __name__ == "__main__":
    main()
