"""One description of the machine, in one shape, served by ``/api/status``.

It lives here rather than inside the FastAPI handler for two reasons. The shape
is a documented surface -- see ``docs/reference/http.md`` -- so it is worth a
module of its own rather than an anonymous dict literal inside a route; and
assembled here it can be built and asserted on without standing up a client.

Nothing here starts, stops or downloads anything. Every call is a read of the
catalogue's arithmetic, a pidfile, or ``nvidia-smi``, which is what makes it
safe to call from anywhere, including from a process that is not the panel.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

from . import (
    catalog,
    config,
    downloads,
    engine,
    engines,
    gguf,
    hardware,
    speculation,
)
from ._version import __version__


def vram() -> dict[str, int] | None:
    """Used and total VRAM in MiB as ``nvidia-smi`` reports it, or None.

    None means the question could not be asked -- no driver, no card, a
    container -- rather than zero, which would render as an empty card.
    """
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
        used, total = (int(x.strip()) for x in out.split(","))
        return {"used_mb": used, "total_mb": total}
    except Exception:
        return None


def disk() -> dict[str, float] | None:
    """Free and total space where the models are kept, in GB, or None.

    The panel is where the decision to fetch 21 GB is actually made, so this
    is the number that has to be next to it. ``doctor`` checks the same thing,
    but by the time anyone runs ``doctor`` the download has already failed.

    The nearest existing parent is measured, not ``MODELS_DIR`` itself: on a
    fresh install the directory is created by the first download, and "no
    answer" would be the wrong one for a disk that has plenty of room.
    """
    d = config.MODELS_DIR
    while not d.exists() and d != d.parent:
        d = d.parent
    try:
        usage = shutil.disk_usage(d)
    except OSError:
        return None
    return {
        "free_gb": round(usage.free / 1e9, 1),
        "total_gb": round(usage.total / 1e9, 1),
    }


def installed_models() -> list[dict[str, Any]]:
    """What is on disk, with a stray GGUF carrying the window it would get.

    The panel draws one merged list, so a checkpoint the catalogue has never
    heard of sits in it beside the curated ones and has to say the same things
    they do. Its window comes from ``catalog.launch_plan`` -- the same call a
    start makes -- rather than from the renderer restating
    ``UNKNOWN_MODEL_CTX`` in its own language, which is how the console and the
    browser would end up describing one model two ways.
    """
    known = {m.name for m in catalog.load_catalog()}
    rows = []
    for m in catalog.installed():
        if m["name"] in known:
            # The catalogue's own row for this model already carries its plan.
            rows.append(m)
            continue
        plan = catalog.launch_plan(m["name"])
        rows.append(
            {**m, "kind": "gguf", "fits": True,
             "max_ctx": plan.per_session, "parallel": plan.parallel,
             # The same question the catalogue rows answer, and for an
             # uncatalogued checkpoint the file is the only source there is.
             # There is nothing to disagree with, so there is no note.
             "mtp": gguf.has_mtp(m["path"]), "mtp_declared": None,
             "mtp_note": ""}
        )
    return rows


def engine_choices() -> dict[str, Any]:
    """Which engines this machine has, which one serves, and what that buys.

    Two backends can sit on one disk and they are not interchangeable, so this
    is a choice rather than a detail: CUDA measures 1.55-1.60x the Vulkan
    engine on the dense 27B, costs about a seventh of the context window, and
    unlocks a speculation profile that *loses* on Vulkan. A front end that
    offered the switch without those three facts beside it would be offering a
    coin flip.

    ``locked`` says the choice is not the panel's to make: ``LLLM3090_LLAMA_DIR``
    outranks anything stored, and a control that silently did nothing would be
    worse than no control.
    """
    active = engines.active_dir()
    return {
        "locked": config.LLAMA_DIR_FROM_ENV,
        "active": active.name,
        "active_backend": engines.backend(active),
        # What the chosen engine does to every model alike. Beside the choice
        # rather than on each row: nine copies of one fact read as nine facts.
        "fixed": engine.fixed_flags(),
        "options": [
            {
                "id": c.id,
                "backend": c.backend,
                "installed": c.installed,
                "stale": c.stale,
                "active": c.path == active,
                # What picking this one changes, in the terms the panel is
                # already showing: speed, window, and which profiles apply.
                "profiles": [
                    p.name for p in speculation.PROFILES.values()
                    if p.allowed_on(c.backend)
                ],
            }
            for c in engines.choices()
        ],
    }


def snapshot() -> dict[str, Any]:
    """Everything the panel needs to draw the machine, in one call.

    Deliberately excludes the panel's ``busy`` and ``last``: those describe a
    server process rather than a machine, and this is the machine. The route
    adds them, which is also what keeps this callable from a plain test.
    """
    profile = hardware.detect()
    return {
        "version": __version__,
        "card": {
            "name": profile.name,
            "vram_gb": round(profile.vram_mib / 1024),
            "measured": profile.measured,
            "present": profile.present,
            "desktop": hardware.graphical(),
            "reference": hardware.reference().name,
        },
        "engine": engine.status(),
        "endpoint": config.ENGINE_URL,
        "vram": vram(),
        "disk": disk(),
        "installed": installed_models(),
        "catalog": catalog.catalog_for_panel(),
        "downloads": downloads.all_downloads(),
        "models_dir": str(config.MODELS_DIR),
        # Where each badge on a row goes when clicked. Sent rather than written
        # into the page so that one map is both what the browser follows and
        # what the test suite checks against the documentation source.
        "docs": {"base": config.DOCS_URL, "badges": config.BADGE_DOCS},
        "engines": engine_choices(),
    }
