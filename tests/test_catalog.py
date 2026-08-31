"""The fit arithmetic is what makes the panel's promises true or false."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lllm3090 import catalog, config, hardware


def test_every_catalogue_entry_is_well_formed():
    models = catalog.load_catalog()
    assert models, "catalogue is empty"
    for m in models:
        assert m.kind in {"dense", "moe"}, f"{m.id} has odd kind {m.kind}"
        assert m.kv_kib_per_token > 0, f"{m.id} has no KV cost"
        assert m.max_ctx > 0, f"{m.id} has no context ceiling"
        assert m.file.endswith(".gguf"), f"{m.id} does not name a GGUF"
        assert "/" in m.repo, f"{m.id} repo is not owner/name"


def test_every_catalogue_entry_fits_the_target_card():
    """The catalogue is curated. An entry that does not fit does not belong."""
    for m in catalog.load_catalog():
        assert catalog.fit(m).fits, f"{m.name} does not fit a 24 GB card"


def test_context_never_exceeds_the_rope_ceiling():
    for m in catalog.load_catalog():
        assert catalog.plan(m).per_session <= m.max_ctx


def test_q8_cache_buys_context():
    for m in catalog.load_catalog():
        f = catalog.fit(m)
        assert f.pool_q8 >= f.pool_f16


def test_plan_never_exceeds_vram_or_the_rope_ceiling():
    """The two limits that must both hold, for every model, at every slot count."""
    for m in catalog.load_catalog():
        for parallel in (1, 2, 4):
            p = catalog.plan(m, parallel)
            assert p.per_session <= m.max_ctx, f"{m.id} would extrapolate RoPE"
            assert p.pool <= catalog.fit(m).pool_q8, f"{m.id} pool exceeds VRAM"
            assert p.pool == p.per_session * p.parallel


def test_a_plan_fits_inside_what_the_driver_leaves():
    """The nameplate capacity is not the budget, and planning as if it were is
    what put a GPU into ``vk::DeviceLostError``.

    The driver takes its reservation before any process allocates, so a plan
    that spends up to ``vram_mib`` is already over by that much. Everything the
    engine needs -- weights, projector, the whole preallocated cache, and the
    workspace the compute buffers grow into -- has to sit inside what is left,
    with the reserves still intact rather than eaten to make the sums work.
    """
    profile = hardware.reference()
    ceiling = profile.vram_mib - profile.driver_reserve_mib
    for m in catalog.load_catalog():
        for desktop in (True, False):
            p = catalog.plan(m, desktop=desktop, profile=profile)
            held = config.WORKSPACE_RESERVE_MIB
            if m.vision:
                held += config.VISION_WORKSPACE_RESERVE_MIB
            if desktop:
                held += config.DESKTOP_RESERVE_MIB
            need = catalog.vram_needed_mib(m, p.pool) + held
            assert need <= ceiling, (
                f"{m.id} (desktop={desktop}) plans {p.pool} tokens needing "
                f"{need:.0f} MiB against {ceiling} MiB the card can hand out"
            )


def test_the_driver_reservation_is_taken_off_the_top():
    """A profile may never offer more than the card can actually allocate."""
    for profile in hardware.load_profiles():
        assert profile.driver_reserve_mib > 0, (
            f"{profile.id} claims the driver reserves nothing"
        )
        for desktop in (True, False):
            budget = profile.usable_vram_mib(desktop)
            assert budget <= profile.vram_mib - profile.driver_reserve_mib


def test_a_split_has_to_earn_its_place():
    """One long conversation is the default; a split must buy enough to justify it.

    Dividing the pool does not create capacity -- it shortens every
    conversation and buys concurrency with the difference. So the automatic
    plan only splits where the total usable context rises by
    ``config.SLOT_SPLIT_GAIN``, which happens when the model's RoPE ceiling
    would otherwise strand a large part of the cache.
    """
    for m in catalog.load_catalog():
        for desktop in (True, False):
            f = catalog.fit(m, desktop)
            if not f.fits:
                continue
            p = catalog.plan(m, desktop=desktop)
            if p.parallel == 1:
                continue
            alone = min(f.pool_q8, m.max_ctx)
            assert f.pool_q8 >= config.SLOT_SPLIT_GAIN * alone, (
                f"{m.id} (desktop={desktop}) split into {p.parallel} for a pool "
                f"only {f.pool_q8 / alone:.2f}x what one conversation could use"
            )


def test_a_further_slot_is_never_taken_when_it_costs_more_than_it_recovers():
    """The slot count is a local optimum, not simply as many as fit.

    Taking every slot the pool allows produces a cliff: past the split
    threshold a *larger* pool yields a *shorter* conversation, which is how
    Qwen3.6-35B-A3B ended up offering 256k on a desktop and 184k headless. A
    further slot is taken only while it recovers more stranded cache than it
    costs in window.
    """
    for m in catalog.load_catalog():
        for desktop in (True, False):
            f = catalog.fit(m, desktop)
            if not f.fits:
                continue
            p = catalog.plan(m, desktop=desktop)
            if p.parallel >= config.MAX_AUTO_PARALLEL:
                continue
            nxt = min(m.max_ctx, f.pool_q8 // (p.parallel + 1))
            nxt = max(1024, (nxt // 1024) * 1024)
            if nxt <= 1024:
                continue
            gained = ((p.parallel + 1) * nxt) / (p.parallel * p.per_session)
            lost = p.per_session / nxt
            assert gained <= lost, (
                f"{m.id} (desktop={desktop}) stopped at {p.parallel} slots but "
                f"one more would have gained {gained:.2f}x for {lost:.2f}x window"
            )


def test_small_models_get_their_whole_window():
    """Spare VRAM goes to concurrency, not to context the model cannot use."""
    small = next(m for m in catalog.load_catalog() if m.name == "Qwen3-8B")
    p = catalog.plan(small)
    assert p.per_session == small.max_ctx
    assert p.capped_by == "rope"


def test_rope_capped_models_are_given_the_free_slots():
    """If the architecture runs out before the card does, spend the rest on slots.

    Rope-capping says the *window* is bounded by the architecture rather than by
    VRAM. It does not promise a surplus large enough for another whole slot: a
    model can be rope-capped with exactly one window's worth of pool, and that
    is an honest plan rather than a missed opportunity. What must hold is that
    no slot is short-changed, and that a whole spare window is never left unused.
    """
    for m in catalog.load_catalog():
        f = catalog.fit(m)
        p = catalog.plan(m)
        if p.capped_by != "rope":
            continue
        # Free means free: every slot still gets the whole window.
        assert p.per_session == m.max_ctx
        assert 1 <= p.parallel <= config.MAX_AUTO_PARALLEL
        affordable = min(f.pool_q8 // m.max_ctx, config.MAX_AUTO_PARALLEL)
        assert p.parallel == max(1, affordable), (
            f"{m.id} can seat {affordable} whole windows but was given "
            f"{p.parallel}"
        )


def test_an_explicit_slot_count_is_honoured_exactly():
    """Auto-expansion must not override what the user asked for."""
    for m in catalog.load_catalog():
        for n in (1, 2, 3):
            assert catalog.plan(m, n).parallel == n


def test_headless_is_never_worse_than_desktop():
    for m in catalog.load_catalog():
        assert catalog.fit(m, desktop=False).pool_q8 >= catalog.fit(m).pool_q8


def test_a_model_larger_than_the_card_does_not_fit():
    huge = catalog.Model(
        id="huge", name="Huge", repo="x/y", file="z.gguf",
        size_gb=40.0, kind="dense", params="", kv_kib_per_token=64,
        max_ctx=8192,
    )
    f = catalog.fit(huge)
    assert not f.fits
    assert f.headline == "does not fit"


@pytest.mark.parametrize("desktop", [True, False])
def test_budget_is_positive_and_below_the_card(desktop: bool):
    from lllm3090 import hardware

    for profile in hardware.load_profiles():
        budget = profile.usable_vram_mib(desktop)
        assert 0 < budget < profile.vram_mib, profile.id


def test_the_agent_floor_separates_usable_models_from_unusable():
    """A window below the harness's own prompt cannot run it at all.

    Qwen3-8B is the case that matters: it is the model the tutorial sends people
    to first, and its 32k ceiling is under Claude Code's ~40k system prompt.
    """
    usable, unusable = [], []
    for m in catalog.load_catalog():
        window = catalog.plan(m).per_session
        (usable if window > config.AGENT_PROMPT_FLOOR else unusable).append(m.name)

    assert "Qwen3-8B" in unusable, "the 8B should be flagged as too small for agents"
    assert "Qwen3.8-27B" in usable
    assert "Qwen3.6-35B-A3B" in usable
    assert usable, "no model can run an agent harness -- the catalogue is broken"


def test_the_recommendation_agrees_with_the_measurements():
    """Exactly one model is recommended, and it is defensible on the numbers.

    The docs recommended the slowest model in the catalogue for a while, because
    the advice was written before anything was measured and nothing forced the
    two to agree. This is that forcing function.
    """
    models = catalog.load_catalog()
    recommended = [m for m in models if "recommended" in m.tags]
    assert len(recommended) == 1, f"expected one recommendation, got {recommended}"
    pick = recommended[0]

    assert pick.verified, "recommending a model whose speed was never measured"
    assert catalog.fit(pick).fits, "recommending a model that does not fit"
    assert catalog.plan(pick).per_session > config.AGENT_PROMPT_FLOOR, (
        "the recommended model cannot hold an agent harness's system prompt"
    )

    faster = [
        m.name for m in models
        if m.verified
        and catalog.fit(m).fits
        and (m.expected_tok_s or 0) > (pick.expected_tok_s or 0)
        and catalog.plan(m).per_session >= catalog.plan(pick).per_session
    ]
    # The exception the docstring allows, made enforceable: a faster model with
    # at least as much context may be passed over, but only if the recommended
    # entry's own note names it and says why -- in the catalogue, where the
    # panel shows it, rather than in a commit message nobody reads.
    unexplained = [n for n in faster if n not in pick.notes]
    assert not unexplained, (
        f"{unexplained} are faster than {pick.name} with no less context -- "
        f"either the recommendation moves, or {pick.name}'s notes name them "
        "and say why it stays"
    )


def test_a_projector_is_counted_against_the_budget():
    """Vision is not free: the projector sits in VRAM like any other weights.

    Leaving it out of the arithmetic would promise context the projector has
    already spent, which is the failure this whole module exists to prevent.
    """
    for m in catalog.load_catalog():
        if not m.vision:
            continue
        text_only = replace(m, mmproj=None, mmproj_gb=0.0)
        assert m.weights_mib > text_only.weights_mib, f"{m.id} projector is free?"
        assert catalog.plan(m).pool <= catalog.plan(text_only).pool, (
            f"{m.id} claims more context with a projector loaded than without"
        )


def test_a_vision_entry_holds_back_workspace_for_the_vision_tower():
    """A projector's file size is not what it costs to run.

    Measured on a 3090: Gemma-4-26B-A4B's 1.19 GB projector occupied 1376 MiB,
    and at a full KV pool the engine loaded, reported itself healthy, and failed
    *every* request with `vk::Device::allocateMemory: ErrorOutOfDeviceMemory`.
    Counting only the file promised context the card could not serve.
    """
    for m in catalog.load_catalog():
        if not m.vision:
            continue
        as_text = replace(m, mmproj=None, mmproj_gb=m.mmproj_gb)
        held_back = catalog.fit(as_text).pool_q8 - catalog.fit(m).pool_q8
        assert held_back > 0, (
            f"{m.id} reserves nothing extra for its vision tower"
        )


def test_a_vision_entry_names_a_projector_and_a_size():
    """One without the other silently under-counts VRAM or fails at launch."""
    for m in catalog.load_catalog():
        assert bool(m.mmproj) == (m.mmproj_gb > 0), (
            f"{m.id} has mmproj={m.mmproj!r} but mmproj_gb={m.mmproj_gb}"
        )
        if m.mmproj:
            assert m.mmproj.endswith(".gguf"), f"{m.id} projector is not a GGUF"


def test_a_projector_on_disk_is_never_served_as_the_model(tmp_path):
    """--model pointed at a projector starts an engine that answers nothing.

    Sorting matters: 'Gemma-4-26B...gguf' sorts before 'mmproj-F16.gguf' but a
    lower-cased model name would not, and the first GGUF in the directory is
    what gets served.
    """
    d = tmp_path / "some-model"
    d.mkdir()
    (d / "mmproj-F16.gguf").write_bytes(b"proj")
    (d / "zz-weights-Q4_K_XL.gguf").write_bytes(b"weights")
    entry = catalog.installed(models_dir=tmp_path)[0]
    assert entry["path"].endswith("zz-weights-Q4_K_XL.gguf")
    assert entry["mmproj"].endswith("mmproj-F16.gguf")


def test_a_directory_holding_only_a_projector_is_not_a_model(tmp_path):
    d = tmp_path / "orphan"
    d.mkdir()
    (d / "mmproj-F16.gguf").write_bytes(b"proj")
    assert catalog.installed(models_dir=tmp_path) == []


def test_linger_is_reported_honestly_when_it_cannot_be_determined(monkeypatch):
    """A missing loginctl is not a failure; a definite "no" is.

    The panel is a user unit, so without lingering it stops with the last
    session -- including the moment you isolate to multi-user.target, which is
    exactly when the headless how-to tells you to do it.
    """
    from lllm3090 import preflight

    monkeypatch.setattr(preflight.shutil, "which", lambda _: None)
    ok, msg = preflight.check_linger()
    assert ok and "loginctl" in msg

    monkeypatch.setattr(preflight.shutil, "which", lambda _: "/bin/loginctl")

    class R:
        stdout = "no\n"

    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: R())
    ok, msg = preflight.check_linger()
    assert not ok, "lingering off must be reported as a problem"
    assert "enable-linger" in msg, "the message must say how to fix it"


def test_headless_never_leaves_you_worse_off_than_a_desktop():
    """Freeing the compositor's VRAM can only add cache, never remove it.

    The pool is the hard part of that and never regresses. The per-conversation
    window is allowed to, but only in exchange for more total capacity -- which
    is the trade the split rule exists to make. ``Qwen3.6-35B-A3B-MTP`` is the
    live case: headless its pool holds 1.977 windows, so two slots get 259072
    each against a desktop's single 262144. Three thousand tokens, 1.2%, for
    twice the capacity.

    What must never happen is a plan that is worse on *both* counts, which is
    what "more VRAM made it worse" would actually mean. An earlier version of
    the rule did exactly that -- 256k on a desktop and 184k headless -- by
    taking every slot the pool allowed.
    """
    for m in catalog.load_catalog():
        d = catalog.plan(m, desktop=True)
        h = catalog.plan(m, desktop=False)
        assert h.pool >= d.pool, f"{m.name} claims less cache with more free"
        assert h.per_session >= d.per_session or h.pool > d.pool, (
            f"{m.name} gives a shorter conversation headless "
            f"({h.per_session} vs {d.per_session}) and no more total capacity"
        )


def test_an_unknown_session_is_assumed_to_be_a_desktop(monkeypatch):
    """Guessing headless would hand out context the card does not have.

    Both failure modes are silent, but they are not symmetric: over-reserving
    costs context, while under-reserving produces an engine that loads, reports
    itself healthy and fails every request.
    """
    from lllm3090 import hardware

    monkeypatch.setattr(hardware.shutil, "which", lambda _: None)
    assert hardware.graphical() is True

    def boom(*a, **k):
        raise OSError("systemctl exploded")

    monkeypatch.setattr(hardware.shutil, "which", lambda _: "/bin/systemctl")
    monkeypatch.setattr(hardware.subprocess, "run", boom)
    assert hardware.graphical() is True


def test_startup_cost_holds_back_the_workspace_the_plan_held_back():
    """What must be free is what the plan reserved, not just what it allocates.

    ``fit()`` subtracts the workspace -- and the vision tower's buffers on top
    of it -- before deciding on a context, so a check that compares free VRAM
    against the load alone approves a plan the card cannot serve.
    """
    for m in catalog.load_catalog():
        held = config.WORKSPACE_RESERVE_MIB
        if m.vision:
            held += config.VISION_WORKSPACE_RESERVE_MIB
        loaded = catalog.vram_needed_mib(m, 65536)
        assert catalog.startup_vram_mib(m, 65536) == loaded + held


def test_the_gap_between_loading_and_serving_is_warned_about(monkeypatch):
    """The regression: free VRAM above the load and below the requirement.

    This is not a corner case, it is the shape of the failure -- the engine
    loads, reports itself healthy, and then fails every request out of device
    memory, because the room it needed to work in was never there.
    """
    m = next(x for x in catalog.load_catalog() if x.vision)
    ctx = 65536
    loaded = catalog.vram_needed_mib(m, ctx)
    required = catalog.startup_vram_mib(m, ctx)
    assert required > loaded, "a vision model must hold back workspace"

    between = int((loaded + required) / 2)
    monkeypatch.setattr(catalog.hardware, "free_vram_mib", lambda: between)
    warning = catalog.free_vram_warning(m, ctx)
    assert warning is not None, "a plan that loads and then starves is a warning"
    assert f"{required / 1024:.1f} GB" in warning, (
        "the warning must quote what serving needs, not what loading needs"
    )

    monkeypatch.setattr(
        catalog.hardware, "free_vram_mib", lambda: int(required) + 1024
    )
    assert catalog.free_vram_warning(m, ctx) is None, "room to spare is not news"


def test_nothing_is_claimed_when_the_card_cannot_be_measured(monkeypatch):
    """No nvidia-smi is no measurement, and a guess here would be a false alarm
    on every start on a machine without one."""
    m = catalog.load_catalog()[0]
    monkeypatch.setattr(catalog.hardware, "free_vram_mib", lambda: None)
    assert catalog.free_vram_warning(m, 65536) is None
    monkeypatch.setattr(catalog.hardware, "free_vram_mib", lambda: 1)
    assert catalog.free_vram_warning(None, 65536) is None, (
        "an uncatalogued GGUF has no known cache cost to compare against"
    )


def test_vram_needed_counts_the_projector_and_the_q8_cache():
    m = next(x for x in catalog.load_catalog() if x.vision)
    bare = catalog.vram_needed_mib(m, 0, backend="vulkan")
    assert bare == m.weights_mib, "no cache should cost only the weights"
    # A q8 cache is 0.53125 of the f16 figure the catalogue stores -- 34 bytes
    # per 32 values, not half -- plus what the engine holds on top of the
    # nominal tensor size.
    grown = catalog.vram_needed_mib(m, 2048, backend="vulkan") - bare
    nominal = 2048 * (m.kv_kib_per_token * config.Q8_0_RATIO) / 1024
    assert grown == pytest.approx(nominal * config.ALLOCATOR_OVERHEAD)
    assert grown > nominal, "the overhead factor must never flatter the cache"
    assert grown > 2048 * (m.kv_kib_per_token / 2) / 1024, (
        "and it must never come out under the half-of-f16 figure the whole "
        "catalogue used to be priced at, which was itself an under-count"
    )


# --- hardware profiles ------------------------------------------------------
# The point of profiles is that the arithmetic travels to cards nobody here
# owns. These are the tests that can be written without one.


@pytest.mark.parametrize(
    "profile_id",
    ["rtx-3090", "rtx-4090", "rtx-5090", "rtx-pro-6000-blackwell"],
)
def test_a_card_of_24gb_or_more_fits_the_whole_catalogue(profile_id):
    """The catalogue is curated for a 3090, so nothing in it may fail on one.

    Counting entries would make this a census that needs editing every time a
    model is added. The invariant is that curation means what it says.
    """
    from lllm3090 import hardware

    profile = next(p for p in hardware.load_profiles() if p.id == profile_id)
    missing = [
        m.name for m in catalog.load_catalog()
        if not catalog.fit(m, profile=profile).fits
    ]
    assert not missing, f"{profile_id} cannot fit curated entries: {missing}"


def test_a_16gb_card_is_told_the_truth_about_what_does_not_fit():
    """The 5080 has less memory than a card from 2020, and must say so.

    This is the case the profile work exists for: before it, the panel would
    have offered a 15.4 GB model on a 16 GB card and let someone find out after
    the download.
    """
    from lllm3090 import hardware

    small = next(p for p in hardware.load_profiles() if p.id == "rtx-5080")
    models = catalog.load_catalog()
    fits = [m.name for m in models if catalog.fit(m, profile=small).fits]
    assert fits, "a 16 GB card should still run something"
    assert len(fits) < len(models), "16 GB cannot hold everything curated for 24"
    assert "Qwen3.8-27B" not in fits, "15.4 GB of weights leaves no usable cache"
    assert "gpt-oss-20b" in fits, "the small models are the point of this card"


def test_a_smaller_card_never_promises_more_context():
    from lllm3090 import hardware

    profiles = {p.id: p for p in hardware.load_profiles()}
    small, big = profiles["rtx-5080"], profiles["rtx-5090"]
    for m in catalog.load_catalog():
        s = catalog.plan(m, profile=small)
        b = catalog.plan(m, profile=big)
        assert s.pool <= b.pool, f"{m.name} claims more on the smaller card"


def test_exactly_one_profile_carries_the_measurements():
    from lllm3090 import hardware

    measured = [p for p in hardware.load_profiles() if p.measured]
    assert len(measured) == 1, "speeds belong to one card, not several"
    assert measured[0].id == config.REFERENCE_PROFILE


def test_a_card_sharing_a_size_is_not_mistaken_for_the_measured_one(monkeypatch):
    """A 3090 Ti is 24 GiB at compute 8.6 and is still not the reference card.

    Matching on capacity alone would hand it the rtx-3090 profile and print its
    speeds as measurements taken on the card in front of you.
    """
    from lllm3090 import hardware

    monkeypatch.setattr(hardware, "_smi", lambda q: {
        "name": "NVIDIA GeForce RTX 3090 Ti", "compute_cap": "8.6",
        "memory.total": "24576",
    }.get(q))
    p = hardware.detect()
    assert p.id != config.REFERENCE_PROFILE
    assert p.detected and not p.measured
    assert p.vram_mib == 24576


def test_no_gpu_does_not_borrow_the_reference_card_s_measurements(monkeypatch):
    """With no GPU, capacity may be borrowed to read the catalogue; speed may not."""
    from lllm3090 import hardware

    monkeypatch.setattr(hardware, "_smi", lambda q: None)
    p = hardware.detect()
    assert not p.present, "a machine with no GPU must not report one"
    assert not p.measured, "no card here, so no figure was measured on it"
    assert p.vram_mib == hardware.reference().vram_mib
    # The catalogue is still inspectable, which is the whole point of the fallback.
    assert catalog.load_catalog()


def test_an_unknown_card_is_honest_rather_than_absent(monkeypatch):
    """An unrecognised GPU must still compute fit, and must not claim speed."""
    from lllm3090 import hardware

    monkeypatch.setattr(hardware, "_smi", lambda q: {
        "name": "NVIDIA GeForce RTX 9090", "compute_cap": "13.0",
        "memory.total": "49152",
    }.get(q))
    p = hardware.detect()
    assert p.detected and not p.measured
    assert p.vram_mib == 49152
    # Fit still works, because capacity is all the arithmetic needs.
    assert all(catalog.fit(m, profile=p).fits for m in catalog.load_catalog())


def test_more_slots_than_pages_is_refused_rather_than_overcommitted():
    """A slot's share is rounded up to a page, so enough slots overrun the pool.

    Ask for more slots than the cache has pages and every slot still gets one,
    which makes the plan's pool larger than the cache that exists. The engine
    then starts, allocates, and dies of VRAM exhaustion -- with a plan that
    looked like any other. Refusing costs nothing and happens before launch.
    """
    heavy = catalog.Model(
        id="h", name="H", repo="x/y", file="y.gguf", size_gb=20.4, kind="dense",
        params="", kv_kib_per_token=64, max_ctx=262144,
    )
    card = hardware.Profile(
        id="t", name="t", compute_capability="8.6", vram_mib=24576,
        bandwidth_gbs=0,
    )
    pool = catalog.fit(heavy, True, card).pool_q8
    seatable = pool // 1024
    assert seatable >= 1, "the fixture must fit at all for this to mean anything"

    ok = catalog.plan(heavy, seatable, desktop=True, profile=card)
    assert ok.pool <= pool, "a plan may never promise more cache than exists"

    with pytest.raises(ValueError, match="do not fit"):
        catalog.plan(heavy, seatable + 1, desktop=True, profile=card)


# ---------------------------------------------------------------------------
# What a speed is a speed of
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# What a token of KV cache costs
# ---------------------------------------------------------------------------

#: The reference card, with the driver reserve this machine actually reports.
#: Every measured ceiling below was taken on it.
REFERENCE = hardware.Profile(
    id="rtx-3090", name="NVIDIA GeForce RTX 3090", compute_capability="8.6",
    vram_mib=24576, bandwidth_gbs=936, measured=True, driver_reserve_mib=451,
)

#: The longest pool of the dense 27B each backend actually held, found by
#: loading until the server died, desktop session running. The planner must sit
#: under these -- at them it would have no margin at all, which is the state
#: CUDA was in before any of this.
CEILINGS = {"vulkan": 208896, "cuda": 172032}


def _dense() -> catalog.Model:
    return next(m for m in catalog.load_catalog() if m.name == "Qwen3.8-27B")


def test_q8_is_not_half_of_f16():
    """34 bytes per 32 values -- 32 quantised bytes and an f16 scale -- so
    0.53125x. Halving it under-counted the cache by 6% on every model."""
    assert config.Q8_0_RATIO == pytest.approx(0.53125)


def test_the_price_of_a_token_is_conservative_where_it_was_measured():
    """The dense 27B is linear in context to within 0.5%, so its resident slope
    is a real per-token cost and the prediction can be checked against it. Over
    is safe and under is a window the card cannot hold."""
    predicted = catalog.kv_cost(_dense(), "vulkan")
    assert predicted >= 37.73, "must not come out under what was measured"
    assert predicted <= 37.73 * 1.05, "and must not be so cautious it is useless"


def test_cuda_costs_more_per_token_than_vulkan():
    """~10% more for the same cache, which compounds with depth rather than
    being a flat tax. It is the larger of CUDA's two costs at a full window."""
    dense = _dense()
    assert catalog.kv_cost(dense, "cuda") > catalog.kv_cost(dense, "vulkan")


def test_a_head_that_drafts_is_a_layer_that_costs():
    """MTP's cache was the whole of the gap between this model's nominal and
    resident cost. A model priced as though the head were free promises a
    window that the head has already spent."""
    dense = _dense()
    without = catalog.Model(**{**vars(dense), "mtp": False})
    assert catalog.kv_cost(dense) > catalog.kv_cost(without)


@pytest.mark.parametrize("backend", ["vulkan", "cuda"])
def test_the_plan_stays_under_the_ceiling_that_was_measured(backend):
    """The failure this prevents is a window that loads the weights and then
    dies on the KV allocation, which reads to a user as the model crashing
    rather than as the plan having been too big."""
    got = catalog.fit(_dense(), True, REFERENCE, backend).pool_q8
    assert got <= CEILINGS[backend], (
        f"{backend}: planned {got} against a measured ceiling of "
        f"{CEILINGS[backend]}"
    )


def test_choosing_cuda_costs_context_and_the_planner_knows_it():
    """Before this, `fit` gave both backends the same 172032 -- which is
    exactly CUDA's ceiling, so the entire margin the planner believed it had
    was gone. The number is quoted to users by `setup`, so it has to be real."""
    dense = _dense()
    vulkan = catalog.fit(dense, True, REFERENCE, "vulkan").pool_q8
    cuda = catalog.fit(dense, True, REFERENCE, "cuda").pool_q8
    assert cuda < vulkan
    cost = 100 * (vulkan - cuda) / vulkan
    assert 10 <= cost <= 20, f"CUDA costing {cost:.0f}% of the window"


def test_a_model_without_an_mtp_head_is_priced_exactly_as_before():
    """The split reorganises a measurement; it must not change one.

    1.12 was measured end to end on Gemma-4-26B-A4B *through* the q8
    arithmetic error, so the error was already absorbed into it. Correcting
    Q8_0_RATIO while keeping 1.12 would double-count -- pricing Gemma 6% above
    what it was actually measured at. Restated properly, everything with no
    draft cache lands on the same number to the last decimal.
    """
    for m in catalog.load_catalog():
        if m.mtp:
            continue
        before = m.kv_kib_per_token / 2 * 1.12       # the single old constant
        assert catalog.kv_cost(m, "vulkan") == pytest.approx(before), m.name


def test_no_model_is_priced_lower_than_it_used_to_be():
    """The correction could have combined into a *larger* window somewhere,
    which would be shipping a new over-promise while fixing an old one."""
    for m in catalog.load_catalog():
        before = m.kv_kib_per_token / 2 * 1.12
        after = catalog.kv_cost(m, "vulkan")
        assert after >= before or after == pytest.approx(before), (
            f"{m.name} got cheaper: {before} -> {after}"
        )


def test_the_only_entries_that_move_are_the_ones_with_a_draft_cache():
    """Which is the whole claim of this change: the models that were never
    paying for MTP's second cache now do, and nothing else is touched."""
    moved = {
        m.name for m in catalog.load_catalog()
        if catalog.kv_cost(m, "vulkan") != pytest.approx(
            m.kv_kib_per_token / 2 * 1.12
        )
    }
    assert moved == {"Qwen3.8-27B", "Qwen3.6-35B-A3B-MTP"}


def test_the_declared_mtp_fields_match_the_real_checkpoints():
    """The only test here that reads real weights.

    `mtp` and `full_attention_layers` are the two things in models.yaml that
    are declared rather than derived, and both are checkable against the file
    they describe. `kv_heads x (key_length + value_length) x 2 x layers` also
    has to reproduce the hand-entered nominal, which is what makes the whole
    derivation trustworthy rather than merely tidy. Skipped in CI.
    """
    from lllm3090 import gguf

    checked = 0
    for entry in catalog.installed():
        known = next(
            (m for m in catalog.load_catalog() if m.name == entry["name"]), None
        )
        if known is None:
            continue
        assert gguf.has_mtp(entry["path"]) == known.mtp, (
            f"{known.name}: models.yaml says mtp={known.mtp}, the file disagrees"
        )
        if known.mtp:
            assert (
                gguf.full_attention_layers(entry["path"])
                == known.full_attention_layers
            ), f"{known.name}: declared layer count is not the file's"
        checked += 1
    if not checked:
        pytest.skip("no catalogued checkpoints on this machine")


def _card(measured: bool) -> hardware.Profile:
    return hardware.Profile(
        id="t", name="t", compute_capability="8.6", vram_mib=24576,
        bandwidth_gbs=0, measured=measured,
    )


def test_the_right_card_on_the_right_backend_is_just_measured():
    applies, note = catalog.speed_qualifier(_card(True), "vulkan")
    assert applies and note == "measured"


def test_the_right_card_on_another_backend_does_not_describe_this_machine():
    """The same dense 27B serves 54.8 tok/s under Vulkan and 84.9 under CUDA.
    A figure labelled only "measured" on a CUDA engine understates by a third
    while reading as a fact about the machine in front of you."""
    applies, note = catalog.speed_qualifier(_card(True), "cuda")
    assert not applies
    assert note == "measured on vulkan"


def test_another_card_says_so_as_it_always_did():
    applies, note = catalog.speed_qualifier(_card(False), "vulkan")
    assert not applies and note == "other card"


def test_both_wrong_says_both():
    """Collapsing the two into one boolean is what made "other card" a lie on
    a CUDA engine with the right card in it."""
    applies, note = catalog.speed_qualifier(_card(False), "cuda")
    assert not applies
    assert "other card" in note and "vulkan" in note


def test_an_engine_that_could_not_be_asked_is_no_objection():
    """`lllm3090 models` runs before `install-engine` on a fresh machine, and
    the planner asks the engine which backend it is. Qualifying every row on
    the strength of a backend that is about to be Vulkan is pure noise -- the
    same rule capability_ok already follows for an unknown capability."""
    applies, note = catalog.speed_qualifier(_card(True), "cpu")
    assert applies and note == "measured"


def test_every_row_carries_the_backend_and_what_to_say_about_it(monkeypatch):
    """Three front ends render this column and they must not each invent the
    wording, which is how a promise gets made in one place and withdrawn in
    another."""
    monkeypatch.setattr(catalog.engines, "backend", lambda directory=None: "cuda")
    rows = catalog.catalog_for_panel(desktop=True)
    assert rows, "the catalogue is not empty"
    for row in rows:
        assert row["backend"] == "cuda"
        assert row["speed_note"]
        assert not row["speed_applies"], "these speeds were taken on vulkan"


def test_the_dense_27b_is_no_longer_quoted_at_its_pre_mtp_speed():
    """It carries a multi-token prediction head, the engine turns that on by
    itself, and 35 tok/s was what it did before that. A catalogue figure below
    what the shipped configuration delivers is not conservative, it is wrong
    about which configuration it is describing."""
    dense = next(m for m in catalog.load_catalog() if m.name == "Qwen3.8-27B")
    assert dense.expected_tok_s == 55
