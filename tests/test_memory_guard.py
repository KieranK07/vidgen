"""The memory guard and the single-active-model policy.

These are the invariants the whole design rests on, so they get direct tests
with the memory probe faked to known values.
"""

from __future__ import annotations

import pytest

from vidgen import memory, model_manager, models
from vidgen.memory import MemorySnapshot


def fake_mem(used_gb: float, total_gb: float = 32.0, swap: float = 0.0):
    return MemorySnapshot(
        total_gb=total_gb,
        available_gb=total_gb - used_gb,
        used_gb=used_gb,
        swap_used_gb=swap,
        source="fake",
    )


@pytest.fixture()
def mgr(env, monkeypatch):
    m = model_manager.get_manager()
    return m


def patch_mem(monkeypatch, used_gb: float, **kw):
    snap = fake_mem(used_gb, **kw)
    monkeypatch.setattr(memory, "snapshot", lambda: snap)
    monkeypatch.setattr(model_manager, "snapshot", lambda: snap)
    return snap


def test_fits_when_headroom_available(mgr, monkeypatch):
    patch_mem(monkeypatch, used_gb=5.0)
    d = mgr.check_headroom(20.5, label="LTX-2.3")
    assert d.ok, d.reason


def test_refuses_when_projected_crosses_ceiling(mgr, monkeypatch):
    # 12GB already in use + 20.5GB model + 1GB margin = 33.5GB > 28GB ceiling.
    patch_mem(monkeypatch, used_gb=12.0)
    d = mgr.check_headroom(20.5, label="LTX-2.3")
    assert not d.ok
    assert "ceiling" in d.reason
    assert "browser tabs" in d.reason  # actionable, not just a number


def test_refuses_model_that_cannot_ever_fit(mgr, monkeypatch):
    patch_mem(monkeypatch, used_gb=1.0)
    d = mgr.check_headroom(30.0, label="LTX-2.3 q8")
    assert not d.ok
    assert "exceeds the 28GB ceiling" in d.reason


def test_refuses_when_probe_unavailable(mgr, monkeypatch):
    snap = MemorySnapshot(0, 0, 0, 0, "unavailable")
    monkeypatch.setattr(model_manager, "snapshot", lambda: snap)
    d = mgr.check_headroom(4.0)
    assert not d.ok
    assert "refusing to load blind" in d.reason


def test_activate_raises_guard_error(mgr, monkeypatch):
    patch_mem(monkeypatch, used_gb=20.0)
    with pytest.raises(model_manager.MemoryGuardError):
        with mgr.activate(
            model_key="ltx2", label="LTX-2.3", quant="q6", preset="quality", estimated_gb=20.5
        ):
            pass
    assert mgr.resident is None


def test_single_active_model_switch_unloads_first(mgr, monkeypatch):
    """An immediate LTX -> FLUX switch must never hold both."""
    patch_mem(monkeypatch, used_gb=4.0)
    seen: list[str | None] = []

    with mgr.activate(
        model_key="ltx2", label="LTX-2.3", quant="q6", preset="quality", estimated_gb=20.5
    ):
        seen.append(mgr.resident.model_key)
    seen.append(mgr.resident.model_key if mgr.resident else None)

    with mgr.activate(
        model_key="flux1-dev", label="FLUX.1 dev", quant="q8", preset="quality", estimated_gb=17.5
    ):
        seen.append(mgr.resident.model_key)
    seen.append(mgr.resident.model_key if mgr.resident else None)

    assert seen == ["ltx2", None, "flux1-dev", None]
    kinds = [e["event"] for e in mgr.events()]
    assert kinds.count("model_loading") == 2
    assert kinds.count("model_unloaded") == 2


def test_resident_never_overlaps_even_on_guard_failure(mgr, monkeypatch):
    patch_mem(monkeypatch, used_gb=4.0)
    with mgr.activate(
        model_key="ltx2", label="LTX-2.3", quant="q6", preset="quality", estimated_gb=20.5
    ):
        assert mgr.resident.model_key == "ltx2"
        # Memory situation deteriorates while LTX is loaded.
        patch_mem(monkeypatch, used_gb=26.0)
    assert mgr.resident is None

    with pytest.raises(model_manager.MemoryGuardError):
        with mgr.activate(
            model_key="flux1-dev", label="FLUX.1 dev", quant="q8", preset="quality", estimated_gb=17.5
        ):
            pass
    assert mgr.resident is None


def test_status_reports_resident_model(mgr, monkeypatch):
    patch_mem(monkeypatch, used_gb=4.0)
    with mgr.activate(
        model_key="flux1-fill-dev", label="Fill", quant="q8", preset="quality", estimated_gb=18.5
    ):
        st = mgr.status()
        assert st["resident_model"]["model_key"] == "flux1-fill-dev"
        assert st["resident_model"]["quant"] == "q8"
    assert mgr.status()["resident_model"] is None


# -- estimator ----------------------------------------------------------------


def test_ltx_q8_super_quality_does_not_fit_but_q6_quality_does(env):
    q8 = models.estimate_peak_gb(
        "ltx2", "q8", "super-quality", job_type="video", width=768, height=512, frames=97
    )
    q6 = models.estimate_peak_gb(
        "ltx2", "q6", "quality", job_type="video", width=768, height=512, frames=97
    )
    assert q8 > 28.0, "22B at q8 should be refused on a 32GB machine"
    assert q6 < 28.0, "q6 quality is the intended default and must fit"


def test_estimate_scales_with_frames(env):
    small = models.estimate_peak_gb("ltx2", "q6", "quality", job_type="video", width=768, height=512, frames=49)
    large = models.estimate_peak_gb("ltx2", "q6", "quality", job_type="video", width=768, height=512, frames=193)
    assert large > small


def test_flux_defaults_are_q8(env):
    assert models.get_model("flux1-dev").default_quant == "q8"
    assert models.get_model("flux1-fill-dev").default_quant == "q8"
    assert models.estimate_peak_gb("flux1-dev", "q8", "quality", job_type="image") < 28.0


def test_wan_default_is_q5_not_q4(env):
    assert models.get_model("wan22-14b").default_quant == "q5"


def test_overrides_file_wins(env, monkeypatch):
    (env.data_dir / "model_footprints.json").write_text('{"ltx2:quality:q6": 9.5}')
    assert models.estimate_peak_gb("ltx2", "q6", "quality", job_type="video", frames=97, width=768, height=512) == 9.5
