"""Check-the-checker: each version fixes the previous version's named bug (metric asserted gone) and
exposes its own (metric asserted present); a planted defect is flagged; V3 frame loss depends on start phase."""
import functools

import pytest

import loops
import metrics
import naive_vs_threaded as demo

OFF = loops.PHASE_MS + 10.0  # +10 ms = half a frame period: the 50 Hz tick no longer collides with frame arrival
PLANTED = functools.partial(loops.v3_frame_stamped, log_in_dwell=False)
RUNS = {  # key -> (loop, kwargs)
    "V0": (loops.v0_naive, {}),
    "V1": (loops.v1_polling, {}),
    "V2": (loops.v2_threaded, {}),
    "planted": (PLANTED, {}),
    **{f"V3_s{s}": (loops.v3_frame_stamped, {}) for s in (1, 2, 3)},
    "V3_off": (loops.v3_frame_stamped, {"phase_ms": OFF}),
    "V4": (loops.v4_double_tick, {}),
}


@pytest.fixture(scope="module")
def sheets():
    out = {}
    for key, (loop, kw) in RUNS.items():
        seed = int(key[4]) if key.startswith("V3_s") else 1
        rig = demo.build_rig(f"test_{key}", seed, v=2.0, tau_relax=1.5, tau_sensor=0.05)
        ds, truth = demo.evaluate(rig, demo.run(loop, rig, demo.protocol(fast=True), 2.0, **kw))
        out[key] = ds, set(metrics.failures(ds, truth, period_ms=20.0))
    return out


@pytest.mark.parametrize("key, must_fail", [
    ("V0", {"no ramps", "<10 dwell samples", "relaxation>20%"}),
    ("V1", {"jitter p99>2T"}),
    ("V2", {"lag>5ms"}),
    ("planted", {"<10 dwell samples", "relaxation>20%"}),
    ("V3_s1", {"frames lost>1%"}),  # V3 loses frames at the pinned phase
])
def test_flagged(sheets, key, must_fail):
    assert must_fail <= sheets[key][1]


@pytest.mark.parametrize("key, metric, ok", [  # direct metric of the bug each version removes
    ("V1", "ramp_samples", lambda x: x > 0),  # V0: no ramp data
    ("V1", "dwell_samples_min", lambda x: x >= 10),  # V0: no dwell data
    ("V2", "dt_p99_ms", lambda x: x <= 40.0 + 1e-6),  # V1: disk-stall jitter (V1 itself: > 40 ms, below)
    ("V3_s1", "sensor_force_lag_ms", lambda x: abs(x) <= 5.0),  # V2: stamp-on-read lag
    ("V3_off", "frames_lost_true_pct", lambda x: x < 1.0),  # V3 is clean at the other phase (< 1%: edge frames) ...
    ("V4", "frames_lost_true_pct", lambda x: x < 1.0),  # V4 fixes V3 at the pinned phase ...
    ])
def test_previous_bug_gone(sheets, key, metric, ok):
    assert ok(sheets[key][0][metric])


@pytest.mark.parametrize("key, metric, bad", [  # direct metric of the bug each version still has
    ("V1", "dt_p99_ms", lambda x: x > 40.0),
    ("V2", "sensor_force_lag_ms", lambda x: abs(x) > 5.0),
    *[(f"V3_s{s}", "frames_lost_true_pct", lambda x: x > 1.0) for s in (1, 2, 3)],
])
def test_own_bug_present(sheets, key, metric, bad):
    assert bad(sheets[key][0][metric])


def test_v2_loss_true_but_invisible(sheets):  # V2 rows are tick-stamped with no seq: the log cannot show the loss
    assert sheets["V2"][0]["frames_lost_true_pct"] > 5.0
    assert sheets["V2"][0]["frames_lost_visible_pct"] != sheets["V2"][0]["frames_lost_visible_pct"]  # NaN = n/a


@pytest.mark.parametrize("key", ["V3_s1", "V3_s2", "V3_s3"])
def test_v3_visible_loss_matches_true(sheets, key):  # frame time + seq make the loss visible
    ds = sheets[key][0]
    assert abs(ds["frames_lost_visible_pct"] - ds["frames_lost_true_pct"]) < 2.0


@pytest.mark.parametrize("key", ["V3_off", "V4"])
def test_clean_runs_fail_no_check(sheets, key):
    assert sheets[key][1] == set()


@pytest.mark.parametrize("seed", range(1, 7))
@pytest.mark.parametrize("phase_ms", [loops.PHASE_MS, loops.PHASE_MS + 5.0])
def test_v4_loses_no_frame(seed, phase_ms):  # frames arriving after the last completed tick are not lost frames
    rig = demo.build_rig(f"test_v4_s{seed}", seed, v=2.0, tau_relax=1.5, tau_sensor=0.05)
    log = demo.run(loops.v4_double_tick, rig, [1.0, 0.0], 0.5, phase_ms=phase_ms)
    assert demo.evaluate(rig, log)[0]["frames_lost_true_pct"] == 0.0


def test_v0_hysteresis_not_measurable(sheets):  # V0 logs one point per step: no branch to fit
    assert sheets["V0"][0]["hysteresis_pct"] != sheets["V0"][0]["hysteresis_pct"]  # NaN = n/a
