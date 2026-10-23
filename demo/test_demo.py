"""Check-the-checker: each version fixes the previous version's named bug (metric asserted gone) and
exposes its own (metric asserted present); a planted defect is flagged; V3 frame loss depends on start phase.

The runs use the framework's cycle protocol with a shorter ramp than its default (the run controls of settings.py), so that a
test takes seconds. Everything is written to a temporary folder."""
import csv
import functools
import math
import re
import sqlite3
import threading
import time

import numpy as np
import pytest

import freeze_numbers
import loops
import metrics
import naive_vs_threaded as demo
import settings
import timeline

OFF = loops.PHASE_MS + 10.0  # +10 ms = half a frame period: the 50 Hz tick no longer collides with frame arrival
PLANTED = functools.partial(loops.v3_stamped_at_read, log_in_dwell=False)
STEPS = demo.protocol(depth=0.5, hold=0.5, cycles=1)  # 2 legs
SPEED = 0.4  # mm/s: a ramp of 1.25 s instead of the framework's 5 s
V3_SENSOR_TIME = functools.partial(loops.v3_stamped_at_read, sensor_time=True)
RUNS = {  # key -> (loop, kwargs)
    "V0": (loops.v0_naive, {}),
    "V1": (loops.v1_polling, {}),
    "V2": (loops.v2_threaded, {}),
    "planted": (PLANTED, {}),
    **{f"V3_s{s}": (loops.v3_stamped_at_read, {}) for s in (1, 2, 3)},
    "V3_off": (loops.v3_stamped_at_read, {"phase_ms": OFF}),
    "V3_sensor_time": (V3_SENSOR_TIME, {}),
    "V4": (loops.v4_double_tick, {}),
    "V4_sensor_time": (functools.partial(loops.v4_double_tick, sensor_time=True), {}),
}


def rig_in(folder, name, seed=1, speed=SPEED, **kw):
    return demo.build_rig(name, seed, speed, tau_relax=1.5, tau_sensor=0.05, out=folder, **kw)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """key -> (rig, log, datasheet, set of failed checks)."""
    folder, out = tmp_path_factory.mktemp("demo"), {}
    for key, (loop, kw) in RUNS.items():
        seed = int(key[4]) if re.fullmatch(r"V3_s\d", key) else 1
        rig = rig_in(folder, f"test_{key}", seed)
        log = demo.run(loop, rig, STEPS, **kw)
        ds, truth = demo.evaluate(rig, log)
        out[key] = rig, log, ds, set(metrics.failures(ds, truth, period_ms=20.0))
    return out


@pytest.fixture
def sheets(runs):
    return {key: (ds, failed) for key, (_, _, ds, failed) in runs.items()}


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
    ("V3_s1", "sensor_force_lag_ms", lambda x: abs(x) < 10.0),  # V2: stamp-on-read lag (16 ms): the read time is within a frame's latency
    ("V3_off", "frames_lost_true_pct", lambda x: x < 1.0),  # V3 is clean at the other phase (< 1%: edge frames) ...
    ("V4", "frames_lost_true_pct", lambda x: x < 1.0),  # V4 fixes V3 at the pinned phase ...
    ])
def test_previous_bug_gone(sheets, key, metric, ok):
    assert ok(sheets[key][0][metric])


@pytest.mark.parametrize("key, metric, bad", [  # direct metric of the bug each version still has
    ("V1", "dt_p99_ms", lambda x: x > 40.0),
    ("V2", "sensor_force_lag_ms", lambda x: abs(x) > 10.0),
    *[(f"V3_s{s}", "frames_lost_true_pct", lambda x: x > 1.0) for s in (1, 2, 3)],
])
def test_own_bug_present(sheets, key, metric, bad):
    assert bad(sheets[key][0][metric])


def test_the_read_time_leaves_the_latency_and_the_sensors_own_time_removes_it(sheets):
    """The fix of pull request #3 stamps the frame when the computer reads it: 4 ms of sensor latency and the reader's wake-up
    stay (software cannot know the sensor's moment). A sensor that sends its own time has none of that (story beat T15)."""
    read, own = sheets["V3_s1"][0]["sensor_force_lag_ms"], sheets["V3_sensor_time"][0]["sensor_force_lag_ms"]
    assert 2.5 < read < 9.0, read
    assert abs(own) < 2.5, own


def test_v2_loss_true_but_invisible(sheets):  # V2 rows are tick-stamped with no seq: the log cannot show the loss
    assert sheets["V2"][0]["frames_lost_true_pct"] > 5.0
    assert sheets["V2"][0]["frames_lost_visible_pct"] != sheets["V2"][0]["frames_lost_visible_pct"]  # NaN = n/a


@pytest.mark.parametrize("key", ["V3_s1", "V3_s2", "V3_s3"])
def test_v3_visible_loss_matches_true(sheets, key):  # the frame number makes the loss visible
    ds = sheets[key][0]
    assert abs(ds["frames_lost_visible_pct"] - ds["frames_lost_true_pct"]) < 2.0


@pytest.mark.parametrize("key", ["V3_off", "V4"])
def test_a_log_that_takes_the_computers_time_fails_at_most_the_lag_check(sheets, key):
    """V3 at the other phase and V4 lose no frame. Their sensor value is stamped when the computer reads the frame, late by
    its measured lag (5.5-7.5 ms in the freeze): the lag check (5 ms) may flag it, and no other check."""
    assert sheets[key][1] <= {"lag>5ms"}


@pytest.mark.parametrize("key, failed", [
    ("V3_sensor_time", {"frames lost>1%"}),  # at the pinned phase: V3 loses frames, and fails that check only
    ("V4_sensor_time", set()),  # the setting "The sensor sends its own time" is what passes every check
])
def test_a_sensor_that_sends_its_own_time_passes_the_lag_check(sheets, key, failed):
    assert sheets[key][1] == failed


@pytest.mark.parametrize("lag_ms, fails", [(5.0, False), (-5.0, False), (5.5, True), (-5.5, True), (17.0, True)])
def test_the_lag_check_is_5_ms_whatever_the_logging_period(sheets, lag_ms, fails):
    ds, truth = dict(sheets["V4"][0]), dict(sheets["V4"][0])
    ds["sensor_force_lag_ms"] = lag_ms
    for period_ms in (10.0, 20.0, 100.0):
        assert ("lag>5ms" in metrics.failures(ds, truth, period_ms)) == fails


@pytest.mark.parametrize("key", ["V2", "V3_s1", "V3_s2", "V3_s3"])
def test_the_share_of_repeated_sensor_values_is_the_loss(runs, key):
    """The sensor's value changes at every frame, so a value equal to the one before is a frame the loop read twice, and
    one read twice is one that was lost (the loop ticks as often as frames come). Counted twice: by the datasheet, and from
    the CSV file that the loop wrote (second route)."""
    rig, _, ds, _ = runs[key]
    with open(rig.disk.path) as f:
        values = [row["sensor_V"] for row in csv.DictReader(f)]
    from_file = 100 * np.mean([a == b != "nan" for a, b in zip(values[1:], values)])  # a row before the first frame is no repeat
    assert ds["sensor_repeat_pct"] == pytest.approx(from_file, abs=0.5)
    assert abs(ds["sensor_repeat_pct"] - ds["frames_lost_true_pct"]) < 4.0  # measured 0.1-0.7 pp apart over seeds 1-3, 63 s


@pytest.mark.parametrize("key", ["V0", "V1", "V4", "V3_off"])
def test_no_sensor_value_is_logged_twice_where_each_frame_is_read_once(sheets, key):
    assert sheets[key][0]["sensor_repeat_pct"] == pytest.approx(0.0, abs=0.5)  # V3_off: a tick at +14 ms always finds a new frame


def test_the_file_of_v3_has_the_frame_number_and_the_time_of_the_sensor_value_and_v2_has_neither(runs):
    with open(runs["V3_s1"][0].disk.path) as f:
        header = f.readline().strip().split(",")
    assert header == ["t_s", "force_N", "sensor_V", "sensor_seq", "sensor_t_s"]
    with open(runs["V2"][0].disk.path) as f:
        assert f.readline().strip().split(",") == ["t_s", "force_N", "sensor_V"]


def test_v3_writes_no_zeros_before_the_first_frame_and_v2_does(runs):
    """Pull request #3: NaN and frame number 0 before the first frame. Before it, the framework wrote zeros."""
    first_v3, first_v2 = np.array(runs["V3_s1"][1])[0], np.array(runs["V2"][1])[0]
    assert first_v3[3] == 0 and np.isnan(first_v3[2]) and np.isnan(first_v3[4])
    assert first_v2[3] == 0 and first_v2[2] == 0.0


def test_a_row_before_the_first_frame_has_no_effect_on_the_measures_that_need_the_sensor(runs):
    """V3 writes NaN for the sensor before the first frame. The measures of the sensor leave that row out: the same values as
    for the log without it (a NaN in the sensor made the hysteresis n/a, once)."""
    rig, log = runs["V3_s1"][:2]
    rows = np.array(log)
    assert np.isnan(rows[:, 2]).any()
    with_nan = metrics.datasheet(*rows[:, :3].T, rig.world.ramps, rig.t_end)
    without = metrics.datasheet(*rows[~np.isnan(rows[:, 2]), :3].T, rig.world.ramps, rig.t_end)
    for key in ("force_fs_N", "relaxation_pct_fs", "sensitivity_V_per_N", "hysteresis_pct", "sensor_repeat_pct"):
        assert np.isfinite(with_nan[key]) and with_nan[key] == pytest.approx(without[key]), key
    truth = np.array(rig.world.truth)
    assert metrics.lag_ms(*rows[:, :3].T, truth, rig.world.ramps, sensor_t=rows[:, 4]) == pytest.approx(
        metrics.lag_ms(*rows[~np.isnan(rows[:, 2]), :3].T, truth, rig.world.ramps, sensor_t=rows[~np.isnan(rows[:, 2]), 4]))


def test_v4_writes_each_frame_once(runs):
    rig, log, _, _ = runs["V4"]
    seq = np.array(log)[:, 3]
    assert np.all(np.diff(seq) > 0) and np.all(seq >= 1)  # each frame at most once, in order, none before the first frame
    t = np.array(log)[:, 0]
    assert np.median(np.diff(t)) == pytest.approx(rig.sensor.period, abs=0.002)  # a row for each frame, 20 ms apart
    assert len(log) == pytest.approx(len(set(seq)), abs=0)  # no frame twice


def test_a_sensor_that_sends_its_own_time_has_it_in_the_file(runs):
    rig, log, *_ = runs["V3_sensor_time"]
    rows = np.array(log)
    seen = rows[:, 3] > 0
    assert np.allclose(rows[seen, 4], rows[seen, 3] * rig.sensor.period)  # the moment frame k was taken: k * period
    read = np.array(runs["V3_s1"][1])
    read_seen = read[:, 3] > 0
    assert np.all(read[read_seen, 4] > read[read_seen, 3] * runs["V3_s1"][0].sensor.period + 0.003)  # the read time is later


def test_a_row_takes_the_sensor_before_it_takes_its_own_time(tmp_path):
    """Framework _emit: the sensor is snapshotted first, so the time of its value is never later than the row's time (a frame
    read after the row's time would be a value from the future). Made certain, not left to chance: the clock is slowed by 3 ms
    for the coordinator only, and frames arrive 2 ms after its tick, inside that 3 ms. A coordinator that took its time first
    would then see frames that were read after its time."""
    rig = rig_in(tmp_path, "emit_order", speed=2.0, p_stall=0.0)
    coordinator, real_now = [], rig.world.now

    def now_slow_for_the_coordinator():
        t = real_now()
        if coordinator and threading.current_thread() is coordinator[0]:
            time.sleep(0.003)  # the time is taken, then 3 ms pass before the caller goes on
        return t

    rig.world.now = now_slow_for_the_coordinator
    log = np.array(demo.run(loops.v3_stamped_at_read, rig, [(1.0, 0.5), (0.0, 0.0)],
                            phase_ms=2.0, on_tick=lambda: coordinator or coordinator.append(threading.current_thread())))
    seen = log[:, 3] > 0
    assert seen.sum() > 50  # many rows with a frame: the check is not empty
    assert np.all(log[seen, 4] <= log[seen, 0])  # sensor_t_s (when the frame was read) <= t_s (the row's time)


LATE_S = 0.005  # a wake-up this much after its time can cost V4 a frame (see test_v4_loses_no_frame)
V4_TICK_S = 0.01  # loops.v4_double_tick: 100 Hz


def late_wake_ups(rig, recorder, phase_ms):
    """Number of wake-ups of a traced V4 run more than LATE_S after their time: the sensor reader's (frames arrive at
    latency + n * period) and the coordinator's (ticks fall at phase + n * 10 ms)."""
    spans = recorder.spans()
    sensor = [(t1 - rig.sensor.latency) % rig.sensor.period for _, _, t1 in spans["sensor"]]
    coordinator = [(t1 - phase_ms * 1e-3) % V4_TICK_S for kind, _, t1 in spans["coordinator"] if kind == "sleeping"]
    return sum(delay > LATE_S for delay in sensor + coordinator)


@pytest.mark.parametrize("seed", range(1, 7))
@pytest.mark.parametrize("phase_ms", [loops.PHASE_MS, loops.PHASE_MS + 5.0])
def test_v4_loses_no_frame(seed, phase_ms, tmp_path):
    """Frames arriving after the last completed tick are not lost frames. The real bound (measured: 3 runs in 160 lose one
    frame with 8 busy processes on the machine, none lose two): V4 loses a frame only when a wake-up, the sensor reader's or
    the coordinator's, comes more than 5 ms late, and then at most two. At phase 9 ms a frame arrives 5 ms before a tick:
    a reader 6 ms late misses that tick, and the next tick is the only one left before the next frame. A run with every
    wake-up on time loses nothing; V3 loses 35 % of the frames at the same phases."""
    rig = rig_in(tmp_path, f"test_v4_s{seed}", seed, speed=2.0)
    recorder = timeline.Recorder(timeline.THREADS["V4"], capacity=50_000)
    log = demo.run(loops.v4_double_tick, rig, [(1.0, 0.5), (0.0, 0.5)], phase_ms=phase_ms, trace=recorder)
    lost = demo.evaluate(rig, log)[0]["frames_lost_true_pct"]
    assert lost == 0.0 or (late_wake_ups(rig, recorder, phase_ms) > 0 and lost < 3.0), f"{lost:.2f} % lost"


def test_v0_hysteresis_not_measurable(sheets):  # V0 logs one point per step: no branch to fit
    assert sheets["V0"][0]["hysteresis_pct"] != sheets["V0"][0]["hysteresis_pct"]  # NaN = n/a


def test_a_perfect_log_passes_the_checks_almost_always(runs):
    """Check the checks: the noiseless truth of a real run, sampled every 20 ms with the rig's noise, is a log that has no
    fault. Measured on the framework's 63 s defaults: 200 of 200 draws pass every check; at a sample stiffness of 1 N/mm the
    same test fails full scale in 78 % of the draws and the force drop in a hold in 44 %. This run has two holds, not six,
    so the force drop is less certain: at most 5 failures in 100 draws (expected: under 1)."""
    rig = runs["V4"][0]
    truth, ramps, t_end = np.array(rig.world.truth), rig.world.ramps, rig.t_end
    reference = metrics.datasheet(*truth.T, ramps, t_end)
    reference.update(sensor_force_lag_ms=0.0, frames_lost_true_pct=0.0, frames_lost_visible_pct=0.0)
    failed = []
    for seed in range(100):
        rng = np.random.default_rng(seed)
        t = np.arange(0.3, t_end - 0.1, 0.02) + rng.uniform(0, 0.02)
        f = np.interp(t, truth[:, 0], truth[:, 1]) + rng.normal(0, 0.02, len(t))  # noise: LoadCell, SoftSensor
        s = np.interp(t, truth[:, 0], truth[:, 2]) + rng.normal(0, 0.005, len(t))
        ds = metrics.datasheet(t, f, s, ramps, t_end)
        ds.update(frames_lost_true_pct=0.0, sensor_force_lag_ms=metrics.lag_ms(t, f, s, truth, ramps))
        failed += metrics.failures(ds, reference, 20.0)
    assert len(failed) <= 5, sorted(set(failed))


def test_run_writes_truth_and_ramps_beside_log(tmp_path):
    rig = rig_in(tmp_path, "t", speed=2.0)
    log = demo.run(loops.v4_double_tick, rig, [(1.0, 0.5), (0.0, 0.0)])
    truth, ramps = (np.loadtxt(tmp_path / f"t_{n}.csv", delimiter=",", skiprows=1, ndmin=2) for n in ("truth", "ramps"))
    t = np.array(log)[:, 0]
    assert np.allclose(truth, rig.world.truth) and np.allclose(ramps, rig.world.ramps)
    assert truth[0, 0] <= t[0] and truth[-1, 0] >= t[-1]


@pytest.mark.parametrize("csv_on, database_on", [(True, False), (False, True), (True, True), (False, False)])
def test_log_to_chooses_the_files_the_writer_writes(tmp_path, csv_on, database_on):
    """Log to: CSV file and/or database file (framework gui/main_window.py:322-324). Both files hold the same rows."""
    rig = rig_in(tmp_path, "t", csv=csv_on, database=database_on, p_stall=0.0)
    log = demo.run(loops.v3_stamped_at_read, rig, [(0.2, 0.3)] , rate_hz=50.0)
    assert rig.disk.closed
    assert rig.disk.path.exists() == csv_on and rig.disk.path.with_suffix(".db").exists() == database_on
    if csv_on:
        with open(rig.disk.path) as f:
            assert len(f.readlines()) == len(log) + 1
    if database_on:
        db = sqlite3.connect(rig.disk.path.with_suffix(".db"))
        rows = db.execute("SELECT * FROM data").fetchall()
        assert len(rows) == len(log) and [c[1] for c in db.execute("PRAGMA table_info(data)")] == list(demo.Disk.COLUMNS)
        assert rows[0][2] is None and rows[0][3] == 0  # before the first frame: NULL sensor value, frame number 0
        db.close()


def test_with_no_file_chosen_nothing_is_written_and_nothing_stalls(tmp_path):
    rig = rig_in(tmp_path, "t", csv=False, database=False, p_stall=1.0)  # a disk that would stall at every write
    rig.disk.write((0.0, 1.0, 2.0))
    assert rig.disk.cols == 3 and not list(tmp_path.glob("t_log.*"))


def test_slow_disk_stalls_more_often_and_longer():
    normal, slow = settings.rig_options(settings.defaults()), settings.rig_options({**settings.defaults(), "slow_disk": True})
    assert slow["p_stall"] > normal["p_stall"] and slow["stall_s"][0] > normal["stall_s"][0] and slow["stall_s"][1] > normal["stall_s"][1]
    assert (normal["p_stall"], normal["stall_s"]) == (0.05, (0.02, 0.04))  # the disk of the frozen numbers


# --- the framework's cycle protocol (core/coordinator.py:174-208) ------------------------------------------------------


@pytest.mark.parametrize("version", ["V1", "V2", "V3", "V4"])
def test_the_cycle_protocol_has_cycles_plus_one_legs_of_load_hold_and_unload(tmp_path, version):
    """cycles = 2 -> 3 legs; each leg: a load ramp of depth / speed, a hold, an unload ramp of depth / speed, and no pause
    at the surface before the next load ramp. Read from the stage's own record of its moves (second route)."""
    depth, speed, hold, cycles = 0.5, 1.0, 0.4, 2
    steps = demo.protocol(depth, hold, cycles)
    assert steps == [(depth, hold), (0.0, 0.0)] * 3
    rig = rig_in(tmp_path, f"p_{version}", speed=speed, p_stall=0.0)
    demo.run(loops.VERSIONS[version], rig, steps)
    t0, t1, direction = np.array(rig.world.ramps).T
    assert list(direction) == [1, -1] * (cycles + 1)
    assert np.allclose(t1 - t0, depth / speed, atol=0.002)  # every ramp takes depth over speed
    assert np.allclose(t0[1::2] - t1[0::2], hold, atol=0.06)  # from the end of a load ramp to the start of its unload ramp
    assert np.all(t0[2::2] - t1[1:-1:2] < 0.06)  # no hold at the surface: the next load ramp follows at once


def test_v0_makes_one_blocking_move_to_each_target_of_the_protocol_and_sleeps_instead_of_holding(tmp_path, monkeypatch):
    monkeypatch.setattr(loops, "V0_SLEEP_S", 0.05)  # the number of moves does not depend on the sleep, which is 2 s in the real V0
    rig = rig_in(tmp_path, "p_V0", speed=5.0, p_stall=0.0)
    log = demo.run(loops.v0_naive, rig, demo.protocol(0.5, 0.4, 2))
    t0, t1, direction = np.array(rig.world.ramps).T
    assert list(direction) == [1, -1] * 3 and np.allclose(t1 - t0, 0.1, atol=0.002) and len(log) == 6
    pauses = t0[1:] - t1[:-1]
    assert np.all(pauses > 0.05) and np.all(pauses < 0.3)  # the sleep and one read pair, not the 0.4 s hold


def test_the_hold_is_the_framework_hold_for_v1_to_v4_and_two_seconds_for_v0():
    assert loops.V0_SLEEP_S == 2.0  # AIS SI Listing 1


# The frozen numbers (out/freeze/NUMBERS.md) are the ones the slides quote: a run at the defaults must land in them.
def frozen_range(cell):
    """(low, high) of a NUMBERS.md cell: "a" or "a–b" (an en dash; a minus sign is a hyphen)."""
    low, _, high = cell.partition("–")
    return float(low), float(high or low)


# One run is one more draw of a random process (the disk stalls, the wake-ups of the threads) on a computer whose load is not
# the load of the freeze, and three seeds do not show all of its spread. A fresh run may lie outside the frozen range by
# ROOM[metric] = (share of the range's larger end, least room in the unit of the metric).
#
# The frames lost have their own room for each version, in points of loss, one constant (LOSS_ROOM_POINTS):
# - V2 and V3: 1.5 points. Re-runs of seed 1 seen while this test was written (their output was not kept): quiet re-runs lay
#   0.5-0.6 points outside the frozen range, loaded re-runs up to 2.2 points outside. At the pinned start phase a few ms
#   of shift in a wake-up change the loss (the sweep, out/freeze/sweep.txt: V3 22 % at 2 ms, 34 % at 4 ms, 15 % at 6 ms), and the
#   load shifts the wake-ups. 1.5 is a choice made now, before any new run; it is not a measurement.
# - V4: 0.25 points, kept. It was raised from a smaller room after a run lost 0.13 % (4 frames: the sweep at the 2 ms phase,
#   out/freeze/sweep.txt). 0.25 is about twice that; a loop that misses frames as V3 does loses 35 %.
LOSS_ROOM_POINTS = {"V2": 1.5, "V3": 1.5, "V4": 0.25}
ROOM = {
    "sensor_repeat_pct": (0.10, 0.04),  # follows the loss within 0.05 points (the three runs of the freeze)
    # The load of the computer moves the wake-ups: fresh lags were 1-2 ms below the freeze (V4: 5-6 against 7-7.5 ms), and
    # the lag of 20 s runs spreads over 4.5-10 ms.
    "sensor_force_lag_ms": (0.0, 2.5),
    "force_fs_N": (0.0, 0.05),  # the largest of 3000 readings with a noise of 0.02 N: fresh runs were 0.02 N outside
    "relaxation_pct_fs": (0.0, 1.0),  # fresh runs were 0.5 points outside (the noise of the first and last reading of a hold)
}
LOSS_METRIC = freeze_numbers.LOSS_METRIC


def room_for(metric, version, low, high):
    """How far a fresh run of `version` may lie outside the frozen range [low, high]."""
    if metric == LOSS_METRIC:
        return LOSS_ROOM_POINTS[version]
    share, least = ROOM[metric]
    return max(share * max(abs(low), abs(high)), least)


@pytest.mark.parametrize("version", ["V2", "V3", "V4"])
def test_a_run_at_the_defaults_gives_the_frozen_numbers(tmp_path, version):
    """Seed 1 is one of the seeds of the freeze: a run at the defaults lands in the frozen range, widened by ROOM.
    Out of it: the freeze is old, or the code changed."""
    frozen = freeze_numbers.read_md_table(freeze_numbers.FREEZE / "NUMBERS.md")
    values = settings.defaults()
    rig = demo.build_rig("frozen", 1, values["speed"], tau_relax=1.5, tau_sensor=0.05, out=tmp_path, **settings.rig_options(values))
    kw = {"rate_hz": values["rate"], "phase_ms": loops.PHASE_MS}
    ds, _ = demo.evaluate(rig, demo.run(loops.VERSIONS[version], rig, demo.protocol(values["depth"], values["hold"], values["cycles"]), **kw))
    for metric in (LOSS_METRIC, *ROOM):
        low, high = frozen_range(frozen[(metric, version)])
        room = room_for(metric, version, low, high)
        assert low - room <= ds[metric] <= high + room, (metric, ds[metric], frozen[(metric, version)])


@pytest.mark.parametrize("n, k", [(100, 1), (101, 2), (150, 2), (200, 2), (201, 3), (1000, 10), (1001, 11), (2899, 29)])
def test_one_gap_in_100_is_the_kth_longest_gap_with_k_the_number_of_gaps_over_100_rounded_up(n, k):
    """The handout: sort the gaps from the longest, read the value 1 in 100 from the top ("with 1000 gaps, the 10th longest").
    Gap number i is i ms, so the k-th longest of n gaps is n - k + 1 ms, whatever the order they come in."""
    gaps = np.random.default_rng(n).permutation(np.arange(1, n + 1)).astype(float)
    assert metrics.one_gap_in_100(gaps) == n - k + 1
    assert metrics.one_gap_in_100(gaps * 1e-3) == pytest.approx((n - k + 1) * 1e-3)  # the unit is the caller's


def test_one_gap_in_100_below_a_hundred_gaps_is_the_longest_and_with_no_gap_it_is_nan():
    assert metrics.one_gap_in_100([5.0, 9.0, 7.0]) == 9.0
    assert math.isnan(metrics.one_gap_in_100([]))
    assert metrics.GAPS_FOR_1_IN_100 == 100


def test_the_datasheet_reads_1_gap_in_100_by_the_same_function():
    t = np.concatenate([[0.0], np.cumsum([0.02] * 148 + [0.06, 0.08])])  # 150 gaps: the 2nd longest is 60 ms
    ds = metrics.datasheet(t, np.ones_like(t), np.ones_like(t), [(0.0, 1.0, 1)], t[-1])
    assert ds["dt_p99_ms"] == pytest.approx(60.0) and ds["dt_max_ms"] == pytest.approx(80.0)
