"""One datasheet function applied identically to every log (naive, threaded, ground truth)."""
import numpy as np

NAN = float("nan")
GAPS_FOR_1_IN_100 = 100  # "1 gap in 100" needs this many gaps between rows (the handout says "at least 100 gaps")


def one_gap_in_100(gaps):
    """"1 gap in 100 is longer than ...": the one definition of the app, the notebook, the freeze table and the handout.

    The handout's recipe: sort the gaps between rows from the longest to the shortest and read the value 1 in 100 from the
    top. With n gaps that is the k-th longest gap, k = ceil(n / 100): the 10th of 1000 gaps, the 2nd of 150, the longest of 100.
    No interpolation: the value is one of the gaps. `gaps` in any unit; the result is in the same unit. NaN with no gap.
    Below GAPS_FOR_1_IN_100 gaps it is the longest gap, and it is no "1 gap in 100": whoever shows it checks that count first."""
    gaps = np.asarray(gaps, float)
    if gaps.size == 0:
        return NAN
    k = -(-gaps.size // 100)  # ceil(n / 100) in whole numbers
    return float(np.sort(gaps)[::-1][k - 1])


def holds(ramps, t_end):
    """Hold windows: from the end of each loading ramp to the start of the next ramp (the last one to t_end).

    The framework holds at the depth only (core/coordinator.py: load, hold, unload): after an unloading ramp the next
    loading ramp follows at once, so a window there would have no rows."""
    ends = [r[0] for r in ramps[1:]] + [t_end]
    return [(r[1], e) for r, e in zip(ramps, ends) if r[2] > 0]


def _in_ramps(t, ramps):
    return np.any([(t >= a) & (t <= b) for a, b, _ in ramps], axis=0)


def _direction(t, ramps):
    """+1 loading / -1 unloading: direction of the latest ramp started at or before t (0 before any)."""
    starts = np.array([r[0] for r in ramps])
    dirs = np.array([r[2] for r in ramps])
    idx = np.searchsorted(starts, t, side="right") - 1
    return np.where(idx >= 0, dirs[np.clip(idx, 0, None)], 0)


def _hysteresis_pct(force, sensor, direction, band, n_dwell):
    """Max gap between quadratic fits of loading vs unloading branches, % of sensor span."""
    if n_dwell < 10:  # too few points per step to fit a branch
        return NAN
    fits = []
    for d in (1, -1):
        m = band & (direction == d)
        if m.sum() < 3:
            return NAN
        fits.append((np.polyfit(force[m], sensor[m], 2), force[m].min(), force[m].max()))
    lo, hi = max(f[1] for f in fits), min(f[2] for f in fits)
    if hi <= lo:
        return NAN
    grid = np.linspace(lo, hi, 100)
    gap = np.abs(np.polyval(fits[0][0], grid) - np.polyval(fits[1][0], grid))
    return float(100 * gap.max() / np.ptp(sensor[np.isfinite(sensor)]))


def frames_lost_pct(seq, n_frames=None):
    """% of frames never in `seq` (frame indices >= 1). n_frames: frames produced (simulator truth);
    default: the span first..last seen (all a log can know). NaN without a seq column."""
    if seq is None:
        return NAN
    k = np.unique(np.asarray(seq, int))
    k = k[(k >= 1) & (k <= (n_frames or k[-1]))]
    return float(100 * (1 - len(k) / (n_frames or k[-1] - k[0] + 1)))


def sensor_repeat_pct(sensor):
    """% of rows whose sensor value equals the value of the row before. A sensor that changes at every frame repeats
    a value only when the same frame is logged twice. A row before the first frame (value NaN) is left out."""
    sensor = np.asarray(sensor, float)
    sensor = sensor[np.isfinite(sensor)]
    return float(100 * np.mean(sensor[1:] == sensor[:-1])) if len(sensor) > 1 else NAN


def datasheet(t, force, sensor, ramps, t_end):
    """ramps = [(t_start, t_end, direction)] from the stage; t_end = end of protocol.

    A row before the first frame has the sensor value NaN (the framework writes no zeros there): it counts as a row,
    and the measures that need the sensor value leave it out."""
    t, force, sensor = (np.asarray(a, float) for a in (t, force, sensor))
    dt_ms = np.diff(t) * 1e3
    fs = force.max()
    hold_masks = [(t > a) & (t <= b) for a, b in holds(ramps, t_end)]
    # unmeasured (< 2 samples) in any hold -> NaN: relaxation cannot be read off one point
    relax = [abs(force[m][0] - force[m][-1]) / fs if m.sum() >= 2 else NAN for m in hold_masks]
    band = (force >= 0.1 * fs) & (force <= 0.9 * fs) & np.isfinite(sensor)
    dwell_min = int(min(m.sum() for m in hold_masks))
    return {
        "samples": len(t),
        "rate_Hz": (len(t) - 1) / (t[-1] - t[0]),
        "dt_p50_ms": float(np.percentile(dt_ms, 50)),
        "dt_p99_ms": one_gap_in_100(dt_ms),  # the key keeps its old name: it holds "1 gap in 100" by the recipe above
        "dt_max_ms": float(dt_ms.max()),
        "ramp_samples": int(_in_ramps(t, ramps).sum()),
        "dwell_samples_min": dwell_min,
        "force_fs_N": float(fs),
        "relaxation_pct_fs": float(100 * np.mean(relax)),
        "sensitivity_V_per_N": float(np.polyfit(force[band], sensor[band], 1)[0]) if band.sum() >= 2 else NAN,
        "hysteresis_pct": _hysteresis_pct(force, sensor, _direction(t, ramps), band, dwell_min),
        "sensor_repeat_pct": sensor_repeat_pct(sensor),
    }


def lag_ms(t, force, sensor, truth, ramps, max_ms=60.0, sensor_t=None):
    """Sensor-minus-force timestamp error (ms; positive = sensor older than force): each channel's delay vs the noiseless truth
    (columns t, force, sensor), least squares over ramp samples; NaN without >= 10 ramp samples.

    sensor_t: the time stamp the log gives the sensor value, when it has a column of its own for it (V3, V4); else the
    row's time t. Rows without a sensor value (before the first frame) are left out."""
    t = np.asarray(t, float)
    ts = t if sensor_t is None else np.asarray(sensor_t, float)
    m = _in_ramps(t, ramps)
    ms = m & np.isfinite(sensor) & np.isfinite(ts)
    if ms.sum() < 10:
        return NAN
    delays = np.arange(-max_ms, max_ms + 0.25, 0.5) * 1e-3

    def delay(y, times, mask, col):
        sse = [np.sum((np.asarray(y)[mask] - np.interp(times[mask] - d, truth[:, 0], truth[:, col])) ** 2) for d in delays]
        return delays[int(np.argmin(sse))]

    return float(1e3 * (delay(sensor, ts, ms, 2) - delay(force, t, m, 1)))


def fmt(x):
    return f"{x:.4g}" if np.isfinite(x) else "n/a"


def _rel_err(a, b):
    return abs(a - b) / abs(b) if np.isfinite(a) else np.inf


LAG_LIMIT_MS = 5.0


def failures(ds, truth, period_ms):
    """What is wrong with a log: the name of each check it fails against the noiseless truth."""
    checks = {
        "no ramps": ds["ramp_samples"] > 0,
        "<10 dwell samples": ds["dwell_samples_min"] >= 10,
        "jitter p99>2T": ds["dt_p99_ms"] <= 2 * period_ms + 1e-6,  # tolerance: 40 ms gaps differ from 40.0 by float rounding
        "frames lost>1%": ds["frames_lost_true_pct"] <= 1.0,  # true loss: the simulator knows every frame
        # A fixed 5 ms: no division by the logging period or the frame period. A time stamp taken when the computer reads the
        # frame is late by the sensor's own latency (4 ms) and the reader's wake-up (about 1-3 ms), so V3 and V4 show as late
        # by their measured lag (5.5-7.5 ms); only a sensor that sends its own time passes (owner decision, 2026-10-19: keep 5 ms).
        "lag>5ms": abs(ds["sensor_force_lag_ms"]) <= LAG_LIMIT_MS,
        "full-scale>5%": _rel_err(ds["force_fs_N"], truth["force_fs_N"]) <= 0.05,
        "relaxation>20%": _rel_err(ds["relaxation_pct_fs"], truth["relaxation_pct_fs"]) <= 0.2,
        "sensitivity>5%": _rel_err(ds["sensitivity_V_per_N"], truth["sensitivity_V_per_N"]) <= 0.05,
        # hysteresis_pct is not checked: it follows the time offset between the force and sensor values of a row (critique/hysteresis_cause_2026-10-20.md), see README
    }
    return [name for name, ok in checks.items() if not ok]
