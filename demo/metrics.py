"""One datasheet function applied identically to every log (naive, threaded, ground truth)."""
import numpy as np

NAN = float("nan")


def holds(ramps, t_end):
    """Dwell windows: from the end of each ramp to the start of the next (last one to t_end)."""
    ends = [r[0] for r in ramps[1:]] + [t_end]
    return [(r[1], e) for r, e in zip(ramps, ends)]


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
    return float(100 * gap.max() / np.ptp(sensor))


def frames_lost_pct(seq, n_frames=None):
    """% of frames never in `seq` (frame indices >= 1). n_frames: frames produced (simulator truth);
    default: the span first..last seen (all a log can know). NaN without a seq column."""
    if seq is None:
        return NAN
    k = np.unique(np.asarray(seq, int))
    k = k[(k >= 1) & (k <= (n_frames or k[-1]))]
    return float(100 * (1 - len(k) / (n_frames or k[-1] - k[0] + 1)))


def datasheet(t, force, sensor, ramps, t_end):
    """ramps = [(t_start, t_end, direction)] from the stage; t_end = end of protocol."""
    t, force, sensor = (np.asarray(a, float) for a in (t, force, sensor))
    dt_ms = np.diff(t) * 1e3
    fs = force.max()
    hold_masks = [(t > a) & (t <= b) for a, b in holds(ramps, t_end)]
    # unmeasured (< 2 samples) in any hold -> NaN: relaxation cannot be read off one point
    relax = [abs(force[m][0] - force[m][-1]) / fs if m.sum() >= 2 else NAN for m in hold_masks]
    band = (force >= 0.1 * fs) & (force <= 0.9 * fs)
    dwell_min = int(min(m.sum() for m in hold_masks))
    return {
        "samples": len(t),
        "rate_Hz": (len(t) - 1) / (t[-1] - t[0]),
        "dt_p50_ms": float(np.percentile(dt_ms, 50)),
        "dt_p99_ms": float(np.percentile(dt_ms, 99)),
        "dt_max_ms": float(dt_ms.max()),
        "ramp_samples": int(_in_ramps(t, ramps).sum()),
        "dwell_samples_min": dwell_min,
        "force_fs_N": float(fs),
        "relaxation_pct_fs": float(100 * np.mean(relax)),
        "sensitivity_V_per_N": float(np.polyfit(force[band], sensor[band], 1)[0]) if band.sum() >= 2 else NAN,
        "hysteresis_pct": _hysteresis_pct(force, sensor, _direction(t, ramps), band, dwell_min),
    }


def lag_ms(t, force, sensor, truth, ramps, max_ms=60.0):
    """Sensor-minus-force timestamp error (ms; positive = sensor older than force): each channel's delay vs the noiseless truth
    (columns t, force, sensor), least squares over ramp samples; NaN without >= 10 ramp samples."""
    t = np.asarray(t, float)
    m = _in_ramps(t, ramps)
    if m.sum() < 10:
        return NAN
    delays = np.arange(-max_ms, max_ms + 0.25, 0.5) * 1e-3

    def delay(y, col):
        sse = [np.sum((np.asarray(y)[m] - np.interp(t[m] - d, truth[:, 0], truth[:, col])) ** 2) for d in delays]
        return delays[int(np.argmin(sse))]

    return float(1e3 * (delay(sensor, 2) - delay(force, 1)))


def fmt(x):
    return f"{x:.4g}" if np.isfinite(x) else "n/a"


def _rel_err(a, b):
    return abs(a - b) / abs(b) if np.isfinite(a) else np.inf


def failures(ds, truth, period_ms):
    """What is wrong with a log: the name of each check it fails against the noiseless truth."""
    checks = {
        "no ramps": ds["ramp_samples"] > 0,
        "<10 dwell samples": ds["dwell_samples_min"] >= 10,
        "jitter p99>2T": ds["dt_p99_ms"] <= 2 * period_ms + 1e-6,  # tolerance: 40 ms gaps differ from 40.0 by float rounding
        "frames lost>1%": ds["frames_lost_true_pct"] <= 1.0,  # true loss: the simulator knows every frame
        "lag>5ms": abs(ds["sensor_force_lag_ms"]) <= 5.0,
        "full-scale>5%": _rel_err(ds["force_fs_N"], truth["force_fs_N"]) <= 0.05,
        "relaxation>20%": _rel_err(ds["relaxation_pct_fs"], truth["relaxation_pct_fs"]) <= 0.2,
        "sensitivity>5%": _rel_err(ds["sensitivity_V_per_N"], truth["sensitivity_V_per_N"]) <= 0.05,
        # hysteresis_pct is not checked: noise-level (0.5-1.3 % across seeds, full length), see README
    }
    return [name for name, ok in checks.items() if not ok]
