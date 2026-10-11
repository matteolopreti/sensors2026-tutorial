"""Same simulated soft sensor, one acquisition loop in five versions (loops.py), five datasheets.

V0 naive (AIS SI Listing 1) -> V1 single-thread polling -> V2 threaded, stamped on read
-> V3 threaded, stamped at frame time -> V4 coordinator at 2x frame rate. The "bug exposed" row lists the checks each version still fails.
Run:  uv run --with numpy --with matplotlib python naive_vs_threaded.py --fast --seed 1
"""
import argparse
import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import loops
import metrics
import plots

OUT = Path(__file__).resolve().parent / "out"


def sensor_curve(f):
    """Static soft-sensor transfer (V vs N), mildly saturating."""
    return 0.5 * f - 0.02 * f * f


class World:
    """Stage + standard-linear-solid sample + lagged sensor, integrated on demand at 1 kHz (= truth)."""

    def __init__(self, k_inf=1.0, k1=1.0, tau_relax=1.5, tau_sensor=0.05, dt=1e-3):
        self.k_inf, self.k1, self.tau_r, self.tau_s, self.dt = k_inf, k1, tau_relax, tau_sensor, dt
        self.y = self.f_lag = self.t = 0.0
        self.move = (0.0, 0.0, 0.0, 1.0)  # t0, x0, target, velocity
        self.ramps, self.truth = [], [(0.0, 0.0, 0.0)]
        self.lock = threading.Lock()
        self.t0 = time.perf_counter()

    def now(self):
        return time.perf_counter() - self.t0

    def _x(self, t):
        t0, x0, xt, v = self.move
        step = v * max(t - t0, 0.0)
        return xt if step >= abs(xt - x0) else x0 + math.copysign(step, xt - x0)

    def _advance(self):
        end = self.now()
        while self.t + self.dt <= end:
            self.t += self.dt
            x = self._x(self.t)
            self.y += (x - self.y) * self.dt / self.tau_r  # dashpot arm relaxes
            f = max(self.k_inf * x + self.k1 * (x - self.y), 0.0)  # probe cannot pull
            self.f_lag += (f - self.f_lag) * self.dt / self.tau_s
            self.truth.append((self.t, f, sensor_curve(self.f_lag)))
        return self.truth[-1]

    def sample(self):
        """(t, force, sensor) noiseless, at the current time."""
        with self.lock:
            return self._advance()

    def value_at(self, t):
        """Noiseless (t, force, sensor) at a past time t (1 ms grid)."""
        with self.lock:
            self._advance()
            return self.truth[min(max(round(t / self.dt), 0), len(self.truth) - 1)]

    def position(self):
        return self._x(self.now())

    def start_move(self, target, v):
        with self.lock:
            self._advance()
            t0 = self.now()  # same clock as log timestamps
            x0 = self._x(t0)
            t1 = t0 + abs(target - x0) / v
            self.move = (t0, x0, target, v)
            self.ramps.append((t0, t1, 1 if target > x0 else -1))
            return t1


class Stage:
    def __init__(self, world, v):
        self.world, self.v = world, v

    def start(self, z):
        """Non-blocking move command."""
        return self.world.start_move(z, self.v)

    def position(self):
        return self.world.position()

    def move_to(self, z):
        """Blocks until the ramp ends, like a real stage driver's move_to."""
        t1 = self.start(z)
        time.sleep(max(t1 - self.world.now(), 0.0))


class LoadCell:
    def __init__(self, world, rng, noise=0.02):
        self.world, self.rng, self.noise = world, rng, noise

    def read(self):
        return self.world.sample()[1] + self.rng.normal(0, self.noise)


class SoftSensor:
    """Samples a frame every `period` s; each frame arrives `latency` s later. Returns (value, frame_time, frame index)."""

    def __init__(self, world, rng, period=0.02, latency=0.004, noise=0.005):
        self.world, self.rng, self.period, self.latency, self.noise = world, rng, period, latency, noise

    def read(self):
        """Blocks until the next frame arrives (serial.read()); returns (value, frame_time, frame index)."""
        now = self.world.now()
        k = math.floor((now - self.latency) / self.period) + 1
        time.sleep(max(k * self.period + self.latency - now, 0.0))
        t_frame = k * self.period
        return self.world.value_at(t_frame)[2] + self.rng.normal(0, self.noise), t_frame, k


class Disk:
    """CSV writer with occasional OS stalls of tens of ms."""

    def __init__(self, path, rng, p_stall=0.05, stall_s=(0.02, 0.04)):
        self.f, self.rng, self.p, self.stall = open(path, "w"), rng, p_stall, stall_s  # noqa: SIM115 closed in run()
        self.cols = 0  # columns logged (3, or 4 with the frame seq)

    def write(self, row):
        if not self.cols:
            self.cols = len(row)
            self.f.write(",".join(("t_s", "force_N", "sensor_V", "seq")[:self.cols]) + "\n")
        self.f.write(",".join(f"{v:.6f}" for v in row) + "\n")
        self.f.flush()
        if self.rng.random() < self.p:
            time.sleep(self.rng.uniform(*self.stall))


def build_rig(name, seed, v, tau_relax, tau_sensor, p_stall=0.05, out=OUT):
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    world = World(tau_relax=tau_relax, tau_sensor=tau_sensor)
    out.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(world=world, stage=Stage(world, v), load_cell=LoadCell(world, rngs[0]),
                           sensor=SoftSensor(world, rngs[1]), disk=Disk(out / f"{name}_log.csv", rngs[2], p_stall))


def run(loop, rig, targets, dwell, **kw):
    """Run one loop version to completion; records the protocol end time and closes the log file."""
    log = loop(rig, targets, dwell, **kw)
    rig.world.sample()  # integrate truth to the end of the run
    rig.t_end = rig.world.now()  # protocol end (dwell windows); frame counting uses rig.t_stop, set by the loop
    rig.disk.f.close()
    stem = rig.disk.f.name.removesuffix("_log.csv")  # ground truth beside the log
    np.savetxt(f"{stem}_truth.csv", rig.world.truth, delimiter=",", header="t,force,sensor", comments="")
    np.savetxt(f"{stem}_ramps.csv", rig.world.ramps, delimiter=",", header="t0,t1,dir", comments="")
    return log


def evaluate(rig, log):
    """(datasheet of the log, datasheet of the same run's noiseless 1 kHz truth)."""
    truth_arr = np.array(rig.world.truth)
    t, f, s, k = np.array(log).T
    ds = metrics.datasheet(t, f, s, rig.world.ramps, rig.t_end)
    n_produced = math.floor((rig.t_stop - rig.sensor.latency) / rig.sensor.period)
    ds["frames_lost_true_pct"] = metrics.frames_lost_pct(k, n_produced)  # k: simulator's frame index
    ds["frames_lost_visible_pct"] = metrics.frames_lost_pct(k if rig.disk.cols == 4 else None)  # only if logged
    ds["sensor_force_lag_ms"] = metrics.lag_ms(t, f, s, truth_arr, rig.world.ramps)
    truth = metrics.datasheet(*truth_arr.T, rig.world.ramps, rig.t_end)
    truth.update(sensor_force_lag_ms=0.0, frames_lost_true_pct=0.0, frames_lost_visible_pct=0.0)
    return ds, truth


def protocol(fast):
    """Indentation staircase 0 -> top -> 0 mm; --fast: 1 mm steps to 3 mm, else 0.5 mm steps to 4 mm."""
    step, top = (1.0, 3.0) if fast else (0.5, 4.0)
    up = list(np.arange(step, top + 1e-9, step))
    return up + up[-2::-1] + [0.0]


def sweep(a):
    """Chosen threaded versions at several start phases: the same code, a different datasheet. Short protocol: 2 steps."""
    targets, dwell = (protocol(False) if not a.fast else [1.0, 0.0]), (a.dwell if not a.fast else 0.5)
    print(f"phase sweep, seed {a.seed}: coordinator tick offset vs sensor frame clock (frame arrives at +4 ms)")
    print(f"{'version':>8s}{'phase_ms':>9s}{'lost_true_%':>13s}{'lost_visible_%':>16s}{'lag_ms':>9s}")
    for v in a.sweep_versions.split(","):
        for ph in range(0, 20, 2):
            rig = build_rig("sweep", a.seed, a.velocity, a.tau_relax, a.tau_sensor, out=a.out)
            ds, _ = evaluate(rig, run(loops.VERSIONS[v], rig, targets, dwell, phase_ms=ph))
            print(f"{v:>8s}{ph:9d}" + "".join(f"{metrics.fmt(ds[k]):>{w}s}" for k, w in (
                ("frames_lost_true_pct", 13), ("frames_lost_visible_pct", 16), ("sensor_force_lag_ms", 9))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="6 steps (~15 s per version) instead of 16 (~40 s)")
    ap.add_argument("--versions", default=",".join(loops.VERSIONS), help="comma list, e.g. V0,V3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--velocity", type=float, default=2.0, help="stage ramp speed, mm/s")
    ap.add_argument("--dwell", type=float, default=2.0, help="hold per step, s (Listing 1: 2.0)")
    ap.add_argument("--tau-relax", type=float, default=1.5, help="sample relaxation time constant, s")
    ap.add_argument("--tau-sensor", type=float, default=0.05, help="sensor first-order lag, s")
    ap.add_argument("--phase-ms", type=float, default=loops.PHASE_MS, help="V2-V4 coordinator tick offset vs frame clock")
    ap.add_argument("--phase-sweep", action="store_true", help="run --sweep-versions at phases 0..18 ms and print loss and lag")
    ap.add_argument("--sweep-versions", default="V2,V3", help="threaded versions for --phase-sweep")
    ap.add_argument("--out", type=Path, default=OUT, help="output folder for logs and plots (default: demo/out)")
    ap.add_argument("--no-stalls", action="store_true", help="disk never stalls (isolates the stall-driven V1 jitter)")
    a = ap.parse_args()
    if a.phase_sweep:
        return sweep(a)
    runs, sheets = {}, {}
    for v in a.versions.split(","):
        kw = {"phase_ms": a.phase_ms} if v in ("V2", "V3", "V4") else {}
        rig = build_rig(v, a.seed, a.velocity, a.tau_relax, a.tau_sensor, 0.0 if a.no_stalls else 0.05, a.out)
        t0 = time.perf_counter()
        runs[v] = (rig, run(loops.VERSIONS[v], rig, protocol(a.fast), a.dwell, **kw))
        sheets[v] = evaluate(*runs[v])
        print(f"{v} {loops.VERSIONS[v].__name__}: {time.perf_counter() - t0:.1f} s wall")
    truth = sheets[v][1]
    print(f"\ncoordinator phase (V2-V4): {a.phase_ms:g} ms after the frame clock origin")
    print(f"{'metric':22s}{'truth':>10s}" + "".join(f"{c:>10s}" for c in sheets))
    for k in truth:
        print(f"{k:22s}{metrics.fmt(truth[k]):>10s}" + "".join(f"{metrics.fmt(d[k]):>10s}" for d, _ in sheets.values()))
    print("hysteresis_pct: noise-level, not checked")
    bugs = {v: ", ".join(metrics.failures(ds, tr, 20.0)) or "none" for v, (ds, tr) in sheets.items()}
    print("bug exposed (failed checks vs truth):")
    for v, b in bugs.items():
        print(f"  {v}: {b}")
    plots.save_all(a.out, runs, sheets, truth, bugs)
    print(f"plots -> {a.out}")


if __name__ == "__main__":
    main()
