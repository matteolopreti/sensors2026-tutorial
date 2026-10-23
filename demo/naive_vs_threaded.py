"""Same simulated soft sensor, one acquisition loop in five versions (loops.py), five datasheets.

V0 naive (AIS SI Listing 1) -> V1 single-thread polling -> V2 threaded, stamped on read
-> V3 threaded, the fix of the framework's pull request #3 (the time the frame was read, and its number)
-> V4 coordinator at 2x the logging rate. The "bug exposed" row lists the checks each version still fails.
The protocol is the framework's: cycles + 1 legs of load ramp, hold, unload ramp (defaults of settings.py).
Run:  uv run --with numpy --with matplotlib python naive_vs_threaded.py --fast --seed 1
"""
import argparse
import bisect
import math
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import loops
import metrics
import plots
import settings
import timeline

OUT = Path(__file__).resolve().parent / "out"


def sensor_curve(f):
    """Static soft-sensor transfer (V vs N), mildly saturating."""
    return 0.5 * f - 0.02 * f * f


# Sample stiffness (N/mm, both springs). The framework's depth is 0.5 mm (settings.py): a sample of 1 N/mm pushes back with
# 0.65 N there, and the load cell's noise (0.02 N) hides the full-scale force and the force drop in a hold (a perfect log
# of that rig fails 3 checks in 4). At 7 N/mm the largest force is 4.5 N, as in the staircase freeze it replaces.
STIFFNESS_N_PER_MM = 7.0


class World:
    """Stage + standard-linear-solid sample + lagged sensor, integrated on demand at 1 kHz (= truth)."""

    def __init__(self, k_inf=STIFFNESS_N_PER_MM, k1=STIFFNESS_N_PER_MM, tau_relax=1.5, tau_sensor=0.05, dt=1e-3):
        self.k_inf, self.k1, self.tau_r, self.tau_s, self.dt = k_inf, k1, tau_relax, tau_sensor, dt
        self.y = self.f_lag = self.t = 0.0
        self.move = (0.0, 0.0, 0.0, 1.0)  # t0, x0, target, velocity
        self.ramps, self.truth = [], [(0.0, 0.0, 0.0)]
        self.legs = [self.move]  # every move, to replay the stage depth at a past time
        self.lock = threading.Lock()
        self.t0 = time.perf_counter()

    def now(self):
        return time.perf_counter() - self.t0

    def _x(self, t, move=None):
        t0, x0, xt, v = move or self.move
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

    def depth_at(self, t):
        """Stage depth (mm) at a past time t, replayed from the recorded legs."""
        return self._x(t, self.legs[bisect.bisect_right([leg[0] for leg in self.legs], t) - 1])

    def start_move(self, target, v):
        with self.lock:
            self._advance()
            t0 = self.now()  # same clock as log timestamps
            x0 = self._x(t0)
            t1 = t0 + abs(target - x0) / v
            self.move = (t0, x0, target, v)
            self.legs.append(self.move)
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

    def halt(self):
        """Stop where the stage is now."""
        self.world.start_move(self.position(), self.v)

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
    """The file writer's destination: a CSV file and/or a database file (SQLite), with occasional OS stalls of tens of ms.

    With neither chosen nothing is written, and nothing stalls."""
    COLUMNS = ("t_s", "force_N", "sensor_V", "sensor_seq", "sensor_t_s")  # a row has 3 of them (V0-V2) or all 5 (V3, V4)
    COMMIT_INTERVAL_S = 1.0  # framework core/data_logger.py: the database commits at most once a second

    def __init__(self, path, rng, p_stall=0.05, stall_s=(0.02, 0.04), csv=True, database=False):
        self.path, self.rng, self.p, self.stall = Path(path), rng, p_stall, stall_s
        self.f = open(self.path, "w") if csv else None  # noqa: SIM115 closed in close()
        self.db = sqlite3.connect(self.path.with_suffix(".db"), check_same_thread=False) if database else None
        if self.db:
            self.db.execute("PRAGMA journal_mode=WAL")  # as the framework does
        self.last_commit = time.perf_counter()
        self.cols = 0  # columns written (3 or 5)
        self.closed = False

    @property
    def stem(self):
        return self.path.with_name(self.path.name.removesuffix("_log.csv"))  # ground truth and ramps go beside the log

    def logs_frame_numbers(self):
        return self.cols > 3

    def write(self, row):
        if not self.cols:
            self.cols = len(row)
            names = self.COLUMNS[:3] + self.COLUMNS[3:self.cols]
            if self.f:
                self.f.write(",".join(names) + "\n")
            if self.db:
                self.db.execute(f"CREATE TABLE data ({', '.join(f'{n} REAL' for n in names)})")
        if self.f:
            self.f.write(",".join(f"{v:.6f}" for v in row) + "\n")
            self.f.flush()
        if self.db:
            self.db.execute(f"INSERT INTO data VALUES ({', '.join('?' * len(row))})", [None if v != v else v for v in row])
            if time.perf_counter() - self.last_commit >= self.COMMIT_INTERVAL_S:
                self.db.commit()
                self.last_commit = time.perf_counter()
        if (self.f or self.db) and self.rng.random() < self.p:
            time.sleep(self.rng.uniform(*self.stall))

    def close(self):
        if self.f:
            self.f.close()
        if self.db:
            self.db.commit()
            self.db.close()
        self.closed = True


class StallWatch:
    """The longest stall of a thread wake-up while it runs: a thread sleeps SLEEP_S again and again and notes how late it wakes.

    A late wake-up here means the computer did not run this process, or one thread held the interpreter, for that long: the
    cause of a long gap in a log that the model does not have. Used by the command line (the freeze), not by the window."""
    SLEEP_S = 0.01

    def __init__(self):
        self.longest_s = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            time.sleep(self.SLEEP_S)
            self.longest_s = max(self.longest_s, time.perf_counter() - t0 - self.SLEEP_S)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def stall_line(seconds, where):
    """The line of the output that freeze_numbers.py reads: the longest wake-up stall of a run, and where it was."""
    return f"longest wake-up stall: {seconds * 1e3:.0f} ms ({where})"


def build_rig(name, seed, v, tau_relax, tau_sensor, p_stall=0.05, out=OUT, sensor_fps=50.0, stall_s=(0.02, 0.04),
              csv=True, database=False):
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    world = World(tau_relax=tau_relax, tau_sensor=tau_sensor)
    out.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(world=world, stage=Stage(world, v), load_cell=LoadCell(world, rngs[0]),
                           sensor=SoftSensor(world, rngs[1], period=1.0 / sensor_fps),
                           disk=Disk(out / f"{name}_log.csv", rngs[2], p_stall, stall_s, csv, database))


def protocol(depth, hold, cycles):
    """The framework's cycle protocol (core/coordinator.py:174-208): cycles + 1 legs, each one a load ramp to the depth,
    a hold there, and an unload ramp to the surface. Steps: (target mm, hold s after reaching it)."""
    return [(depth, hold), (0.0, 0.0)] * (cycles + 1)


def run(loop, rig, steps, **kw):
    """Run one loop version to completion; records the protocol end time and closes the log file."""
    log = loop(rig, steps, **kw)
    rig.world.sample()  # integrate truth to the end of the run
    rig.t_end = rig.world.now()  # protocol end (hold windows); frame counting uses rig.t_stop, set by the loop
    rig.disk.close()
    np.savetxt(f"{rig.disk.stem}_truth.csv", rig.world.truth, delimiter=",", header="t,force,sensor", comments="")
    np.savetxt(f"{rig.disk.stem}_ramps.csv", rig.world.ramps, delimiter=",", header="t0,t1,dir", comments="")
    return log


def evaluate(rig, log):
    """(datasheet of the log, datasheet of the same run's noiseless 1 kHz truth)."""
    truth_arr = np.array(rig.world.truth)
    t, f, s, k, *sensor_t = np.array(log).T  # sensor_t: the time stamp of the sensor value, in V3 and V4 only
    ds = metrics.datasheet(t, f, s, rig.world.ramps, rig.t_end)
    n_produced = math.floor((rig.t_stop - rig.sensor.latency) / rig.sensor.period)
    ds["frames_lost_true_pct"] = metrics.frames_lost_pct(k, n_produced)  # k: simulator's frame index
    ds["frames_lost_visible_pct"] = metrics.frames_lost_pct(k if rig.disk.logs_frame_numbers() else None)  # only if logged
    ds["sensor_force_lag_ms"] = metrics.lag_ms(t, f, s, truth_arr, rig.world.ramps, sensor_t=sensor_t[0] if sensor_t else None)
    truth = metrics.datasheet(*truth_arr.T, rig.world.ramps, rig.t_end)
    truth.update(sensor_force_lag_ms=0.0, frames_lost_true_pct=0.0, frames_lost_visible_pct=0.0, sensor_repeat_pct=0.0)
    return ds, truth


def loop_kwargs(version, a):
    """The keyword arguments of the loop `version` for the command-line settings `a`."""
    kw = {} if version == "V0" else {"rate_hz": a.rate}  # V4 ticks at twice the logging rate by itself
    if version in ("V2", "V3", "V4"):
        kw["phase_ms"] = a.phase_ms
    if version in ("V3", "V4"):
        kw["sensor_time"] = a.sensor_time
    return kw


def rig_for(name, a, out):
    options = settings.rig_options({"fps": a.fps, "csv": True, "database": False, "slow_disk": a.slow_disk})
    options["p_stall"] = 0.0 if a.no_stalls else options["p_stall"]
    return build_rig(name, a.seed, a.speed, a.tau_relax, a.tau_sensor, out=out, **options)


def sweep(a):
    """Chosen threaded versions at several start phases: the same code, a different datasheet."""
    steps = protocol(a.depth, a.hold, a.cycles)
    print(f"phase sweep, seed {a.seed}: coordinator tick offset vs sensor frame clock (frame arrives at +4 ms)")
    print(f"{'version':>8s}{'phase_ms':>9s}{'lost_true_%':>13s}{'lost_visible_%':>16s}{'lag_ms':>9s}")
    longest = (0.0, "")
    for v in a.sweep_versions.split(","):
        for ph in range(0, 20, 2):
            a.phase_ms = ph
            rig = rig_for("sweep", a, a.out)
            with StallWatch() as watch:
                log = run(loops.VERSIONS[v], rig, steps, **loop_kwargs(v, a))
            ds, _ = evaluate(rig, log)
            longest = max(longest, (watch.longest_s, f"{v}, phase {ph} ms"))
            print(f"{v:>8s}{ph:9d}" + "".join(f"{metrics.fmt(ds[k]):>{w}s}" for k, w in (
                ("frames_lost_true_pct", 13), ("frames_lost_visible_pct", 16), ("sensor_force_lag_ms", 9))))
    print(stall_line(*longest))


def export_timelines(a):
    """--timeline OUTDIR: one slide-sized PNG per version, the window the app shows at the end of a run (timeline.py).

    Only the PNGs go to OUTDIR; the log files of the runs go to a temporary folder."""
    a.timeline.mkdir(parents=True, exist_ok=True)
    steps = protocol(a.depth, a.hold, a.cycles)
    with tempfile.TemporaryDirectory(prefix="demo_timeline_") as logs:
        for v in a.versions.split(","):
            rig = rig_for(v, a, Path(logs))
            recorder = timeline.Recorder(timeline.THREADS[v])
            log = run(loops.VERSIONS[v], rig, steps, trace=recorder, **loop_kwargs(v, a))
            window = timeline.window_ending_at(v, rig.t_end)
            fig = timeline.timeline_figure(v, recorder.spans(since=window[0]), window, np.array(log)[:, 0])
            fig.savefig(a.timeline / f"timeline_{v}.png", dpi=144)
            print(f"{v} -> {a.timeline / f'timeline_{v}.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="the app's Short run: 1 cycle, hold 0.3 s (about 20 s per version)")
    ap.add_argument("--versions", default=",".join(loops.VERSIONS), help="comma list, e.g. V0,V3")
    ap.add_argument("--seed", type=int, default=0)
    for key, flag, unit in (("depth", "--depth", "mm"), ("speed", "--speed", "mm/s"), ("hold", "--hold", "s")):
        ap.add_argument(flag, type=float, default=settings.BY_KEY[key].default, help=f"framework control, {unit}")
    ap.add_argument("--cycles", type=int, default=settings.BY_KEY["cycles"].default, help="the run has cycles + 1 legs")
    ap.add_argument("--rate", type=float, default=settings.BY_KEY["rate"].default, help="logging rate, Hz (V1 to V4)")
    ap.add_argument("--fps", type=float, default=settings.BY_KEY["fps"].default, help="simulated sensor frames per second")
    ap.add_argument("--sensor-time", action="store_true", help="simulated: the sensor sends its own time (V3, V4)")
    ap.add_argument("--slow-disk", action="store_true", help="simulated: the disk stalls five times as often, twice as long")
    ap.add_argument("--tau-relax", type=float, default=1.5, help="sample relaxation time constant, s")
    ap.add_argument("--tau-sensor", type=float, default=0.05, help="sensor first-order lag, s")
    ap.add_argument("--phase-ms", type=float, default=loops.PHASE_MS, help="V2-V4 coordinator tick offset vs frame clock")
    ap.add_argument("--phase-sweep", action="store_true", help="run --sweep-versions at phases 0..18 ms and print loss and lag")
    ap.add_argument("--sweep-versions", default="V2,V3", help="threaded versions for --phase-sweep")
    ap.add_argument("--out", type=Path, default=OUT, help="output folder for logs and plots (default: demo/out)")
    ap.add_argument("--lanes", action="store_true", help="lanes 4-6 demo runs A vs B, seeds 1-3 -> OUT/lanes (needs ~2 min)")
    ap.add_argument("--timeline", type=Path, metavar="OUTDIR",
                    help="one PNG per version of where the time goes (1920 x 1080), written to OUTDIR; use with --fast")
    ap.add_argument("--no-stalls", action="store_true", help="disk never stalls (isolates the stall-driven V1 jitter)")
    a = ap.parse_args()
    if a.fast:
        a.cycles, a.hold = settings.SHORT_RUN["cycles"], settings.SHORT_RUN["hold"]
    if a.lanes:
        import lanes  # here, not at the top: lanes imports this module

        with StallWatch() as watch:
            lanes.main(a.out)
        print(stall_line(watch.longest_s, "lanes 4-6"))
        return
    if a.timeline:
        return export_timelines(a)
    if a.phase_sweep:
        return sweep(a)
    runs, sheets, stalls = {}, {}, {}
    steps = protocol(a.depth, a.hold, a.cycles)
    for v in a.versions.split(","):
        rig = rig_for(v, a, a.out)
        t0 = time.perf_counter()
        with StallWatch() as watch:
            runs[v] = (rig, run(loops.VERSIONS[v], rig, steps, **loop_kwargs(v, a)))
        sheets[v] = evaluate(*runs[v])
        print(f"{v} {loops.VERSIONS[v].__name__}: {time.perf_counter() - t0:.1f} s wall")
        stalls[v] = watch.longest_s
    truth = sheets[v][1]
    print(f"\ncoordinator phase (V2-V4): {a.phase_ms:g} ms after the frame clock origin")
    print(f"{'metric':22s}{'truth':>10s}" + "".join(f"{c:>10s}" for c in sheets))
    for k in truth:
        print(f"{k:22s}{metrics.fmt(truth[k]):>10s}" + "".join(f"{metrics.fmt(d[k]):>10s}" for d, _ in sheets.values()))
    print("hysteresis_pct: not checked (it follows the force-sensor time offset of each row)")
    print(stall_line(*max((t, v) for v, t in stalls.items())))
    bugs = {v: ", ".join(metrics.failures(ds, tr, 1e3 / a.rate)) or "none" for v, (ds, tr) in sheets.items()}
    print("bug exposed (failed checks vs truth):")
    for v, b in bugs.items():
        print(f"  {v}: {b}")
    plots.save_all(a.out, runs, sheets, truth, bugs)
    print(f"plots -> {a.out}")


if __name__ == "__main__":
    main()
