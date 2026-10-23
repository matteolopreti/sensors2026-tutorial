"""Lanes 4-6 of Fig. S8: three demo runs, each run A (the obvious code) against run B (the design).

Both runs use the V4 loop (100 Hz tick, one row per new frame with its read time and number, FIFO + file writer) and change one thing:
lane 4 blocking vendor call in the coordinator vs a motion thread per move; lane 5 plot redraw inside the tick vs a
20 Hz GUI timer; lane 6 STOP checked only between moves vs every tick, with halt and forced retract.
Truth (force, stage depth) comes from the simulator. Run:  python naive_vs_threaded.py --lanes
"""
import contextlib
import threading
import time

import numpy as np

import loops
import naive_vs_threaded as demo
import plots

SEEDS = (1, 2, 3)
# lane 5 cost model: fit of the measured matplotlib Agg redraw time, cost = A + B * points (redraw_cost.py, this Mac)
REDRAW_A_S, REDRAW_B_S = 0.0118, 0.96e-6
RESIZE_FACTOR = 3.0  # measured: a redraw after doubling the figure size takes ~3x a plain redraw (2.9-3.1)
N0_POINTS = 50_000  # points already on screen at the start (largest measured size)
RESIZE_AT_S, GUI_HZ = 1.5, 20.0
RETRACT_Z, STOP_AT_S = -5.0, 0.5  # depth axis: +5 mm above the surface is depth -5 mm


def motion_thread(rig, z):
    """A short-lived thread makes the blocking vendor call (framework motion_worker); the caller joins it."""
    th = threading.Thread(target=rig.stage.move_to, args=(z,), daemon=True)
    th.start()
    th.join()


@contextlib.contextmanager
def coordinator_moves(rig, log):
    """Lane 4 run A: the coordinator tick makes the blocking call itself, so no tick runs until the move returns."""
    jobs = []

    def mover(rig, z):
        done = threading.Event()
        jobs.append((z, done))
        done.wait()

    def on_tick():
        while jobs:
            z, done = jobs.pop()
            rig.stage.move_to(z)
            done.set()

    yield {"mover": mover, "on_tick": on_tick}


@contextlib.contextmanager
def motion_threads(rig, log):
    """Lane 4 run B: one motion thread per move."""
    yield {"mover": motion_thread}


class Redraw:
    """Modelled redraw cost for n points; the first call after RESIZE_AT_S is a window resize (a big stall)."""

    def __init__(self, world):
        self.world, self.resized = world, False

    def __call__(self, n_points):
        cost = REDRAW_A_S + REDRAW_B_S * n_points
        if not self.resized and self.world.now() >= RESIZE_AT_S:
            self.resized, cost = True, cost * RESIZE_FACTOR
        time.sleep(cost)


@contextlib.contextmanager
def redraw_in_tick(rig, log):
    """Lane 5 run A: the coordinator redraws the plot every tick."""
    redraw = Redraw(rig.world)
    yield {"on_tick": lambda: redraw(N0_POINTS + len(log))}


@contextlib.contextmanager
def gui_timer(rig, log):
    """Lane 5 run B: a GUI thread at 20 Hz copies the log and redraws; the coordinator never waits for it."""
    redraw, done = Redraw(rig.world), threading.Event()

    def gui():
        lim = loops.RateLimiter(GUI_HZ)
        while not done.is_set():
            redraw(N0_POINTS + len(list(log)))  # list(): the copy the GUI plots
            lim.wait(done)

    th = threading.Thread(target=gui, daemon=True)
    th.start()
    try:
        yield {}
    finally:
        done.set()
        th.join()


@contextlib.contextmanager
def stop_button(rig):
    """STOP is pressed STOP_AT_S into the run; rig.t_press is the press time on the world clock."""
    stop = threading.Event()

    def press():
        rig.t_press = rig.world.now()
        stop.set()

    timer = threading.Timer(STOP_AT_S, press)
    timer.start()
    try:
        yield stop
    finally:
        timer.cancel()


@contextlib.contextmanager
def stop_between_moves(rig, log):
    """Lane 6 run A (public framework today): the flag is read between moves; the move in flight is joined."""
    with stop_button(rig) as stop:
        yield {"mover": lambda rig, z: motion_thread(rig, z) if z == RETRACT_Z or not stop.is_set() else None}


@contextlib.contextmanager
def stop_every_tick(rig, log):
    """Lane 6 run B (paper D.5): the tick reads the flag, halts the motion thread, then forces the retract."""
    halted, retract_end = threading.Event(), [0.0]
    with stop_button(rig) as stop:
        def on_tick():
            if stop.is_set() and not halted.is_set():
                rig.stage.halt()
                retract_end[0] = rig.stage.start(RETRACT_Z)
                halted.set()

        def mover(rig, z):
            if halted.is_set():  # skipped move: just wait for the retract
                return time.sleep(max(retract_end[0] - rig.world.now(), 0.0))
            th = threading.Thread(target=rig.stage.move_to, args=(z,), daemon=True)
            th.start()
            while th.is_alive() and not halted.is_set():
                th.join(0.005)

        yield {"mover": mover, "on_tick": on_tick}


def stop_metrics(rig, log, ds):
    """Time from STOP until the depth peaks, and the depth gained since STOP; from the simulator's stage trace."""
    t = np.arange(rig.t_press, rig.world.t, 1e-3)
    z = np.array([rig.world.depth_at(x) for x in t])
    i = int(np.argmax(z))
    return {"stop_to_halt_ms": 1e3 * (t[i] - t[0]), "extra_depth_mm": z[i] - z[0]}


def redraw_metrics(rig, log, ds):
    return {"dt_p99_ms": ds["dt_p99_ms"], "dt_max_ms": ds["dt_max_ms"],
            "gaps_over_100ms": int((np.diff(np.array(log)[:, 0]) > 0.1).sum())}


# lane -> (title, targets, dwell, {"A": run, "B": run}, metrics)
LANES = {
    4: ("Blocking vendor call", [1.0, 0.0], 0.5, {"A": coordinator_moves, "B": motion_threads},
        lambda rig, log, ds: {"ramp_samples": ds["ramp_samples"]}),
    5: ("Plot redraw", [1.0, 0.0], 1.0, {"A": redraw_in_tick, "B": gui_timer}, redraw_metrics),
    6: ("STOP in a blocking move", [3.0, 0.0, RETRACT_Z], 0.2, {"A": stop_between_moves, "B": stop_every_tick}, stop_metrics),
}


def run_lane(lane, variant, seed, out):
    """One run -> (rig, log, metrics dict). The log is shared with the hooks (the GUI copies it)."""
    _, targets, dwell, variants, measure = LANES[lane]
    rig = demo.build_rig(f"lane{lane}_{variant}_s{seed}", seed, 2.0, 1.5, 0.05, out=out)
    log = []
    with variants[variant](rig, log) as hooks:
        demo.run(loops.v4_double_tick, rig, [(z, dwell) for z in targets], log=log, **hooks)
    return rig, log, measure(rig, log, demo.evaluate(rig, log)[0])


def lane_plot(out, lane, runs):
    """runs: variant -> (rig, log). Lanes 4-5 plot force (true line, logged dots); lane 6 the true stage depth."""
    stem, rb = f"lane{lane}", runs["B"][0].world
    if lane == 6:
        traces = []
        for v, (rig, _) in runs.items():
            t = np.arange(0, rig.world.t, 1e-2)
            traces.append((f"run {v}", t, [rig.world.depth_at(x) for x in t], {"color": plots.COLORS[v]}))
        return plots.save_lane(out, stem, "stage depth (mm)", traces, vline=runs["B"][0].t_press)
    truth = np.array(rb.truth)
    traces = [("true force", truth[:, 0], truth[:, 1], {"color": "0.3", "lw": 1.5})]
    for v, (_, log) in runs.items():
        lg = np.array(log)
        traces.append((f"run {v} ({len(lg)} rows)", lg[:, 0], lg[:, 1], plots._style(v)))
    plots.save_lane(out, stem, "force (N)", traces, shade=[r[:2] for r in rb.ramps])


def markdown(results):
    """results: lane -> variant -> list of metric dicts (one per seed) -> one table per lane, ranges over seeds."""
    lines = [f"# Lanes 4-6 (generated by `naive_vs_threaded.py --lanes`; seeds {SEEDS[0]}-{SEEDS[-1]}; ranges min-max over seeds)\n"]
    for lane, (title, *_rest) in LANES.items():
        res = results[lane]
        lines += [f"## Lane {lane}: {title}\n", "| metric | run A | run B |", "|---|---|---|"]
        for k in res["A"][0]:
            cells = [f"{min(r[k] for r in res[v]):.4g} - {max(r[k] for r in res[v]):.4g}" for v in "AB"]
            lines.append(f"| {k} | {cells[0]} | {cells[1]} |")
        lines.append("")
    return "\n".join(lines)


def main(out):
    """Seeds 1-3, run A and B for every lane -> out/lanes: logs, truth, one plot per lane; the numbers -> out/freeze/LANES.md."""
    frozen, out = out / "freeze", out / "lanes"
    results, first = {}, {}
    for lane in LANES:
        results[lane] = {"A": [], "B": []}
        for seed in SEEDS:
            for v in "AB":
                rig, log, m = run_lane(lane, v, seed, out)
                results[lane][v].append(m)
                if seed == SEEDS[0]:
                    first.setdefault(lane, {})[v] = (rig, log)
        lane_plot(out, lane, first[lane])
    md = markdown(results)
    frozen.mkdir(parents=True, exist_ok=True)
    (frozen / "LANES.md").write_text(md + "\n")
    print(md)
    print(f"plots -> {out}, numbers -> {frozen / 'LANES.md'}")
