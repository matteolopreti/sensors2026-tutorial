"""Single source for every demo number quoted in slides, handout and README.

Reads the frozen full-length runs (out/freeze/run_s*.txt: `naive_vs_threaded.py --seed N`, the framework's defaults,
phase 4 ms) and the phase sweep (out/freeze/sweep.txt) and writes out/freeze/NUMBERS.md: per metric and version, min-max
over seeds, plus per-version loss range over phases. It also writes out/freeze/NUMBERS_CHANGES.md: every metric and
version, the value of the staircase freeze (out/freeze/staircase/) and the new one. Quote numbers only from NUMBERS.md.

It writes the conditions of each run as well (out/freeze/conditions.json, the "Run conditions" table of NUMBERS.md): the
1-minute load average of the computer at the start and the end of the run, and the longest stall of a thread wake-up. A run with a
stall longer than STALL_LIMIT_S is made once more; both are kept, the clean one gives the ranges.

"1 gap in 100" (the row dt_p99_ms) is made by metrics.one_gap_in_100, the recipe of the handout. A run file prints that row by the rule
of the day it was made, so the row is made again here from the logs that sit next to the run file (out/freeze/s1/V1_log.csv, ...),
and the line of lane 5 of LANES.md from the logs of out/lanes. The other rows are the ones the run printed.

Run:  python freeze_numbers.py          reads the runs that exist and writes the two files
      python freeze_numbers.py --lost-frames V4   the frame numbers that the logs of the freeze lack (V3 or V4)
      python freeze_numbers.py --run    first makes the runs: seeds 1-3, the phase sweep and the lanes (about 50 minutes,
                                        and nothing else running on the computer)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import metrics

HERE = Path(__file__).parent
FREEZE = HERE / "out" / "freeze"
STAIRCASE = FREEZE / "staircase"
LANES_LOGS = HERE / "out" / "lanes"
GAP_ROW = "dt_p99_ms"  # the row of the datasheet table that holds "1 gap in 100"
VERSIONS = ["truth", "V0", "V1", "V2", "V3", "V4"]
SEEDS = (1, 2, 3)
SWEEP_SEED = 2
CONDITIONS = FREEZE / "conditions.json"
STALL_LIMIT_S = 0.2  # a thread that wakes up later than this was held by the computer: the run is made once more
LOSS_METRIC = "frames_lost_true_pct"
EVERY_RUN_VERSION = "V4"  # the version whose loss the talk quotes: its sentence counts every full-length run
EVERY_RUN_COLUMN = f"{EVERY_RUN_VERSION} lost true %"
SHOWN = 10  # lost frames printed for each seed by --lost-frames
DEFAULTS = ("the framework's defaults: depth 0.5 mm, speed 0.1 mm/s, hold 0.5 s, 5 cycles (6 legs), logging rate 50 Hz, "
            "sensor 50 frames per second")


def table(path):
    """{metric: [value per column]} from the datasheet table of one run; n/a -> None."""
    rows = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == len(VERSIONS) + 1 and re.match(r"^[a-z][a-zA-Z0-9_]*$", parts[0]):
            try:
                rows[parts[0]] = [None if p == "n/a" else float(p) for p in parts[1:]]
            except ValueError:
                continue
    return rows


def gap_1_in_100_ms(log_csv):
    """"1 gap in 100" of a log file, in ms: metrics.one_gap_in_100 of the gaps between the rows (the first column)."""
    t = np.loadtxt(log_csv, delimiter=",", skiprows=1, usecols=0, ndmin=1)
    return metrics.one_gap_in_100(np.diff(t) * 1e3)


def current_table(path):
    """table(path) with the row GAP_ROW made again from the logs of the run (the folder s1 of run_s1.txt), by the definition of
    metrics.one_gap_in_100. The truth column is the log of the noiseless truth, 1 kHz (V4_truth.csv). A run whose logs are not
    there keeps the row it printed."""
    rows = table(path)
    folder = path.with_name(path.stem.replace("run_", ""))
    logs = [folder / ("V4_truth.csv" if v == "truth" else f"{v}_log.csv") for v in VERSIONS]
    if all(log.exists() for log in logs):
        rows[GAP_ROW] = [gap_1_in_100_ms(log) for log in logs]
    return rows


def refresh_lane_gaps(lanes_md, logs_dir=LANES_LOGS):
    """LANES.md with the line GAP_ROW of lane 5 made again from the logs of the lane runs (lane5_A_s1_log.csv, ...): min - max over
    the seeds, for run A and run B. The runs that lanes.py made are not made again."""
    cells = []
    for variant in "AB":
        values = [gap_1_in_100_ms(p) for p in sorted(logs_dir.glob(f"lane5_{variant}_s[0-9]_log.csv"))]
        cells.append(f"{min(values):.4g} - {max(values):.4g}")
    text = re.sub(rf"^\| {GAP_ROW} \| .* \|$", f"| {GAP_ROW} | {cells[0]} | {cells[1]} |", lanes_md.read_text(), flags=re.M)
    lanes_md.write_text(text)


def checks(path):
    """{version: [names of the failed checks]} from the "bug exposed" lines of one run."""
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^  (V\d): (.*)$", line)
        if m:
            out[m.group(1)] = [] if m.group(2) == "none" else m.group(2).split(", ")
    return out


def rng(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    lo, hi = min(vals), max(vals)
    return f"{lo:.4g}" if lo == hi else f"{lo:.4g}–{hi:.4g}"


def sweep_rows(path):
    """{version: [(phase ms, lost true %, lost visible % or None, lag ms)]} from the output of --phase-sweep."""
    sweep = {}
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) == 5 and p[0].startswith("V") and p[1].replace(".", "").isdigit():
            sweep.setdefault(p[0], []).append((float(p[1]), float(p[2]), None if p[3] == "n/a" else float(p[3]), float(p[4])))
    return sweep


def checks_text(per_seed):
    """One cell: each check with the number of seeds that fail it ("all" when every seed does)."""
    names = sorted({n for seed in per_seed for n in seed})
    cells = [n if all(n in seed for seed in per_seed) else f"{n} ({sum(n in seed for seed in per_seed)} of {len(per_seed)})"
             for n in names]
    return ", ".join(cells) or "none"


def conditions_md(conditions):
    """The "Run conditions" section: one line for each attempt at a run, the ones left out marked."""
    lines = ["", "## Run conditions",
             "The 1-minute load average of the computer (it includes the run itself) at the start and at the end of the run, and "
             "the longest time a thread woke up late during the run (`StallWatch`). A run with a stall longer than "
             f"{STALL_LIMIT_S * 1e3:.0f} ms was made once more; the clean one gives the ranges.",
             "| run | attempt | load average at start | load average at end | longest wake-up stall (ms) | used for the ranges |",
             "|---|---|---|---|---|---|"]
    if not all(c["used"] for c in conditions):
        lines.insert(3, "The logs of a run that was left out are overwritten by its second run; its printed table stays in "
                        "`<run>_left_out.txt`.")
    for c in conditions:
        used = "yes" if c["used"] else f"no: left out, the stall is longer than {STALL_LIMIT_S * 1e3:.0f} ms"
        lines.append(f"| {c['run']} | {c['attempt']} | {c['load_start']:.2f} | {c['load_end']:.2f} | "
                     f"{c['stall_ms']:.0f} ({c['stall_where']}) | {used} |")
    return lines


def numbers_md(runs, sweep, seeds, conditions=()):
    """The text of NUMBERS.md and the cells of its tables: ({(metric, version): cell}, {(version, column): cell}, text)."""
    tables = [current_table(p) for p in runs]
    cells, lines = {}, [
        "# Frozen demo numbers (generated by freeze_numbers.py; do not edit)",
        f"Full length, {DEFAULTS}; phase 4 ms, seeds: {', '.join(map(str, seeds))}. Range = min–max over seeds.", "",
        "| metric | " + " | ".join(VERSIONS) + " |", "|---" * (len(VERSIONS) + 1) + "|"]
    for m in tables[0]:
        row = [rng([t[m][i] for t in tables if m in t]) for i in range(len(VERSIONS))]
        cells.update({(m, v): c for v, c in zip(VERSIONS, row)})
        lines.append(f"| {m} | " + " | ".join(row) + " |")
    lines += ["", "## Frames lost, seed by seed (true %, full-length runs)",
              "| version | " + " | ".join(f"seed {n}" for n in seeds) + " |", "|---" * (len(seeds) + 1) + "|"]
    lines += [f"| {v} | " + " | ".join(f"{t[LOSS_METRIC][i]:.4g}" for t in tables) + " |" for i, v in enumerate(VERSIONS) if v != "truth"]
    every_run = every_run_losses(tables, sweep)
    lines += ["", f"## {EVERY_RUN_VERSION}: frames lost in every full-length run",
              f"{len(every_run)} runs: the {len(tables)} seeds above and the {len(every_run) - len(tables)} start phases of the phase "
              f"sweep below (seed {SWEEP_SEED}; the sweep's runs are full length too, the framework's defaults).",
              f"| run | {EVERY_RUN_COLUMN} |", "|---|---|"]
    lines += [f"| {name} | {x:.4g} |" for name, x in every_run]
    lines += ["", loss_sentence([x for _, x in every_run])]
    failed = [checks(p) for p in runs]
    lines += ["", "## Checks each version fails (against the noiseless truth; a check that only some seeds fail says how many)",
              "| version | failed checks |", "|---|---|"]
    for v in VERSIONS[1:]:
        text = checks_text([f[v] for f in failed])
        cells[("failed checks", v)] = text
        lines.append(f"| {v} | {text} |")
    sweep_cells = {}
    lines += ["", f"## Phase sweep (seed {SWEEP_SEED}, phase 0–18 ms)", "| version | lost true % | lost visible % | lag ms | phases with 0 % |",
              "|---|---|---|---|---|"]
    for v, rows in sweep.items():
        zero = ", ".join(f"{r[0]:g}" for r in rows if r[1] == 0) or "none"
        row = [rng([r[1] for r in rows]), rng([r[2] for r in rows]), rng([r[3] for r in rows]), zero]
        sweep_cells.update({(v, c): x for c, x in zip(("lost true %", "lost visible %", "lag ms", "phases with 0 %"), row)})
        lines.append(f"| {v} | " + " | ".join(row) + " |")
    if conditions:
        lines += conditions_md(conditions)
    return cells, sweep_cells, "\n".join(lines) + "\n"


def read_md_table(path):
    """{(row label, column): cell} from the markdown tables of a NUMBERS.md (the file is made by `python freeze_numbers.py`)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: make it with `python freeze_numbers.py` in the demo folder")
    out, header = {}, None
    for line in path.read_text().splitlines():
        if not line.startswith("|") or set(line) <= set("|-"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] in ("metric", "version", "run"):
            header = cells
        elif header:
            out.update({(cells[0], h): c for h, c in zip(header[1:], cells[1:])})
    return out


def every_run_losses(tables, sweep, version=EVERY_RUN_VERSION):
    """[(run, frames lost true %)] of every full-length run of `version`: one per seed (`tables`, the printed tables of the
    runs, one per seed) and one per start phase of the phase sweep (`sweep`, from sweep_rows)."""
    seeds = [(f"seed {n}", t[LOSS_METRIC][VERSIONS.index(version)]) for n, t in enumerate(tables, 1)]
    return seeds + [(f"sweep, phase {phase:g} ms", lost) for phase, lost, _, _ in sweep.get(version, [])]


def frozen_losses(path):
    """The % of frames lost in each full-length run of the freeze, one number per run, from the table of NUMBERS.md that
    every_run_losses made."""
    return [float(cell) for (_, column), cell in read_md_table(path).items() if column == EVERY_RUN_COLUMN]


def loss_sentence(lost):
    """Plain words for the frames a version lost in the full-length runs of the freeze (the redraw is off in them):
    `lost` is the % lost in each run. Only what the numbers say: in how many runs it lost no frame, and the most it lost in one."""
    clean, runs = sum(x == 0 for x in lost), len(lost)
    if clean == runs:
        return f"In all {runs} of our full-length runs with the redraw off, this loop lost no frame."
    worst = f"{float(f'{max(lost):.2g}'):g}"  # two digits, never in the 1e+02 form
    return (f"In {clean} of our {runs} full-length runs with the redraw off, this loop lost no frame, "
            f"and in none did it lose more than {worst} % of the frames.")


def lost_frames(log_csv):
    """[(frame number, ms between the rows of the nearest frames before and after it)] for each frame that a V3 or V4 log lacks
    between its first and its last frame number. Reads the file the loop wrote (the frame number is its column sensor_seq)."""
    t_s, seq = (np.loadtxt(log_csv, delimiter=",", skiprows=1, usecols=c) for c in (0, 3))
    seen = seq > 0
    t_s, seq = t_s[seen], seq[seen].astype(int)
    missing = np.setdiff1d(np.arange(seq[0], seq[-1] + 1), seq)
    after = np.searchsorted(seq, missing)  # index of the first frame seen after each missing one
    return [(int(k), 1e3 * (t_s[i] - t_s[i - 1])) for k, i in zip(missing, after)]


def lost_frames_md(version="V4"):
    """What the logs of the freeze say about the frames a version lost: for each seed, the frame numbers, and how far apart in
    time the rows around each lost frame are (a frame period is 20 ms, so 40 ms apart is one wake-up that did not come)."""
    lines = [f"{version}: frames that the log lacks (frame number: ms between the rows before and after it)"]
    for run in sorted(FREEZE.glob("run_s[0-9].txt")):
        found = lost_frames(FREEZE / run.stem[-2:] / f"{version}_log.csv")
        lines.append(f"  seed {run.stem[-1]}: {len(found)} lost" + "".join(f"\n    frame {k}: {gap:.0f} ms" for k, gap in found[:SHOWN]))
        if len(found) > SHOWN:
            lines.append(f"    ... and {len(found) - SHOWN} more")
    return "\n".join(lines)


def lanes_cells(path):
    """{(lane metric, run): cell} from LANES.md."""
    out, lane = {}, ""
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            lane = line[3:].split(":")[0].lower()
        m = re.match(r"^\| ([a-z0-9_]+) \| (.+) \| (.+) \|$", line)
        if m and m.group(1) != "metric":
            out[(f"{lane} {m.group(1)}", "run A")], out[(f"{lane} {m.group(1)}", "run B")] = m.group(2), m.group(3)
    return out


def changes_md(new, new_sweep, new_lanes):
    """NUMBERS_CHANGES.md: one row per metric and version, the old value and the new one."""
    old = read_md_table(STAIRCASE / "NUMBERS.md")
    old_checks = {v: checks_text([checks(p).get(v, []) for p in sorted(STAIRCASE.glob("postfix2_s[0-9].txt"))]) for v in VERSIONS[1:]}
    old.update({("failed checks", v): c for v, c in old_checks.items()})
    lines = ["# Frozen demo numbers: what changed (generated by freeze_numbers.py; do not edit)",
             "Old: `staircase/NUMBERS.md`, the staircase (0.5 mm steps to 4 mm, 2 mm/s ramps, 2 s hold, 16 steps, seeds 1–3, phase 4 ms). "
             f"New: `NUMBERS.md`, {DEFAULTS}, seeds 1–3, phase 4 ms. Both: min–max over seeds. "
             "A cell with – was not measured in that freeze.", "",
             "## Runs (full length)", "| metric | version | old | new |", "|---|---|---|---|"]
    for (metric, version), cell in new.items():
        lines.append(f"| {metric} | {version} | {old.get((metric, version), '–')} | {cell} |")
    old_sweep = {(v, c): old.get((v, c), "–") for v in ("V2", "V3", "V4") for c in ("lost true %", "lost visible %", "lag ms", "phases with 0 %")}
    lines += ["", "## Phase sweep (seed 2, phase 0–18 ms)", "| version | column | old | new |", "|---|---|---|---|"]
    for (version, column), cell in new_sweep.items():
        lines.append(f"| {version} | {column} | {old_sweep.get((version, column), '–')} | {cell} |")
    old_lanes = lanes_cells(STAIRCASE / "LANES.md")
    lines += ["", "## Lanes 4–6 (`LANES.md`; run A, run B)", "| lane and metric | run | old | new |", "|---|---|---|---|"]
    for (metric, run), cell in new_lanes.items():
        lines.append(f"| {metric} | {run} | {old_lanes.get((metric, run), '–')} | {cell} |")
    return "\n".join(lines) + "\n"


def stall_of(output):
    """(seconds, where) of the longest wake-up stall that a run printed (naive_vs_threaded.stall_line)."""
    m = re.search(r"^longest wake-up stall: (\d+) ms \((.*)\)$", output.read_text(), re.MULTILINE)
    return int(m.group(1)) / 1e3, m.group(2)


def one_run(name, attempt, output, command):
    """Make one run: `command` prints into `output`. -> its conditions (load average at the start and the end, longest stall)."""
    load_start = os.getloadavg()[0]
    with open(output, "w") as f:
        subprocess.run(command, stdout=f, check=True)
    stall_s, where = stall_of(output)
    return {"run": name, "attempt": attempt, "load_start": load_start, "load_end": os.getloadavg()[0],
            "stall_ms": stall_s * 1e3, "stall_where": where, "used": True}


def run_job(name, output, command):
    """One run of the freeze -> its conditions: one entry, or two when the first had a stall longer than STALL_LIMIT_S.

    The first run is then kept as <output>_left_out.txt and marked not used; the second one is used (with a long stall too,
    it is used all the same and the table shows it)."""
    first = one_run(name, 1, output, command)
    if first["stall_ms"] <= STALL_LIMIT_S * 1e3:
        return [first]
    first["used"] = False
    output.replace(output.with_name(output.stem + "_left_out.txt"))
    return [first, one_run(name, 2, output, command)]


def make_runs():
    """The runs of the freeze, one after the other (they share the processor: never two at once, and nothing else running)."""
    py, script = sys.executable, str(HERE / "naive_vs_threaded.py")
    FREEZE.mkdir(parents=True, exist_ok=True)
    jobs = [(f"seed {seed}", FREEZE / f"run_s{seed}.txt", [py, script, "--seed", str(seed), "--out", str(FREEZE / f"s{seed}")])
            for seed in SEEDS]
    jobs.append(("phase sweep", FREEZE / "sweep.txt", [py, script, "--phase-sweep", "--sweep-versions", "V2,V3,V4",
                                                       "--seed", str(SWEEP_SEED), "--out", str(FREEZE / "sweep")]))
    jobs.append(("lanes 4-6", FREEZE / "lanes.txt", [py, script, "--lanes"]))  # also writes out/lanes and out/freeze/LANES.md
    conditions = [c for job in jobs for c in run_job(*job)]
    CONDITIONS.write_text(json.dumps(conditions, indent=1) + "\n")
    print(f"{len(conditions)} runs made, {time.strftime('%H:%M')}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="make the runs first (about 50 minutes)")
    ap.add_argument("--lost-frames", metavar="VERSION", help="print the frames that the logs of the freeze lack (V3 or V4) and stop")
    a = ap.parse_args()
    if a.lost_frames:
        return print(lost_frames_md(a.lost_frames))
    if a.run:
        make_runs()
    runs = sorted(FREEZE.glob("run_s[0-9].txt"))
    refresh_lane_gaps(FREEZE / "LANES.md")
    conditions = json.loads(CONDITIONS.read_text()) if CONDITIONS.exists() else []
    cells, sweep_cells, text = numbers_md(runs, sweep_rows(FREEZE / "sweep.txt"), [int(p.stem[-1]) for p in runs], conditions)
    (FREEZE / "NUMBERS.md").write_text(text)
    (FREEZE / "NUMBERS_CHANGES.md").write_text(changes_md(cells, sweep_cells, lanes_cells(FREEZE / "LANES.md")))
    print(text)


if __name__ == "__main__":
    main()
