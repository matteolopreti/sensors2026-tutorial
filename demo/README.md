# One acquisition loop, five versions: same sensor, different datasheet

One simulated soft sensor, the framework's own run (`tactile_characterization_framework` at 84668db) and five versions of
the loop that records it (`loops.py`), each fixing the previous version's bug. **All numbers on the slides are measured
at the framework's defaults** (below); `out/freeze/NUMBERS.md` is the only place to quote them from.

## Run it

Install uv once (it fetches Python, numpy and matplotlib by itself):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                                         # Mac and Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"      # Windows
```

```bash
uv run demo/app.py                                           # the app as a page in the browser; from the repository root
uv run demo/app.py --classic                                 # the old matplotlib window
uv run --with numpy --with matplotlib demo/naive_vs_threaded.py --fast --seed 1   # the table of the five versions, one Short run each (about 2 min)
uv run --with pytest --with numpy --with matplotlib python -m pytest -q   # the tests, from demo/
```

`uv run demo/app.py` starts a small web server on this computer only (`127.0.0.1`, Python's standard library) and opens
the page in the default browser: the views Run, Guess first (What happened after a run), Compare runs and Settings. The
page needs no internet. Ctrl+C in the terminal stops the run and the server; `--no-browser` prints the address and opens
nothing.

`app.py` carries its own list of packages (PEP 723): `numpy==2.0.2` (`2.1.3` from Python 3.13: numpy 2.0.2 has no
Python 3.13 wheel) and `matplotlib==3.9.4`, Python 3.9 to 3.13 (`requires-python = ">=3.9,<3.14"`). Both packages have wheels for Python 3.9 to 3.13 on Mac and
Windows, so nothing is built on your computer. Checked: the list parses (`tomllib`, Python 3.12) and the pins exist on PyPI
with those wheels. Not checked: a Windows computer, and `uv run` itself (the build sandbox has no network).

## The run, as the framework does it

`core/coordinator.py:174-208`: the stage goes down to the depth at the speed, holds there, and goes back up to the surface;
that is one leg. A run is `cycles + 1` legs. The hold is only at the depth, none at the surface. The controls of the
framework's window and their defaults (`settings.py` lists the line of the framework file for each):

| control | default | range | | simulated, not in the framework | default |
|---|---|---|---|---|---|
| Logging rate | 50 Hz | 1–200 | | Sensor frames per second | 50 |
| Depth | 0.5 mm | 0.1–20 (the framework: 0–20) | | Computer busy | off |
| Speed | 0.1 mm/s | 0.01–5 | | Slow disk | off |
| Hold | 0.5 s | 0–60 | | The sensor sends its own time (V3, V4) | off |
| Cycles | 5 | 1–100 | | | |
| Log to | CSV file on, database file off | | | | |
| Plot window | 10 s | 5–300 | | | |

At the defaults a leg is 5 s down, 0.5 s hold, 5 s up: **a run is 6 legs, 63 s**. The lowest depth is 0.1 mm because at
0 mm the stage never moves and nothing can be measured. In the window the settings are on their own page ("Settings"
button); each box refuses a number outside its range and says so in plain words. The line **"Not the defaults: the numbers on
the slides do not apply."** shows when any setting is away from its default (the redraw switch is not a setting of the run
and does not show it). "Back to defaults" puts every setting back.

**Short run** (button): 1 cycle and a hold of 0.3 s at the framework's depth and speed, so the stage moves at the real
speed: 2 legs, about **21 s** for V1 to V4 and 28 s for V0 (Listing 1 sleeps 2 s after each move). Two Short runs
(V2, then V4) take about 42 s, which fits the 60 s of the story beat. A Short run is not a frozen number: it is for seeing.

| version | loop | fixes | what it still gets wrong (full run, defaults, seeds 1–3; `out/freeze/NUMBERS.md`) |
|---|---|---|---|
| V0 | AIS SI Listing 1: blocking `move_to`, `sleep(2.0)`, one read pair after each move | – | 12 rows in 84 s; no ramp or hold data; full-scale force 3.78–3.79 N against the true 4.51 N (−16 %); 99.7 % of the frames lost |
| V1 | one thread, non-blocking stage, reads at the logging rate, writes the file in the same loop | coverage | gaps up to 70 ms (the slow file writes, see below); lag −11 to −11.5 ms (the loop waits for the frame, then reads the force); 7.6–8.6 % of the frames lost |
| V2 | threads: one reader per device at its own rate, latest-value registers, coordinator, FIFO, file writer (the framework used for the paper) | gaps (1 gap in 100 is longer than 29 ms) | each row stamped on read: the sensor value is 17–18.5 ms older than the force; **35 % of the frames lost (34.6–35.1 %), and the file cannot show it** (no frame number) |
| V3 | V2 plus the fix of pull request #3: each row also has the time the computer read the sensor frame and the frame number (NaN and number 0 before the first frame) | lag (17–18.5 ms down to 6.5–7 ms) | in full-length runs the lag stays above the 5 ms of the lag check (see below). A wake-up just before the frame arrives misses it: **35 % of the frames lost (34.5–35.1 %), and now the file shows it** (the loss seen in the file equals the true loss) |
| V4 | V3 with the coordinator waking up twice per frame (twice the logging rate), writing a row only for a new frame | lost frames | V4 lost no frame in 12 of 13 full-length runs (the 3 seeds and the 10 start phases of the sweep), and at most 0.13 % (4 frames, at the 2 ms phase); in full-length runs the lag is 7–7.5 ms, so it fails the lag check and no other |

**Sign convention.** Lag = sensor time stamp minus force time stamp error; positive means the sensor value is older than the force.

**The new metric: sensor values equal to the one before** (`sensor_repeat_pct`). The sensor sends a new value at every
frame, so a value equal to the one before is a frame that the loop read twice, and a frame read twice means a frame lost
(the loop wakes up as often as frames come). It needs no truth and no frame number: V2 has neither and still shows
35 %. V2: 34.7–35.2 %, V3: 34.6–35.1 %, V0, V1 and V4: 0 %. (In the three full runs the share is within 0.05 points of the true loss; the
test allows 4 points. V4 writes only new frames, so it has none by construction.)

**Frame loss: true and visible.** The simulator numbers every frame it produces, so `frames_lost_true_pct` (frames never
logged) exists for every version. `frames_lost_visible_pct` uses only what the file holds: V0–V2 log no frame number, so it
is `n/a`; V3 and V4 log it, so the gaps show. Checks use the true loss. Loss counts only the frames produced while logging
ran (`rig.t_stop`).

**What a row and a point are.** A *row* is one line of the file; a *point* is its dot on the plot. The thread that writes the rows is
the *file writer*. The names of the metrics in the files still say `samples` (they predate this wording); a "sample" in the
text below is always the material that is pressed.

**V3 and V4 write one row per tick** (V3) or one row per new frame (V4), with five columns:
`t_s, force_N, sensor_V, sensor_seq, sensor_t_s`. `t_s` is the time of the row; `sensor_t_s` is the time at which the
computer read the frame (pull request #3, `_emit` snapshots the sensor first). With **"The sensor sends its own time"** on,
`sensor_t_s` is the frame's own time instead (the sensor's clock). Before the first frame the sensor value is NaN and the
number 0: no value is made up. V2 is unchanged and writes zeros there.

**Start-phase dependence (V2, V3).** Sensor frames arrive at n × 20 ms + 4 ms; the 50 Hz coordinator tick falls on
`--phase-ms` + n × 20 ms. Both clocks are exactly 50 Hz, so what is lost is set by the offset, not by a beat, and by the
host's wake-up jitter (a computer with less jitter may lose less at a given phase). `--phase-ms` (default 4, pinned in
`loops.PHASE_MS`: the tick coincides with the frame's arrival, where the loss peaks). `--phase-sweep` runs the versions of
`--sweep-versions` at 0..18 ms. Frozen sweep (seed 2): V2 loses 0–35 % (lag 14.5–21.5 ms), V3 loses 0–34 % and has 0 % at 10, 14, 16 and 18 ms,
V4 loses 0 % at nine of the ten phases and 0.13 % (4 frames) at 2 ms (`out/freeze/NUMBERS.md`). Same code, different start,
different datasheet.

## The physics, and what is assumed

Standard linear solid sample (k∞ = k₁ = **7 N/mm**, τ = 1.5 s, the probe cannot pull); sensor output `0.5F − 0.02F²` V
through a 50 ms first-order lag; 20 ms frames with 4 ms latency; noise 0.02 N and 5 mV. At the defaults the full-scale
force is 4.5 N and the sample relaxes 6.3 % of it.

- **Assumption, not a measurement: the stiffness, 7 N/mm.** The earlier freeze used 1 N/mm with a 4 mm staircase. At the
  framework's default depth of 0.5 mm that gives 0.65 N, and the noise of 0.02 N then fails the checks of a *perfect* log
  (a log of the truth: full-scale 78 %, relaxation 44 %, lag 10 % of the runs, simulated). At 7 N/mm a perfect log fails none
  of 200 draws. Real stiffness depends on the sample; replace it if a measured one exists (`STIFFNESS_N_PER_MM`).
  `test_demo.py::test_a_perfect_log_passes_the_checks_almost_always` keeps this true.
- **Assumption: the disk.** 5 % of the file writes stall 20–40 ms (`Disk`, `NORMAL_DISK`). V1's gaps come from these stalls
  (`--no-stalls` removes them). "Slow disk": 25 % stall 40–80 ms (`SLOW_DISK`), also an assumption. SQLite uses WAL and
  commits at most once a second (as `data_logger.py` does).
- **Assumption: "Computer busy"** starts one program per processor core that only calculates. It is a real load on the real
  computer, so what it does depends on that computer. No loss figure for a busy machine is quoted: the freeze did not measure
  one. What was measured, with the redraw off and nothing started on purpose: V4 lost no frame in 12 of 13 full-length runs (the
  3 seeds and the 10 start phases of the sweep: `out/freeze/NUMBERS.md`), 4 frames (0.13 %) in the other one, and in the window
  runs (Short run, `out/probes/window_ticks.txt`) 0.00 % in 6 of 8 runs and 0.10 % (one frame) in 2 of 8.
- **The lag check is 5 ms** (`metrics.LAG_LIMIT_MS`; the code divides by nothing, it does not depend on the logging rate).
  V3 and V4 take the computer's time at the moment the frame is read, so their sensor value is late by the sensor's 4 ms
  latency and the wake-up of the reader: measured in the full-length runs of the freeze 6.5–7 ms (V3) and 7–7.5 ms (V4), and
  **they fail this check** (a single Short run can pass it) (owner decision, 2026-10-19: keep 5 ms). Only the setting "The sensor sends its own time" passes it (lag within
  2.5 ms in the tests). In the full-length runs V2's 17–18.5 ms and V1's −11 to −11.5 ms fail it too.
- Ground truth is the simulator's noiseless 1 kHz trace through the same `metrics.datasheet` as every log. `n/a` = not
  measurable from that log. Hysteresis is not a check: it follows the time offset between the force and the sensor value of each row (2.2–4.1 % over V1 to V4 in the freeze, against a truth of 2.9 %; pairing each sensor value with the force at its own time gives 2.9 % for every version: critique/hysteresis_cause_2026-10-20.md).

`metrics.failures` names the checks a log fails against truth. Frozen result: V0 fails 8, V1 fails frames lost, jitter and
lag, V2 fails frames lost and lag, V3 fails frames lost and lag, **V4 fails the lag check only** (it lost no frame in 12 of 13 full-length runs). With the
setting "The sensor sends its own time" V4 fails none (`test_demo.py`). `test_demo.py` also checks that a run at the defaults
gives the frozen numbers (V2, V3 and V4, seed 1; the frozen range widened by `ROOM`, and the frames lost by `LOSS_ROOM_POINTS`: 1.5 points for V2 and V3, 0.25 for V4) and that a planted defect is flagged.

Outputs go to `out/` (`--out DIR`): the plots (`timeline`, `loops`, `datasheet`; SVG and PDF), one CSV file per version, and
`V*_truth.csv` and `V*_ramps.csv` (the truth at 1 kHz and the ramps). In `out/freeze/s1…s3` the truth files of V0 to V3 are
left out to save 60 MB (V4's is kept; run `naive_vs_threaded.py --seed N` to make them).

## The numbers of the freeze

`python freeze_numbers.py` writes, from the three full runs and the sweep (`python freeze_numbers.py --run` makes them,
about 50 minutes, nothing else running):

- `out/freeze/NUMBERS.md`: per metric and version, min–max over seeds 1–3, the frames lost seed by seed, the sweep, and the
  **Run conditions** (the load average at the start and the end of each run, the longest wake-up stall; a run with a stall over
  200 ms is made once more, both are listed, the clean one gives the ranges). The raw values are in `out/freeze/conditions.json`.
- `out/freeze/LANES.md`: the numbers of lanes 4 to 6.
- `out/freeze/NUMBERS_CHANGES.md`: one row for each metric and version, the old value (the staircase freeze, kept in
  `out/freeze/staircase/`) and the new one.

The runs were made on this Mac (15 processor cores; Python 3.9.6, numpy 2.0.2, matplotlib 3.9.4). **The Mac was not idle**: other
programs ran on it that I could not stop, and the 1-minute load average was 3.4–5.3 at the start of the runs (the "Run
conditions" table of `NUMBERS.md` lists it for each run, and it includes the run itself). So these are not the runs of a computer with
nothing else on it. What the runs show about that: each run records the longest time a thread woke up late (`StallWatch`,
`naive_vs_threaded.py`), and in all five runs it is 11–28 ms, far below the 200 ms at which a run is made again (none was). The
previous freeze (its logs are in commit 68d6988, `demo/out/freeze/s1/`) had a hold of the whole process in seed 1 (V2: one gap of 2.2 s;
V3 and V4: 0.28 and 0.27 s); this freeze has none:
the longest gap in any log is 70 ms for V1 (the slow file writes), 30 ms for V2 and V3, 35 ms for V4, and V0 has its 7 s steps.

The watch is a thread that sleeps 10 ms again and again, so it runs beside the loops in every run of the freeze. Its effect on the
lag was not measured with a kept output: the frozen numbers are those of the loops with the watch on.

**What the logs say about V4's lost frames.** `python freeze_numbers.py --lost-frames V4` lists the frame numbers that a log
lacks. In seeds 1–3 it lists none (the file has every frame number from the first seen to the last). In the sweep (seed 2) the run at the 2 ms
phase lost 4 frames (0.13 %); the sweep keeps only the last log of its runs, so the numbers of those frames are not known. A V4 loss
is a late wake-up: the sensor reader's or the coordinator's, more than 5 ms late (`test_v4_loses_no_frame`); the longest stall of the
sweep is 25 ms and was in a V3 run (phase 12 ms); the sweep does not say what the stall was in the V4 run at 2 ms. The earlier freeze's explanation, "V4's lost frames are the 0.27 s hold", does not apply to
this freeze, and it was not complete even for the earlier one: its 30 lost frames (seed 1, `demo/out/freeze/s1/V4_log.csv` in commit 68d6988; `--lost-frames` on it) are spread over frame numbers 69 to 2280.

## Lanes 4–6 (Fig. S8): why the last three lanes exist

`python naive_vs_threaded.py --lanes` (about 1 minute; seeds 1–3) runs each lane as run A (the obvious code) against run B
(the design), both on the V4 loop (`lanes.py`). It prints one table per lane and writes `out/lanes/` (plots, logs, truth) and
`out/freeze/LANES.md` (ranges over seeds; **quote numbers only from it**). Tests: `test_lanes.py` (passes on B, fails on A).

| lane | run A | run B | metric |
|---|---|---|---|
| 4 blocking vendor call | coordinator calls blocking `move_to` itself | a short-lived motion thread per move (framework `motion_worker`) | ramp rows logged |
| 5 plot redraw | redraw inside the coordinator tick | 20 Hz GUI timer thread copies the log and redraws (framework `QTimer` 50 ms) | 1 gap in 100 (`dt_p99_ms`), longest gap (`dt_max_ms`), gaps > 100 ms |
| 6 STOP in a blocking move | flag read between moves; the move in flight is joined, then the retract | flag read every tick; halt the stage; forced retract to +5 mm | time from STOP until depth stops growing; extra depth |

**Lane 5 cost model (assumption built on a measurement).** `redraw_cost.py` times `fig.canvas.draw()` headless
(`MPLBACKEND=Agg`, matplotlib 3.11, this Mac) at 1k / 10k / 50k points of one line (`out/lanes/redraw_cost.txt`). The simulator
uses the linear fit, cost = 11.8 ms + 0.96 µs × points (`lanes.REDRAW_A_S`, `REDRAW_B_S`), with 50 000 points on screen when the
run starts, growing with every logged row. The window resize is one redraw at 3× the cost, at 1.5 s. The wait is `time.sleep`:
no GIL contention of a real GUI thread is modelled.

**Public framework probe (read-only).** `out/lanes/probe_public.txt`: a probe on the framework's own mock drivers. Lane 6 run A
models what it showed: stop() during a move does not interrupt the move (`t.join()` waits for it), then the retract runs. Run B is
the design as published (D.5), not the repo.

## The window (`app.py`)

```bash
uv run demo/app.py        # one window; Mac, Windows and Linux
```

Left: the version (V0 to V4), the redraw switch, **Start**, **Stop**, **Hold the picture**, **Short run**, **Back to defaults**, the
"Not the defaults" line and the counters. Right: three tabs over the one page on show (see "The teaching layer" below). The page
**The run**: the true force and the points written to the file (the last *Plot window* seconds of the run). Under it, **where the
time goes**: one row for each thread of the version (V0 and V1: one; V2 to V4: main loop (the coordinator), sensor reader, load-cell reader, the
file writer) with one coloured block for each wait, at its real start and end time, and on top of them the **frame strip** (one mark
for each frame the sensor sent, at the time it arrived: seen once, seen twice, lost). A thin orange line on the last row marks the
time in each row of the file. The load-cell reader reads for 2 µs and sleeps for 1 ms, so its
row is drawn as the sleep it mostly is: where waits are narrower than a pixel, the row has the colour of the wait that takes most
of the time. The colour key lists only the kinds of wait the shown version can have ("reading the load cell" only in V0 and V1;
"redrawing the plot" only when the redraw switch is on) and the frame marks that version can show. `loops.py` reports each wait through an optional
`trace(thread, kind, t0, t1)` argument; without it (the default, and for every frozen number) the loops run exactly as before.
The same colours are on the slides: `python naive_vs_threaded.py --fast --timeline OUTDIR` writes one 1920 × 1080 PNG per version.

**"1 gap in 100 is longer than ..."** has one definition everywhere: the window, the notebook, the freeze table (`NUMBERS.md`, the row
`dt_p99_ms`, whose name is the old one) and the handout. It is the recipe of the handout (`metrics.one_gap_in_100`): sort the gaps between
two rows from the longest to the shortest and read the value 1 in 100 from the top. With *n* gaps that is the *k*-th longest gap,
*k* = ceil(*n* / 100): the 10th of 1000 gaps, the 2nd of 150, the longest of 100. No interpolation: the value is one of the gaps. From 100
gaps on; with fewer the window shows only the longest gap (the function gives the longest gap then, so the lane 5 run A of
`LANES.md`, with 45 or 46 gaps, has the same value on both lines). Before this, the window, the freeze and the notebook used a
percentile that interpolates between two gaps and could differ by 1 ms from the handout's number.

**The teaching layer** (`teach.py` is the logic, `app.py` the window; the words are those of `design/APP_TEACHING_SPEC.md` section 4,
except the fixed words of 2026-10-19: the panel is "Where the time goes", a *row* is a line of the file, a *point* is a dot of the
plot, "the file writer", and "longest gap" is the maximum only). "Tick" is the loop's pass; "check" is only "Checks on the file".
All of it is read from the real run: the rows, the frame numbers and the waits the loops report. Nothing is slowed down.

| part | what it does |
|---|---|
| Counters | "Frames lost, as your file shows it" (from the frame numbers; V0 to V2 have none and the line says so); "Frames lost, the simulator knows" (in a muted colour: only the simulator knows it); "Sensor values equal to the one before" (the share of rows that repeat the value of the row before); the gaps |
| Frame strip | the marks above; the fate of a frame is known when the file is final (a frame not yet seen is not yet lost) |
| "What just happened" | one caption line under the timeline. A rule of `teach.Detector` turns what the run did (a late write, a missed frame, a repeat, a long gap, a redraw) into one sentence, 14 of them; the captions queue and each stays long enough to read (`teach.CaptionLine`) |
| **Hold the picture** | stops the redraw of the plot, the timeline, the bars and the caption so that one moment can be read; the run and the counters go on. **Let it run** goes on from the present |
| Guess first | the tab before a run: a question about the chosen version with four answers to think over (nothing to click; no score). When the run has finished the same tab, now **What happened**, gives the answer worked out from that run, then "What happened in this run" (up to 5 events) above "Checks on the file" |
| Compare runs | "Where the time goes": for each thread, the share of the whole run spent on each wait (from `teach.Tracer`, which sums every wait of the run; the timeline holds only the last seconds). A part of 5 % or more of the run has its percent; the part that no wait fills is empty with an outline and is called "no wait recorded"; a colour key at the bottom of the page names the colours. Then the table "Runs so far", newest on top, with the settings of each run |

The question and the answer of V3 describe the fix of pull request #3 (the file carries the frame number), as the rest of the app does.
Not built: runs side by side, slow motion, scored quizzes, a start-time control, live highlighting of the code, a tour.

Colours are palette A (paper background, ink text, orange for the points written to the file, warm grey for the true force). The
font is IBM Plex Sans when it is installed; if not, matplotlib's own font, and the program says so once.

| version | what its timeline shows |
|---|---|
| V0 the script the paper warns against | blocked in the move and in the sleep nearly all the time; one row after each step |
| V1 one loop | the loop waits for the sensor frame most of the time; each slow disk write is one long block, and no row is written inside it |
| V2 the framework used for the paper | four threads; a slow disk write stops only the file writer; the main loop and the sensor reader wake up almost together, so a row gets the new frame or the old one by chance |
| V3 the fix of pull request #3 | the same threads; a wake-up just before the frame arrives misses it, so about a third of the frames are missing (35 %, `out/freeze/NUMBERS.md`); the frame numbers show which |
| V4 the loop looks twice per frame | the main loop wakes up twice per frame; the sentence in the window says in how many of the 13 full-length runs of the freeze (3 seeds, 10 sweep phases) it lost no frame with the redraw off (12 of 13), and the most it lost: made from `NUMBERS.md` by `freeze_numbers.loss_sentence`, when the caption is made |

**Stop and close.** Stop ends the run at the next tick for V1 to V4 (measured under 0.2 s; the limit in the code is 2 s). V0 is
blocked in its move and its sleep and stops between steps: at the defaults up to a 5 s leg plus 2 s, about 7 s. Closing the
window does the same and waits for it, so the window does not answer meanwhile. Start while a run is going or stopping does
nothing; the first counter line says so. A *Short run* or any run ends on its own.

**"Redraw inside the logging loop" (V2 to V4 only).** The window itself is always drawn by the window's own timer and only the
main thread touches it. The switch is read at Start and stays fixed for that run. On: every tick of the logging loop first draws a
second copy of the plot (its own picture in memory, the size of the window) and waits for it. Off: the loop never waits for a
draw. The draw shows in the timeline as black blocks on the main-loop row, and the gap it makes in the counter "Gaps longer
than 30 ms". A 20 ms gap is about one pixel on the force plot, so the plot cannot show it.

Measured as in "The window's own cost" below: the Short run (21 s), a 1333 × 750 window with the panel on and its timer at
100 ms (`app.GUI_PERIOD_MS`), seeds 1–5, five runs for each case (min–max):

| | V2 off | V2 on | V4 off | V4 on |
|---|---|---|---|---|
| longest gap between points (ms) | 26–35 | 43–52 | 32–39 | 41–62 |
| 1 gap in 100 is longer than (ms) | 25–29 | 40–42 | 31 | 39–48 |
| gaps longer than 30 ms | 0–3 | 100–113 | 59–73 | 103–179 |
| frames lost (%) | 32–35 | 5.0–7.0 | 0–0.10 | 4.7–6.4 |
| points in the file | 1031–1032 | 1030–1032 | 1029–1031 | 966–983 |

(The line "1 gap in 100" of this table was read with the percentile that interpolates, before the recipe above; the five runs were not
made again. The recipe may give values about 1 ms away from these.)

**V4 shows the effect honestly**: with the switch on it loses 4.7–6.4 % of its frames, and the timeline shows a black redraw block
of about 5 ms in each tick. **V4 off has gaps over 30 ms too (59–73 of about 1030)**: V4 writes a row only for a new frame, a frame
is found at the next 10 ms tick, so two rows are 20 ms apart plus or minus a tick, and the wake-up jitter puts a few of them
past 30 ms. Compare V4 on with V4 off, not with zero. **V2 shows the redraw in the gaps but not in the frames**: with the switch on it
loses *fewer* frames (5–7 % against 32–35 %), because the draw moves the coordinator's read away from the moment the frame
arrives (the pinned start phase, `loops.PHASE_MS`, is where V2 loses most). Do not quote V2's frame loss for the redraw; quote its
gaps and its timeline. The redraw costs real time only: no sleep and no added cost.


**The window's own cost.** The window is drawn by the main thread, which the loops share with it. The panel is made once and
only updated at each tick; the colour key is plain texts laid out once (a legend is laid out again at every draw, 4 ms).
Measured as the app does it (the window's timer driven by hand, Agg, V4, redraw off, Short run, eight runs, seeds 1–8, this Mac,
other programs on it: the 1-minute load average was 3.2 at the start and 2.9 at the end, and no run had a wake-up stall over 200 ms;
all in `out/probes/window_ticks.txt`, made by `measure_window.py ticks`): the tick (one `on_timer()`, which includes its one draw)
is **23–27 ms** (median per run; the longest tick of a run 35–58 ms), with the timer at 100 ms; **frames lost: 0.00 % in 6 of 8
runs and 0.10 % (one frame) in 2 of 8**. The table above (redraw switch off and on) has no output file in the repository: it was
measured before the settings widgets and the plot window were added, and `measure_window.py redraw` makes it again and writes
`out/probes/window_redraw.txt`.
One caveat on the old figure: `measure_window.py` used to draw the figure a second time after `on_timer()`, which on the Agg canvas
doubled the tick; a real window draws once.

**Rows and frames.** "Rows in the file" can be larger than "Frames the sensor sent" (for example 201 and 200): V3 and V4
write their first row before the first frame has arrived (frame number 0, value NaN), and the frame count stops one tick before
the end of logging. V2 writes a few rows in the same way (with zeros). Second route: count the rows with frame number 0 in the file.

**Tests without a window:** `MPLBACKEND=Agg python -m pytest -q -p no:cacheprovider` in `demo/` (`pytest.ini`
leaves `out/` out: it holds old copies of the code). Every test writes to a temporary folder.

**Pictures of the teaching layer.** `python out/app_preview/render_teaching.py [OUTDIR]` runs the real loops and saves 1920 × 1080
files to `out/app_preview/` without opening a window: `teaching_V0.png` to `teaching_V4.png` (each version at the defaults, in the
middle of the run, with a caption on show), `teaching_question_V3.png`, `teaching_answer_V2.png` and `teaching_compare_V2.png`.
`out/app_preview/NOTES.md` says what each one shows. Each takes as long as the run has gone when it is taken (6 to 12 s).

**Pictures of the window.** `python render_window.py [OUTDIR]` runs the real loops, drives the window's timer by hand and saves
three 1920 × 1080 files to `out/app_preview/` without opening a window: `window_defaults.png` (V2 at the defaults, 12 s into a
run), `window_changed.png` (V1 with the slow disk on: the "Not the defaults" line shows), `window_settings.png` (the settings page).
`python measure_window.py ticks|redraw` makes the figures above.
