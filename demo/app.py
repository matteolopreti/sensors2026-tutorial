# /// script
# requires-python = ">=3.9,<3.14"
# dependencies = [
#   "numpy==2.0.2; python_version < '3.13'",
#   "numpy==2.1.3; python_version >= '3.13'",
#   "matplotlib==3.9.4",
# ]
# ///
"""Interactive window for the live session: runs the real acquisition loops V0-V4 and shows the log as it grows.

Run:  uv run demo/app.py        (from the repository root; or:  python app.py  with numpy and matplotlib installed)

Three parts, kept apart:
  Session     one run of one loop version in a worker thread; thread-safe snapshots; stops cleanly, also after an error.
  View        the matplotlib window; only the main thread touches its figure; a GUI timer reads the snapshots.
  LoopRender  an Agg copy of the plot (its own figure) that the logging loop draws when the redraw switch is on.
The loops, the rig and the checks are the ones of naive_vs_threaded.py, loops.py and metrics.py. The run controls are the
framework's, each starting at the framework's default (settings.py). The panel under the force plot draws the waits that
the loop itself recorded (timeline.py): where the time goes, and one mark for each frame the sensor sent (the frame strip).
The teaching layer (teach.py) tells the causes: a caption line while the run goes, and after it the answer to the question that
was asked before Start, what happened in the run, the share of the time each thread spent on each wait, and the runs so far.
"""
from __future__ import annotations

import argparse
import math
import tempfile
import textwrap
import threading
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Button, CheckButtons, RadioButtons, TextBox

import loops
import metrics
import naive_vs_threaded as demo
import settings
import teach
import timeline
from plots import PALETTE

TAU_RELAX_S, TAU_SENSOR_S = 1.5, 0.05  # the sample and the sensor: the defaults of naive_vs_threaded.py
VERSION_LABELS = {**timeline.TITLES, "V4": "V4  the loop ticks twice per frame"}  # "tick" is the word for one pass of the loop (spec 1)
REDRAW_LABEL = "Redraw inside the logging loop\n(chosen at Start)"
REDRAW_LABEL_UNAVAILABLE = "Redraw inside the logging loop\n(only V2–V4)"
FONT_SIZE = {"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20}
GUI_PERIOD_MS = 100  # 10 updates per second: an update (plot, panel, text) takes 20-30 ms of the main thread, which the loops share
MAX_SENTENCES = 7
MIN_GAPS_FOR_1_IN_100 = metrics.GAPS_FOR_1_IN_100  # fewer gaps than this: "1 gap in 100" says nothing
GAP_LIMIT_MS = 30  # the counter "Gaps longer than 30 ms": a 20 ms tick that was late by half a period or more
COUNTER_GAP_IN = 0.05  # space between two blocks of the counters, inches
TEXT_CHARS = 52  # characters of 20 pt text that fit the text blocks of the right column
NOTE_CHARS = 44  # ... and the left column
CAPTION_LINES = 3  # lines of the caption "What just happened": the longest sentence of teach.CAPTIONS, with its count, fits
PEAK_MARGIN = 1.15  # room above the sample's largest force, for the noise
STOP_LIMIT_S = 2.0  # a stopped run of V1 to V4 ends within this (measured: under 0.2 s); V0 is blocked until its move and sleep end


def plot_font():
    """The size of every text in the window, and IBM Plex Sans when it is installed."""
    return {**FONT_SIZE, **timeline.font_params()}


def _ms(x):
    return f"{x:.0f} ms" if math.isfinite(x) else "-"


# --- the logic: no matplotlib window in here -------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """What the log holds at one moment. rows: (t, force, sensor[, frame number, time of the sensor value]) as written."""
    rows: np.ndarray
    truth: np.ndarray  # columns t, force, sensor at 1 kHz
    now: float
    frames: np.ndarray  # the frame number that each row carries, also for the versions whose file does not (V0-V2)


@dataclass(frozen=True)
class Counters:
    rows: int
    frames_sent: int
    lost_pct: float | None  # the simulator knows; None before any frame is known
    lost_seen_pct: float | None  # as the file shows it; None: the file has no frame numbers (V0-V2), or no frame yet
    repeat_pct: float  # % of the rows whose sensor value equals the one before; NaN below two rows
    longest_gap_ms: float
    gap_1_in_100_ms: float  # metrics.one_gap_in_100: the k-th longest gap, k = ceil(gaps / 100); NaN below 100 gaps
    gaps_over_limit: int  # gaps between rows longer than GAP_LIMIT_MS


def expected_duration_s(version, values):
    """About how long a run takes: V0 sleeps 2 s after each of its moves, the others follow the protocol."""
    if version == "V0":
        return 2 * (values["cycles"] + 1) * (settings.leg_s(values) + loops.V0_SLEEP_S)
    return settings.run_s(values)


class Session:
    """One run of one loop version on its own simulated rig, in a worker thread."""

    def __init__(self, version, values, out, seed=1, on_tick=None):
        """values: the settings of the run (settings.defaults() shape). on_tick: called inside every tick of the logging
        loop (V2-V4 only), in the loop's thread."""
        self.version, self.values = version, dict(values)
        self.steps = demo.protocol(values["depth"], values["hold"], values["cycles"])
        self.redraw_in_loop = on_tick is not None and version in loops.THREADED
        self.on_tick = self._guarded(on_tick) if self.redraw_in_loop else None
        self.rig = demo.build_rig(f"app_{version}", seed, values["speed"], TAU_RELAX_S, TAU_SENSOR_S, out=out,
                                  **settings.rig_options(values))
        self.log, self.sheet, self.error = None, None, None  # sheet: (datasheet, truth datasheet) of a complete run
        self.recorder = timeline.Recorder(timeline.THREADS[version])  # the waits of the loop, for the timeline panel
        self.tracer = teach.Tracer(self.recorder)  # the loop reports to this: the recorder keeps the newest waits, it the sums
        sensor = self.rig.sensor
        self.detector = teach.Detector(version, sensor.period, sensor.latency, len(self.steps))
        self.events = []  # what the detector found, oldest first
        self.t_end = None  # world time at which the run ended
        self.busy_workers = []  # the programs of "Computer busy" while the run goes
        self.cut = False  # Stop ended the run before its end
        self._stop, self._done, self._lock = threading.Event(), threading.Event(), threading.Lock()
        self._rows, self._truth = [], np.empty((0, 3))
        self._frames, self._loop_log = [], []  # frame number of each row: V0 and V1 note it as they read; V2-V4 hand it over in log
        self._read = threading.local()  # the frame number of the newest read of this thread
        write, read = self.rig.disk.write, sensor.read

        def read_and_note():
            frame = read()
            self._read.k = frame[2]
            return frame

        def write_and_keep(row):  # every version writes each row through the disk: the one place to watch the file grow
            write(row)
            with self._lock:
                self._rows.append(row)
                if version not in loops.THREADED:
                    self._frames.append(self._read.k)

        sensor.read, self.rig.disk.write = read_and_note, write_and_keep
        self._thread = threading.Thread(target=self._work, daemon=True)

    @property
    def done(self):
        return self._done.is_set()

    def start(self):
        self._thread.start()

    def stop(self):
        """Ask the loop to end. V1 to V4 end at once; V0 is blocked in its move or its sleep until that ends."""
        self._stop.set()

    def stop_limit_s(self):
        """The longest a stopped run takes to end."""
        if self.version == "V0":
            return settings.leg_s(self.values) + loops.V0_SLEEP_S + 1.0
        return STOP_LIMIT_S

    def fail(self, exc):
        """Keep the first error and stop the run: the run ends at once, with no checks."""
        if self.error is None:
            self.error = exc
        self._stop.set()

    def _guarded(self, on_tick):
        """The hook for the logging loop: an exception there would end the loop's thread without a word, so stop the run."""
        def tick():
            if self.error is None:
                try:
                    on_tick()
                except Exception as exc:
                    self.fail(exc)
        return tick

    def join(self, timeout=None):
        self._thread.join(timeout)

    def loop_kwargs(self):
        """What this version's loop takes of the settings, and the hooks of the window."""
        kw = {"trace": self.tracer, "cancel": self._stop}
        if self.version != "V0":
            kw["rate_hz"] = self.values["rate"]  # V4 ticks at twice this
        if self.version in loops.THREADED:
            kw["log"] = self._loop_log  # the loop's own rows, with the frame number of each
        if self.redraw_in_loop:
            kw["on_tick"] = self.on_tick
        if self.version in ("V3", "V4"):
            kw["sensor_time"] = self.values["sensor_time"]
        return kw

    def _work(self):
        try:
            with settings.busy_machine(self.values["busy"]) as self.busy_workers:
                self.log = demo.run(loops.VERSIONS[self.version], self.rig, self.steps, **self.loop_kwargs())
            self.cut = self._stop.is_set()  # Stop was pressed before the run ended
            if not (self.cut or self.error):
                self.sheet = demo.evaluate(self.rig, self.log)
        except Exception as exc:  # shown in the window; a run that failed has no checks
            self.fail(exc)
        finally:
            self.t_end = self.rig.world.now()
            self._done.set()

    def snapshot(self):
        """Copy of the log and of the true force so far; safe from any thread."""
        world = self.rig.world
        if not self.done:
            world.sample()  # integrate the true force up to now (a finished run already did)
        with self._lock:
            rows = np.array(self._rows, float) if self._rows else np.empty((0, 3))
            frames = np.array(self._frames if self.version not in loops.THREADED else [r[3] for r in self._loop_log], int)
            new = np.array(world.truth[len(self._truth):], float).reshape(-1, 3)
            self._truth = np.vstack([self._truth, new])
            rows, frames = rows[:len(frames)], frames[:len(rows)]  # the loop's own list is a row behind the file for a moment
            return Snapshot(rows, self._truth, world.now(), frames)

    def trace_view(self, snap):
        """What the timeline panel draws: (waits per thread, window in s, times of the rows in the file).

        The window is the last WINDOW_S seconds of the run, ending now, or at the end of a run that has ended;
        in the first seconds it starts at 0 and the bars grow into it."""
        end = self.t_end if self.done else snap.now
        window = timeline.window_ending_at(self.version, end)
        return self.recorder.spans(since=window[0]), window, snap.rows[:, 0]

    def plot_window(self, snap, seconds):
        """(t_lo, t_hi) of the force plot: the last `seconds` of the run, ending now, or at the end of a run that has ended."""
        return timeline.last_seconds(self.t_end if self.done else snap.now, seconds)

    def frames_sent(self, snap):
        """How many frames the sensor has sent: those that have reached the computer, up to the end of logging."""
        rig = self.rig
        t_sent = rig.t_stop if self.done else snap.now
        # LOCAL COPY of the rule of naive_vs_threaded.evaluate (n_produced): the live count needs it before a
        # datasheet exists. One source would need evaluate to expose it: REQUEST in the hand-back of T19b.
        return max(math.floor((t_sent - rig.sensor.latency) / rig.sensor.period), 0)

    def frame_marks(self, snap):
        """(seconds each frame reached the computer, what became of it) for the frame strip: all frames sent so far."""
        sent = self.frames_sent(snap)
        sensor = self.rig.sensor
        return teach.frame_arrivals(sent, sensor.period, sensor.latency), teach.frame_fates(snap.frames, sent, self.done)

    def new_events(self, snap):
        """The causes found since the last call (teach.Detector), and kept in self.events. Call with a snapshot taken just
        before: the waits are read after the rows, so that the waits of every row are there."""
        spans = self.recorder.spans(since=self.detector.cursor)[timeline.THREADS[self.version][0]]
        new = self.detector.step(snap.now, spans, snap.rows[:, 0], snap.frames)
        self.events += new
        return new

    def counters(self, snap):
        """The counters of a snapshot; after a complete run the values of the datasheet check."""
        t = snap.rows[:, 0]
        gaps = np.diff(t) * 1e3
        over = int(np.sum(gaps > GAP_LIMIT_MS))
        sent = self.frames_sent(snap)
        numbered = snap.rows.shape[1] >= 4  # the file has frame numbers (V3, V4)
        if self.sheet:
            ds = self.sheet[0]
            return Counters(ds["samples"], sent, ds["frames_lost_true_pct"],
                            ds["frames_lost_visible_pct"] if numbered else None, ds["sensor_repeat_pct"], ds["dt_max_ms"],
                            ds["dt_p99_ms"] if ds["samples"] > MIN_GAPS_FOR_1_IN_100 else math.nan, over)
        seen = snap.rows[:, 3] if numbered else []
        seen_pct = metrics.frames_lost_pct(seen) if np.any(np.asarray(seen) >= 1) else None  # frame 0 is no frame
        lost = teach.loss_pct(teach.frame_fates(snap.frames, sent, final=False))
        return Counters(len(t), sent, lost, seen_pct, metrics.sensor_repeat_pct(snap.rows[:, 2]),
                        gaps.max() if len(gaps) else math.nan,
                        metrics.one_gap_in_100(gaps) if len(gaps) >= MIN_GAPS_FOR_1_IN_100 else math.nan, over)

    def numbers(self, snap):
        """The numbers that the answers of teach.py are made of, from the counters."""
        c = self.counters(snap)
        return teach.RunNumbers(c.rows, c.frames_sent, c.lost_pct or 0.0, c.lost_seen_pct, c.repeat_pct, c.gap_1_in_100_ms,
                                c.longest_gap_ms)

    def shares(self, snap):
        """{thread: {kind of wait: share of the run}} so far: where the time went (teach.wait_shares)."""
        run_s = max(self.t_end if self.done else snap.now, 1e-9)
        return teach.wait_shares(self.tracer.seconds(), timeline.THREADS[self.version], run_s)

    def failed_checks(self):
        """Names of the checks of metrics.failures that this complete run fails."""
        ds, truth = self.sheet
        return metrics.failures(ds, truth, 1e3 / self.values["rate"])


def _num(x, fmt):
    return format(x, fmt) if math.isfinite(x) else None


def _differs(label, got, true, fmt, unit):
    if (got_text := _num(got, fmt)) is None:
        return f"{label}: cannot be measured."
    return f"{label}: {got_text} {unit} (true {format(true, fmt)} {unit})."


# one plain sentence per check of metrics.failures; the key is the check's name there
SENTENCES = {
    "no ramps": lambda ds, tr: "No points while the stage moved.",
    "<10 dwell samples": lambda ds, tr: "Fewer than 10 points in some hold.",
    "jitter p99>2T": lambda ds, tr: (f"1 gap in 100 is longer than {ds['dt_p99_ms']:.0f} ms."
                                     if ds["samples"] > MIN_GAPS_FOR_1_IN_100
                                     else f"The longest gap between points is {ds['dt_max_ms']:.0f} ms."),  # fewer than 100 gaps
    "frames lost>1%": lambda ds, tr: f"{ds['frames_lost_true_pct']:.0f} % of the sensor's frames are missing.",
    "lag>5ms": lambda ds, tr: (f"Force and sensor differ in time by {abs(ds['sensor_force_lag_ms']):.0f} ms."
                               if math.isfinite(ds["sensor_force_lag_ms"]) else "Force and sensor time cannot be compared."),
    "full-scale>5%": lambda ds, tr: _differs("Largest force", ds["force_fs_N"], tr["force_fs_N"], ".2f", "N"),
    "relaxation>20%": lambda ds, tr: _differs("Force drop in holds", ds["relaxation_pct_fs"], tr["relaxation_pct_fs"], ".0f", "%"),
    "sensitivity>5%": lambda ds, tr: _differs("Sensor output for 1 N", ds["sensitivity_V_per_N"], tr["sensitivity_V_per_N"], ".2f", "V"),
}


def check_sentences(session):
    """Plain sentences for the checks a complete run fails; one sentence if it fails none."""
    ds, truth = session.sheet
    failed = session.failed_checks()
    if not failed:
        return ["This file passes every check."]
    lines = [SENTENCES[name](ds, truth) for name in failed]
    if len(lines) > MAX_SENTENCES:
        lines[MAX_SENTENCES - 1:] = [f"and {len(lines) - MAX_SENTENCES + 1} more checks fail."]
    return [textwrap.fill(line, TEXT_CHARS) for line in lines]


# --- the plot, drawn twice: in the window (main thread) and in the logging loop (loop thread) -----------------------


def plot_artists(ax, window_s, y_top):
    """Axes, true-force line and marks of the plot; the window and the loop's copy are made by this one function."""
    with plt.rc_context(plot_font()):
        timeline.seconds_axis(ax, nbins=6)
        ax.set_ylabel("Force (N)")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.set_ylim(0, y_top)
        ax.set_xlim(0, window_s)
        ax.tick_params(axis="x", pad=14, labelsize=20)  # keeps the two zeros apart
        ax.tick_params(axis="y", labelsize=20)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        (truth_line,) = ax.plot([], [], color=PALETTE["muted"], lw=4, alpha=0.55, label="True force")
        (marks,) = ax.plot([], [], "o", color=PALETTE["accent"], ms=5, label="Points in the file")
        ax.legend(loc="upper right", frameon=False, fontsize=20)
    return truth_line, marks


def peak_force_n(values):
    """The top of the force axis (whole N) for a run with these settings: above the sample's largest force. At the end of
    a ramp the dashpot arm lags the stage by speed x time constant, or by the whole depth if that is less."""
    lag_mm = min(values["depth"], TAU_RELAX_S * values["speed"])
    return float(math.ceil(PEAK_MARGIN * demo.STIFFNESS_N_PER_MM * (values["depth"] + lag_mm)))


def show_snapshot(truth_line, marks, snap, t_lo):
    """Put the truth and the rows from t_lo on the plot. Only these are given to the artists: a long run stays cheap to draw."""
    i = max(int(np.searchsorted(snap.truth[:, 0], t_lo)) - 1, 0)
    j = max(int(np.searchsorted(snap.rows[:, 0], t_lo)) - 1, 0)
    truth_line.set_data(snap.truth[i:, 0], snap.truth[i:, 1])
    marks.set_data(snap.rows[j:, 0], snap.rows[j:, 1])


class LoopRender:
    """The plot as the logging loop draws it: a figure of its own, the size of the window in pixels.

    Only the loop's thread touches it after the run has started; the window's figure never leaves the main thread.
    The whole canvas is drawn at every call, so a bigger window costs the loop more time.
    """

    def __init__(self, pixels, window_s, y_top):
        self.fig = Figure()
        FigureCanvasAgg(self.fig)
        with plt.rc_context(plot_font()):
            self.ax = self.fig.add_axes([0.09, 0.13, 0.89, 0.84])
        self.truth_line, self.marks = plot_artists(self.ax, window_s, y_top)
        self.window_s = window_s
        self.pixels = None
        self.resize(pixels)

    def resize(self, pixels):
        """pixels: (width, height, dpi) of the window, read on the main thread."""
        width, height, dpi = pixels
        self.fig.set_dpi(dpi)
        self.fig.set_size_inches(width / dpi, height / dpi)
        self.pixels = pixels

    def prepare(self, window_s, y_top):
        """Called by the main thread at Start, before the loop's thread exists."""
        self.window_s = window_s
        self.ax.set_ylim(0, y_top)

    def draw(self, snap, pixels):
        if pixels != self.pixels:
            self.resize(pixels)
        lo, hi = timeline.last_seconds(snap.now, self.window_s)
        self.ax.set_xlim(lo, hi)
        show_snapshot(self.truth_line, self.marks, snap, lo)
        self.fig.canvas.draw()


# --- the window ------------------------------------------------------------------------------------------------

HOLD_LABEL, RELEASE_LABEL = "Hold the picture", "Let it run"
HOLD_NOTE = ("The plot, the timeline with its frame row, the bars and the caption are held; the run and the counters go on.")
CAPTION_TITLE = "What just happened"
BARS_TITLE, BARS_NOTE = "Where the time goes", "the share of the run that each thread spent on each wait"
TABLE_TITLE = "Runs so far"
TABLE_NOTE = "Frames lost: the simulator knows. Repeats: the share of the rows with a sensor value equal to the one before."
IDLE_TAB, DONE_TAB = "Guess first", "What happened"
NOTES_LINES = 24  # lines of the notes page
RUN_TAB = "The run"
SHARE_LABEL_MIN = 0.05  # a part of a bar of the shares gets its percent when it is at least this much of the run (spec 3.5)
SHARE_KEY_MIN = 0.01  # the key of the bars names a kind of wait when some thread spent at least this much of the run on it
TABLE_LINES = 6  # lines of the body of "Runs so far"; below them the colour key of the bars
TABLE_COLUMNS = (0.2, 0.95, 5.3, 6.3, 8.0, 9.3, 11.3)  # left edge of each column of the table, in inches (it has the width of the window)
SETTINGS_CHARS = 34  # characters of a line of the column "Settings"
TABLE_NOTE_CHARS = 85  # characters of a line of the note under the title of the table


def counter_blocks(c, version, held_note=""):
    """The three blocks of text of the counters (spec 4.3): the lines of the file and the frames, the line of the simulator
    (in the muted colour), and the lines of the gaps. The "1 gap in 100" line only from 100 gaps on."""
    seen = "cannot tell: the file has no frame numbers" if c.lost_seen_pct is None and not _numbered(version) else \
        ("-" if c.lost_seen_pct is None else f"{c.lost_seen_pct:.0f} %")
    top = [f"Rows in the file: {c.rows}", f"Frames the sensor sent: {c.frames_sent}"]
    top += ["Frames lost, as your file shows it:", seen] if seen.startswith("cannot") else [f"Frames lost, as your file shows it: {seen}"]
    muted = [f"Frames lost, the simulator knows: {'-' if c.lost_pct is None else format(c.lost_pct, '.0f') + ' %'}"]
    repeat = "-" if not math.isfinite(c.repeat_pct) else f"{c.repeat_pct:.0f} %"
    rest = [f"Sensor values equal to the one before: {repeat}", f"Longest gap between rows: {_ms(c.longest_gap_ms)}"]
    if math.isfinite(c.gap_1_in_100_ms):
        rest.append(f"1 gap in 100 is longer than {_ms(c.gap_1_in_100_ms)}")
    rest.append(f"Gaps longer than {GAP_LIMIT_MS} ms: {c.gaps_over_limit}")
    return "\n".join(top), "\n".join(muted), "\n".join(textwrap.fill(line, NOTE_CHARS) for line in rest)


def _numbered(version):
    """Does the file of this version carry the frame number of each row? (V3 and V4: the fix of pull request #3.)"""
    return version in ("V3", "V4")


class View:
    """The window. It owns one Session at a time and reads its snapshots on a GUI timer.

    Only the main thread touches the window's figure, its widgets and its canvas, whatever the redraw switch says.
    The GUI timer always updates the window's plot and counters while a run goes.
    Redraw switch on (read at Start, fixed for the run, V2-V4 only): every tick of the logging loop also draws a copy
    of the plot, the size of the window, in an Agg buffer of its own (LoopRender), and the loop waits for that draw.
    Four pages share the right side of the window: the run (force plot, timeline with its frame strip, the caption "What
    just happened"), the notes (the question to guess before Start; after the run the answer, what happened and the checks),
    Compare runs (where the time goes, as shares, and the table "Runs so far") and the settings (settings.py).
    The settings are read at Start and fixed for the run; the plot window is the one that works at once.
    "Hold the picture" stops the redraw of the force plot, the timeline, the bars and the caption; the run and the counters go on.
    """
    SIZE_IN, DPI = (40 / 3, 7.5), 100  # 1920 x 1080 when saved at 144 dpi
    # inches from the bottom left. Left column: the version, the controls, the counters. Right: tabs, force plot, timeline
    # with its frame strip, the caption, the note. Along the bottom: the colour key.
    RADIO_BOX, REDRAW_BOX = (0.0, 5.75, 4.6, 1.7), (0.0, 5.1, 4.6, 0.6)  # narrow: the labels start 1.15 in from the left edge
    START_BOX, STOP_BOX, HOLD_BOX = (0.2, 4.6, 1.25, 0.45), (1.55, 4.6, 1.25, 0.45), (2.9, 4.6, 3.0, 0.45)
    SHORT_BOX, DEFAULTS_BOX = (0.2, 4.1, 2.3, 0.45), (2.65, 4.1, 3.25, 0.45)
    TAB_BOXES = {"notes": (7.0, 6.97, 2.3, 0.45), "compare": (9.4, 6.97, 2.2, 0.45), "settings": (11.7, 6.97, 1.5, 0.45)}
    COUNTER_TOP = 3.95
    FORCE_BOX, TIMELINE_BOX = (7.15, 5.85, 5.65, 0.95), (8.45, 3.4, 4.35, 1.4)  # the right edge leaves room for the label "12 s" of a tick on it
    TEXT_LEFT = 6.1
    TIMELINE_TITLE_Y, STRIP_LABEL_Y = 5.2, 4.92
    CAPTION_TITLE_Y, CAPTION_TOP, NOTE_TOP = 2.83, 2.65, 1.78
    KEY_BOTTOM = 0.15
    NOTES_TOP = 6.75
    SETTINGS_X = {"label": 6.55, "box": 10.75, "hint": 12.05}  # the columns of the settings page
    SETTINGS_TOP, SETTINGS_PITCH = 6.7, 0.47
    BARS_BOX = (8.45, 4.55, 4.3, 1.4)

    def __init__(self, out, seed=1, values=None):
        self.out, self.seed = Path(out), seed
        self.values = settings.defaults() if values is None else dict(values)
        self.version, self.redraw_in_loop, self.page = "V0", False, "notes"
        self.session = None
        self._shown_final = True
        self._start_refused = False  # Start was pressed while a run was going: say so until the run has ended
        self.held = False  # "Hold the picture": the picture is not redrawn
        self.caption, self._caption_now = teach.CaptionLine(), ""
        self.runs = []  # teach.RunRecord of each complete run of this window, oldest first
        self.run_artists, self.settings_artists, self.settings_widgets, self.key_artists = [], [], [], []
        self.compare_key_artists, self._compare_key_for = [], None  # the key of the bars, and what it was made for
        self.notes_artists, self.compare_artists, self.table_artists = [], [], []
        with plt.rc_context(plot_font()):
            self._build()
        self.read_pixels()
        self.render = LoopRender(self.pixels, self.values["window"], peak_force_n(self.values))
        self.timer = self.fig.canvas.new_timer(interval=GUI_PERIOD_MS)
        self.timer.add_callback(self.on_timer)
        self.timer.start()
        self.fig.canvas.mpl_connect("resize_event", lambda _: self.read_pixels())
        self.fig.canvas.mpl_connect("close_event", lambda _: self.close())

    def read_pixels(self):
        """The window's size in pixels and its dpi, for the loop's copy. Runs on the main thread (start and resize)."""
        width, height = self.fig.canvas.get_width_height(physical=True)
        self.pixels = (width, height, self.fig.dpi)

    def _box(self, x, y, w, h):
        """Axes from inches (x, y from the bottom left)."""
        return self.fig.add_axes([x / self.SIZE_IN[0], y / self.SIZE_IN[1], w / self.SIZE_IN[0], h / self.SIZE_IN[1]])

    def _widget_box(self, x, y, w, h):
        ax = self._box(x, y, w, h)
        ax.set_frame_on(False)
        return ax

    def _lines(self, x, y, **kw):
        with plt.rc_context(plot_font()):  # the font is read when a text is made, also one made after the window was built
            return self.fig.text(x / self.SIZE_IN[0], y / self.SIZE_IN[1], "", va="top", **{"linespacing": 1.0, **kw})

    def _text(self, x, y, text, **kw):
        with plt.rc_context(plot_font()):
            return self.fig.text(x / self.SIZE_IN[0], y / self.SIZE_IN[1], text, va="center", **kw)

    def _button(self, box, label, action):
        button = Button(self._box(*box), label, color=PALETTE["surface"], hovercolor=PALETTE["line"])
        button.on_clicked(lambda _: action())
        return button

    def _build(self):
        self.fig = plt.figure(figsize=self.SIZE_IN, dpi=self.DPI)
        radio = {"radio_props": {"s": 220}, "activecolor": PALETTE["primary"]}
        self.version_radio = RadioButtons(self._widget_box(*self.RADIO_BOX), list(VERSION_LABELS.values()), **radio)
        self.version_radio.on_clicked(self._choose_version)
        self.redraw_check = CheckButtons(self._widget_box(*self.REDRAW_BOX), [REDRAW_LABEL], frame_props={"s": 220},
                                         check_props={"s": 140, "facecolor": PALETTE["primary"]})
        self.redraw_check.on_clicked(self._redraw_clicked)
        self.start_button = self._button(self.START_BOX, "Start", self.start)
        self.stop_button = self._button(self.STOP_BOX, "Stop", self.stop)
        self.hold_button = self._button(self.HOLD_BOX, HOLD_LABEL, self.toggle_hold)
        self.short_button = self._button(self.SHORT_BOX, "Short run", self.short_run)
        self.defaults_button = self._button(self.DEFAULTS_BOX, "Back to defaults", self.back_to_defaults)
        self.notes_button = self._button(self.TAB_BOXES["notes"], IDLE_TAB, lambda: self.toggle_to("notes"))
        self.compare_button = self._button(self.TAB_BOXES["compare"], "Compare runs", lambda: self.toggle_to("compare"))
        self.page_button = self._button(self.TAB_BOXES["settings"], "Settings", lambda: self.toggle_to("settings"))
        self.counter_text = self._lines(0.2, self.COUNTER_TOP)  # the version's status and the counts of rows and frames
        self.simulator_text = self._lines(0.2, self.COUNTER_TOP, color=PALETTE["muted"])  # what only the simulator knows
        self.counter_rest_text = self._lines(0.2, self.COUNTER_TOP)  # repeats and gaps
        self.note_text = self._lines(self.TEXT_LEFT, self.NOTE_TOP, color=PALETTE["accent"])  # "Not the defaults ...", or empty
        self.hold_text = self._lines(self.TEXT_LEFT, self.NOTE_TOP, color=PALETTE["accent"])  # what is held, or empty
        self.notes_text = self._lines(self.TEXT_LEFT, self.NOTES_TOP)  # the question; the answer and the checks
        self.error_type_text = self._lines(self.TEXT_LEFT, self.NOTES_TOP - 0.75, fontsize=13, color=PALETTE["muted"])  # the program's name for the error
        self.ax = self._box(*self.FORCE_BOX)
        self.truth_line, self.marks = plot_artists(self.ax, self.values["window"], peak_force_n(self.values))
        self.timeline_ax = self._box(*self.TIMELINE_BOX)
        self.timeline_title = self._text(self.TEXT_LEFT, self.TIMELINE_TITLE_Y, BARS_TITLE, fontweight="bold")
        self.strip_label = self._text(self.TEXT_LEFT, self.STRIP_LABEL_Y, timeline.FRAME_STRIP_LABEL)
        self.caption_title = self._text(self.TEXT_LEFT, self.CAPTION_TITLE_Y, CAPTION_TITLE, fontweight="bold")
        self.caption_text = self._lines(self.TEXT_LEFT, self.CAPTION_TOP)
        self.run_artists = [self.ax, self.timeline_ax, self.timeline_title, self.strip_label, self.caption_title,
                            self.caption_text, self.note_text, self.hold_text]
        self.notes_artists = [self.notes_text, self.error_type_text]
        self._build_compare_page()
        self._build_settings_page()
        self.show_idle()
        self.refresh_note()
        self.show_page("notes")

    # -- the page of the shares and the table

    def _build_compare_page(self):
        self.bars_ax = self._box(*self.BARS_BOX)
        self.bars_title = self._text(self.TEXT_LEFT, 6.65, BARS_TITLE, fontweight="bold")
        self.bars_note = self._text(self.TEXT_LEFT, 6.2, BARS_NOTE)
        self.table_title = self._text(0.2, 3.7, TABLE_TITLE, fontweight="bold")
        self.table_note = self._lines(0.2, 3.5)
        self.table_note.set_text(textwrap.fill(TABLE_NOTE, TABLE_NOTE_CHARS))
        self.compare_artists = [self.bars_ax, self.bars_title, self.bars_note, self.table_title, self.table_note]

    def show_shares(self, shares, threads):
        """The bars: one 100 % bar for each thread, a part for each kind of wait in the colour of the timeline, and the rest
        (no wait was recorded) empty. A part of SHARE_LABEL_MIN or more of the run gets its percent. The key says which colour is
        which wait."""
        ax = self.bars_ax
        with plt.rc_context(plot_font()):
            ax.clear()
            self._draw_shares(ax, shares, threads)
            self._show_bars_key(shares, threads)

    def _show_bars_key(self, shares, threads):
        """The colour key of the page: the kinds of wait that a bar has a part of, and the empty box. Made again only when they change."""
        kinds = tuple(k for k in timeline.KINDS if any(shares[t].get(k, 0.0) >= SHARE_KEY_MIN for t in threads))
        empty = any(shares[t][teach.WORKING] >= SHARE_KEY_MIN for t in threads)
        if (kinds, empty) == self._compare_key_for:
            return
        for artist in self.compare_key_artists:
            artist.remove()
        self.compare_key_artists = timeline.add_key(self.fig, kinds, with_tick=False, bottom=self.KEY_BOTTOM,
                                                    empty=teach.WORKING if empty else None)
        self._compare_key_for = (kinds, empty)
        for artist in self.compare_key_artists:
            artist.set_visible(self.page == "compare")

    def _draw_shares(self, ax, shares, threads):
        ax.tick_params(axis="both", labelsize=20)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        rows = len(threads)
        ax.set_yticks(range(rows), [timeline.ROW_LABELS[t] for t in threads])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:g} %")
        for y, thread in enumerate(threads):
            left = 0.0
            for kind in [k for k in timeline.KINDS if k in shares[thread]] + [teach.WORKING]:
                share = shares[thread][kind]
                width = 100 * share
                color = timeline.KINDS[kind] if kind in timeline.KINDS else PALETTE["paper"]
                ax.barh(y, width, left=left, height=0.8, color=color, edgecolor=PALETTE["ink"] if kind == teach.WORKING else "none",
                        linewidth=1)
                if share >= SHARE_LABEL_MIN:
                    self._label_part(ax, left, width, y, PALETTE["paper"] if kind in ("moving the stage", "redrawing the plot")
                                     else PALETTE["ink"])
                left += width

    def _label_part(self, ax, left, width, y, color):
        """The percent of one part of a bar, in the middle of it: "5 %", or "5" when the part is too narrow for the sign (the
        axis has the sign)."""
        label = ax.text(left + width / 2, y, f"{width:.0f} %", ha="center", va="center", fontsize=20, color=color)
        part_px = ax.transData.transform((width, 0))[0] - ax.transData.transform((0, 0))[0]
        if label.get_window_extent(self.fig.canvas.get_renderer()).width > part_px:
            label.set_text(f"{width:.0f}")

    def show_table(self):
        """"Runs so far": the newest run on top; as many rows as fit."""
        for artist in self.table_artists:
            artist.remove()
        y, rows = 2.85, teach.runs_table(self.runs)
        head = list(teach.RUNS_HEADER)
        self.table_artists = [self._lines(x, y, fontweight="bold", linespacing=1.0) for x in TABLE_COLUMNS]
        for artist, text in zip(self.table_artists, head):
            artist.set_text(text)
        y -= 0.35
        used = 0
        for row in rows:
            cells = [textwrap.fill(c, SETTINGS_CHARS) if i == 1 else c for i, c in enumerate(row)]
            n = max(c.count("\n") + 1 for c in cells)
            if used + n > TABLE_LINES:
                break
            self.table_artists += [self._lines(x, y, linespacing=1.0) for x in TABLE_COLUMNS]
            for artist, text in zip(self.table_artists[-len(TABLE_COLUMNS):], cells):
                artist.set_text(text)
            y -= 0.28 * n
            used += n
        for artist in self.table_artists:
            artist.set_visible(self.page == "compare")

    # -- the settings page

    def _build_settings_page(self):
        x, page = self.SETTINGS_X, self.settings_artists
        y = self.SETTINGS_TOP
        page.append(self._text(x["label"], y, "Settings", fontweight="bold"))
        self.boxes, self.checks = {}, {}
        for key in ("rate", "depth", "speed", "hold", "cycles"):
            y -= self.SETTINGS_PITCH
            self._number_row(settings.BY_KEY[key], y)
        y -= self.SETTINGS_PITCH
        page.append(self._text(x["label"], y, "Log to"))
        self._check_row(settings.BY_KEY["csv"], y, 8.5, 2.3)
        self._check_row(settings.BY_KEY["database"], y, 10.75, 2.55)
        y -= self.SETTINGS_PITCH
        self._number_row(settings.BY_KEY["window"], y)
        y -= self.SETTINGS_PITCH
        page.append(self._text(x["label"], y, "Simulated rig (not in the framework)", fontweight="bold"))
        y -= self.SETTINGS_PITCH
        self._number_row(settings.BY_KEY["fps"], y)
        for key in ("busy", "slow_disk", "sensor_time"):
            y -= self.SETTINGS_PITCH
            self._check_row(settings.BY_KEY[key], y, x["label"] - 0.22, 4.5)  # its tick box lands on the label column
        y -= self.SETTINGS_PITCH
        self.message_text = self._text(x["label"], y, "", color=PALETTE["accent"])
        page.append(self.message_text)
        for artist in page:
            artist.set_visible(False)
        self._show_widgets(False)

    def _number_row(self, control, y):
        x = self.SETTINGS_X
        label = self._text(x["label"], y, control.label)
        hint = self._text(x["hint"], y, f"{control.lo:g}–{control.hi:g}", color=PALETTE["muted"])
        box = TextBox(self._box(x["box"], y - 0.19, 1.15, 0.38), "", initial=format(control.default, "g"),
                      color=PALETTE["surface"], hovercolor=PALETTE["line"])
        box.on_submit(partial(self.submit, control.key))
        self.boxes[control.key] = box
        self.settings_artists += [label, hint]
        self.settings_widgets.append(box)

    def _check_row(self, control, y, x, width):
        check = CheckButtons(self._widget_box(x, y - 0.19, width, 0.38), [control.label], [control.default],
                             frame_props={"s": 220}, check_props={"s": 140, "facecolor": PALETTE["primary"]})
        check.on_clicked(partial(self._checked, control.key))
        self.checks[control.key] = check
        self.settings_widgets.append(check)

    def show_page(self, page):
        """The right part of the window shows one page: "run", "notes", "compare" or "settings"."""
        self.page = page
        pages = {"run": self.run_artists + self.key_artists, "notes": self.notes_artists,
                 "compare": self.compare_artists + self.table_artists + self.compare_key_artists, "settings": self.settings_artists}
        for name, artists in pages.items():
            for artist in artists:
                artist.set_visible(name == page)
        for artist in (self.counter_text, self.simulator_text, self.counter_rest_text):
            artist.set_visible(page != "compare")  # the table has the width of the window, under the counters' place
        self._show_widgets(page == "settings")
        for name, button in (("notes", self.notes_button), ("compare", self.compare_button), ("settings", self.page_button)):
            button.label.set_text(RUN_TAB if page == name else self._tab_label(name))
        if page == "compare":
            self.show_compare()

    def _tab_label(self, name):
        return {"notes": self._notes_tab(), "compare": "Compare runs", "settings": "Settings"}[name]

    def _notes_tab(self):
        """The first tab asks the question before a run, and holds the answer after it."""
        return IDLE_TAB if self._asking() else DONE_TAB

    def _show_widgets(self, shown):
        for widget in self.settings_widgets:
            widget.ax.set_visible(shown)
            widget.active = shown  # not set_active: for a tick box that toggles it

    def toggle_to(self, page):
        """The button of a page: its page, or the run when that page is already there."""
        self.show_page("run" if self.page == page else page)

    def toggle_page(self):
        self.toggle_to("settings")

    def submit(self, key, text):
        """A number typed into the box of `key`: take it, or say in words why not and put the old number back."""
        try:
            self.values[key] = settings.parse(key, text)
            self.message_text.set_text("")
        except ValueError as exc:
            self.message_text.set_text(str(exc))
        self.show_value(key)
        self.refresh()

    def _checked(self, key, _label):
        self.values[key] = self.checks[key].get_status()[0]
        self.refresh()

    def show_value(self, key):
        """Put the current value of `key` in its box or tick box, without telling anyone."""
        if key in self.boxes:
            box, box.eventson = self.boxes[key], False
            box.set_val(format(self.values[key], "g"))
            box.eventson = True
        elif self.checks[key].get_status()[0] != self.values[key]:
            check, check.eventson = self.checks[key], False
            check.set_active(0)
            check.eventson = True

    def apply(self, new):
        """Set the settings in `new` and show them."""
        self.values.update(new)
        for key in new:
            self.show_value(key)
        self.message_text.set_text("")
        self.refresh()

    def short_run(self):
        self.apply(settings.SHORT_RUN)

    def back_to_defaults(self):
        self.apply(settings.defaults())

    def refresh(self):
        """What follows from a change of the settings: the note, and the plot of a window with no run yet."""
        self.refresh_note()
        if self.session is None:
            self.ax.set_xlim(0, self.values["window"])
            self.ax.set_ylim(0, peak_force_n(self.values))
            self.show_idle()
        elif self.session.done:  # a finished run: the same run, in the new plot window
            self.update_artists()
            self._show_notes()
        else:
            self._show_notes()
        self.fig.canvas.draw_idle()

    def refresh_note(self):
        differs = bool(settings.differing(self.values))
        self.note_text.set_text(textwrap.fill(settings.MESSAGE_NOT_DEFAULT, TEXT_CHARS) if differs and not self.held else "")

    def _redraw_clicked(self, _label):
        self.redraw_in_loop = self.redraw_check.get_status()[0]
        if self.session is None:  # the key follows the switch until a run has been made
            self.show_key()

    def _choose_version(self, label):
        self.version = label[:2]
        switch = self.redraw_check.labels[0]  # the switch only works for the versions that take the per-tick hook
        available = self.version in loops.THREADED
        switch.set_text(REDRAW_LABEL if available else REDRAW_LABEL_UNAVAILABLE)
        switch.set_color(PALETTE["ink"] if available else PALETTE["muted"])
        if self.session is None:  # nothing run yet: show the rows and the key of this version
            self.show_idle()
        if not self._running():  # the question of the version to guess before Start
            self._show_notes()
            self.show_page("notes")
        else:
            self._show_notes()

    # -- what the window says

    def _running(self):
        return self.session is not None and not self.session.done

    def _has_run(self):
        """Is there a finished run of the chosen version, so that the notes hold its answer and not a question?"""
        return self.session is not None and self.session.done and self.session.version == self.version

    def _asking(self):
        """Do the notes ask the question of the chosen version: no run of it has started, and no other run is going?"""
        return not self._running() and not self._has_run()

    def show_idle(self):
        self.counter_text.set_text("Choose a version and press Start.\n"
                                   f"A run takes about {expected_duration_s(self.version, self.values):.0f} s.")
        self.simulator_text.set_text("")
        self.counter_rest_text.set_text("")
        self.place_counters()
        self._show_notes()
        self.error_type_text.set_text("")
        self.show_rows()

    def show_rows(self):
        """The panel of the chosen version, with its rows empty; a run puts its waits on them."""
        threads = timeline.THREADS[self.version]
        self.panel = timeline.Timeline(self.timeline_ax, threads, frames=True)
        self.panel.update({t: [] for t in threads}, (0.0, timeline.WINDOW_S[self.version]))
        self.show_key()

    def show_key(self):
        """The colour key of the version the panel shows: the kinds of wait that its rows can contain, and the marks of the
        frame strip that its frames can have."""
        for artist in self.key_artists:
            artist.remove()
        s = self.session
        version, redraw = (s.version, s.redraw_in_loop) if s else (self.version, self.redraw_in_loop)
        with plt.rc_context(plot_font()):
            self.key_artists = timeline.add_key(self.fig, timeline.kinds_for(version, redraw), with_tick=False,
                                                frames=teach.FATES_OF[version], bottom=self.KEY_BOTTOM)
        for artist in self.key_artists:
            artist.set_visible(self.page == "run")

    def place_counters(self):
        """The three blocks of the counters one under the other, each as high as its text is."""
        renderer, y = self.fig.canvas.get_renderer(), self.COUNTER_TOP
        for artist in (self.counter_text, self.simulator_text, self.counter_rest_text):
            artist.set_y(y / self.SIZE_IN[1])
            y -= (artist.get_window_extent(renderer).height / self.fig.dpi + COUNTER_GAP_IN) if artist.get_text() else 0.0

    def status(self):
        s = self.session
        if s.error:
            return "Stopped: something went wrong." if s.done else f"Something went wrong, stopping {s.version} ..."
        if not s.done:
            if s._stop.is_set():
                return f"Still stopping {s.version}: wait." if self._start_refused else f"Stopping {s.version} ..."
            if self._start_refused:
                return f"{s.version} is running: press Stop first."
            return f"Running {s.version}. Redraw in the loop: {'on' if s.redraw_in_loop else 'off'}"
        return f"Stopped. {s.version} was cut short." if s.cut else f"{s.version} has finished."

    def show_counters(self, c):
        top, muted, rest = counter_blocks(c, self.session.version)
        self.counter_text.set_text(self.status() + "\n" + top)
        self.simulator_text.set_text(muted)
        self.counter_rest_text.set_text(rest)
        self.place_counters()

    # -- the notes page

    def _notes(self):
        """The text of the notes page: the question before a run of the chosen version, a word while it goes, and after a
        complete run the answer, what happened and the checks on the file."""
        s = self.session
        if self._asking():
            return self._question()
        if s.error:
            return "Something went wrong inside the program,\nso the run was stopped."
        if not s.done:
            return self._wrap(timeline.what_it_shows(s.version)) + "\n\nThe answer and the checks appear when the run has finished."
        if s.cut:
            return "Run stopped before the end: no checks."
        return self._answer(s)

    def _wrap(self, text):
        return textwrap.fill(text, TEXT_CHARS)

    def _question(self):
        """Guess first: the question of the chosen version with its four choices, and how to answer (spec 3.4)."""
        v, values = self.version, self.values
        lines = teach.question_lines(v, expected_duration_s(v, values) * values["fps"], 2 * (values["cycles"] + 1))
        choices = [textwrap.fill(line, TEXT_CHARS, subsequent_indent="    ") for line in lines[1:]]
        return "\n".join([self._wrap(lines[0])] + choices + ["", self._wrap(teach.INSTRUCTION), "",
                                                              self._wrap(timeline.what_it_shows(v))])

    def _answer(self, s):
        """After a complete run: "This run: ...", "What happened in this run", then "Checks on the file:" (the causes above, the
        effects below). When the page would have too many lines, the open question of V2 gives way first, then the list of what
        happened from its last line up: the causes of this run come before the open question."""
        answer, *open_question = (self._wrap(p) for p in teach.answer_lines(s.version, s.numbers(s.snapshot())))
        checks = "Checks on the file:\n" + "\n".join(check_sentences(s))
        told = [self._wrap(line) for line in teach.what_happened(s.events)]

        def page():
            happened = ["\n".join(["What happened in this run"] + told)] if told else []
            return "\n\n".join(["\n".join([answer] + open_question)] + happened + [checks])

        while page().count("\n") + 1 > NOTES_LINES and (open_question or told):
            if open_question:
                open_question.clear()
            else:
                told.pop()
        return page()

    def _show_notes(self):
        self.notes_text.set_text(self._notes())
        self.notes_button.label.set_text(RUN_TAB if self.page == "notes" else self._notes_tab())

    def _show_final(self):
        """The notes after the run, and the table of the runs; an error gets the words for it and the program's name, small."""
        s = self.session
        self._show_notes()
        self.error_type_text.set_text(f"({type(s.error).__name__})" if s.error else "")
        if s.sheet and not (s.cut or s.error):
            self.runs.append(teach.RunRecord(s.version, teach.settings_text(s.values), s.numbers(s.snapshot())))
            if self.page == "compare":
                self.show_table()

    # -- the run page

    def update_artists(self):
        """Put the session's snapshot on the counters, and, unless the picture is held, on the plot, the timeline, the caption
        and the bars. Main thread only. Needs a session."""
        snap = self.session.snapshot()
        self.show_counters(self.session.counters(snap))
        self._caption_now = self.caption.update(snap.now, self.session.new_events(snap))  # the causes go on while held
        if not self.held:
            self.draw_picture(snap)

    def draw_picture(self, snap):
        """What "Hold the picture" holds: the force plot, the timeline with its frame row, the bars and the caption."""
        s = self.session
        lo, hi = s.plot_window(snap, self.values["window"])
        self.ax.set_xlim(lo, hi)
        show_snapshot(self.truth_line, self.marks, snap, lo)
        self.panel.update(*s.trace_view(snap), frames=s.frame_marks(snap))
        self.caption_text.set_text(textwrap.fill(self._caption_now, TEXT_CHARS))
        if self.page == "compare":
            self.show_compare()

    def show_compare(self):
        """The page "Compare runs": the bars of the run on show, and the table."""
        s = self.session
        if s is not None and not self.held:
            self.show_shares(s.shares(s.snapshot()), timeline.THREADS[s.version])
        self.show_table()

    def toggle_hold(self):
        """"Hold the picture" / "Let it run": the run and the counters go on, the picture is not redrawn."""
        self.held = not self.held
        self.hold_button.label.set_text(RELEASE_LABEL if self.held else HOLD_LABEL)
        self.hold_text.set_text(textwrap.fill(HOLD_NOTE, TEXT_CHARS) if self.held else "")
        self.refresh_note()
        if not self.held and self.session is not None:
            self.draw_picture(self.session.snapshot())
        self.fig.canvas.draw_idle()

    def render_in_loop(self):
        """The redraw inside the logging loop (runs in the loop's thread, once per tick); not the window's figure."""
        self.render.draw(self.session.snapshot(), self.pixels)

    # -- the buttons

    def start(self):
        if self.session and not self.session.done:  # a run is going or stopping: say why Start does nothing
            self._start_refused = True
            return
        self._start_refused = False
        on_tick = self.render_in_loop if self.redraw_in_loop else None  # read here, fixed for the run
        self.read_pixels()
        self.render.prepare(self.values["window"], peak_force_n(self.values))
        self.session = Session(self.version, self.values, self.out, self.seed, on_tick)
        self.caption, self._caption_now = teach.CaptionLine(), ""
        self.caption_text.set_text("")
        self.ax.set_ylim(0, peak_force_n(self.values))
        self.show_page("run")
        self.show_rows()  # the rows and the key of this run's version (the version may have changed since the last run)
        self._show_notes()
        self.error_type_text.set_text("")
        self._shown_final = False
        self.session.start()

    def stop(self):
        if self.session:
            self.session.stop()

    def close(self):
        """Window closed: end the run, wait for its threads, stop the timer."""
        self.stop()
        if self.session:
            self.session.join(timeout=self.session.stop_limit_s())
        self.timer.stop()

    # -- the GUI timer (main thread)

    def on_timer(self):
        s = self.session
        if s is None:
            return
        try:
            if not s.done:
                self.update_artists()
            if s.done and not self._shown_final:
                self.update_artists()
                self._show_final()
                self._shown_final = True
        except Exception as exc:  # an error here must stop the run and tell the presenter, not freeze the window
            s.fail(exc)
            self.counter_text.set_text(self.status())
        if s.error and not self._shown_final:
            self._show_final()
            self._shown_final = True
        self.fig.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1, help="noise seed of the simulated rig")
    ap.add_argument("--classic", action="store_true", help="the old matplotlib window, not the page in the browser")
    ap.add_argument("--no-browser", action="store_true", help="start the page's server and print its address only")
    a = ap.parse_args()
    if a.classic and a.no_browser:
        ap.error("--no-browser belongs to the browser interface, not to --classic")
    with tempfile.TemporaryDirectory(prefix="demo_app_") as out:  # the log files of the runs
        if a.classic:
            View(out, a.seed)
            plt.show()
        else:
            import sys
            sys.modules.setdefault("app", sys.modules[__name__])  # run as a script, this file is __main__: webapp's `import app` must not load a second copy
            import webapp
            webapp.main(out, a.seed, open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
