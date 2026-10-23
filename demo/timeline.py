"""Where the time goes: the waits of each loop, measured, as one row of coloured blocks per thread.

  Recorder       the trace= hook of loops.py: collects (thread, kind, start, end) from the running loop; thread-safe.
  Timeline       the artists of one panel, made once; update() puts the newest spans on them. The window (app.py) calls
                 update() at every tick, the slide figures once. With frames=True the panel has a top row for the frames
                 the sensor sent, one mark for each, by what became of it (the frame strip).
  timeline_figure  one slide-sized figure (1920 x 1080 at 144 dpi): title, rows, colour key, one sentence.
Times are seconds on the rig's clock (World.now), the same clock as the first column of the log file.
"""
from __future__ import annotations

import collections
import functools
import itertools
import sys
import textwrap
import threading

import matplotlib
import matplotlib.font_manager as font_manager
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

import freeze_numbers
import loops
from plots import PALETTE

KINDS = {  # kind of wait -> colour; the same in the app and on the slides
    "moving the stage": PALETTE["primary"],
    "sleeping": PALETTE["line"],
    "waiting for the sensor frame": PALETTE["sky"],
    "writing to disk": PALETTE["stone"],
    "redrawing the plot": PALETTE["ink"],
    "reading the load cell": "#D9A77F",  # a light tint of the accent: the reads are short and dense
}
KIND_RGBA = {k: to_rgba(c) for k, c in KINDS.items()}  # as the numbers matplotlib draws with: converted once, not at every tick
TICK_COLOR = PALETTE["accent"]  # the time of a row in the file: the colour of the points on the force plot
THREADS = {  # version -> the threads that exist in it, top row first; the last one writes the file
    "V0": ("script",), "V1": ("loop",),
    **{v: ("coordinator", "sensor", "load cell", "file writer") for v in loops.THREADED},
}
ROW_LABELS = {"script": "the script", "loop": "the loop", "coordinator": "main loop", "sensor": "sensor reader",
              "load cell": "load-cell reader", "file writer": "file writer"}
FRAME_ROW_LABEL = "sensor frames"  # the label of the frame strip; it is not a thread, so it is not in ROW_LABELS
NOT_KNOWN, ONCE, TWICE, LOST = range(4)  # what became of a frame the sensor sent: the code of each mark of the frame strip
FRAME_LEGEND = {ONCE: "in the file", TWICE: "in the file twice", LOST: "lost (in no row of the file)", NOT_KNOWN: "not known yet"}
FRAME_STRIP_LABEL = "Sensor frames (only the simulator knows this row)"
FRAME_MARK_PT = (3.2, 8.0)  # the smallest and the largest mark of the strip, in points across: dense frames get small marks
# The kinds of wait in the rows of each version, as drawn. The load-cell reader of V2 to V4 reads for 2 microseconds and
# sleeps for 1 ms: its row is drawn as the sleep it mostly is, so "reading the load cell" is in V0 and V1 only.
KINDS_OF = {
    "V0": ("moving the stage", "sleeping", "waiting for the sensor frame", "writing to disk", "reading the load cell"),
    "V1": ("sleeping", "waiting for the sensor frame", "writing to disk", "reading the load cell"),
    **{v: ("sleeping", "waiting for the sensor frame", "writing to disk") for v in loops.THREADED},
}
TITLES = {
    "V0": "V0  the script the paper warns against",
    "V1": "V1  one loop, read every 20 ms",
    "V2": "V2  the framework used for the paper",
    "V3": "V3  the fix of pull request #3",
    "V4": "V4  the loop looks twice per frame",
}
WHAT_IT_SHOWS = {  # one plain sentence per version: what its timeline shows (each one is checked against a recorded run)
    "V0": "The script is blocked in the move and in the sleep for nearly all the time, and it writes one row after each step.",
    "V1": "One loop does everything in turn. When a disk write is slow, the loop waits and no row is written until it ends.",
    "V2": "Each device has its own reader, running at the same time. A slow disk write now stops only the file writer. The "
          "main loop wakes up at almost the same moment as the sensor reader, so a row gets the new frame or the old one by chance.",
    "V3": "The same readers. Each row now also has the time the computer read the sensor frame, and the frame number. "
          "A wake-up just before the frame arrives still misses it, and the frame numbers show which ones.",
    "V4": "The main loop wakes up twice per sensor frame, so one wake-up comes after each frame has arrived.",
}


@functools.lru_cache(maxsize=None)
def v4_loss_sentence():
    """What the freeze measured about V4's lost frames, read from NUMBERS.md when a caption is made, not when this module is
    imported: a missing NUMBERS.md stops the caption with a message that names the file, not the import."""
    return freeze_numbers.loss_sentence(freeze_numbers.frozen_losses(freeze_numbers.FREEZE / "NUMBERS.md"))


def what_it_shows(version):
    """The sentence under the timeline of `version`: V4's ends with what the freeze measured."""
    return f"{WHAT_IT_SHOWS[version]} {v4_loss_sentence()}" if version == "V4" else WHAT_IT_SHOWS[version]


WINDOW_S = {"V0": 8.0, "V1": 2.0, "V2": 1.5, "V3": 1.5, "V4": 1.5}  # seconds shown: long enough to see one cycle of the loop
TICK_LABEL = "the time in each row"  # the tick is the time stamp of a row, not the moment it is written
LABEL_PT = 20  # smallest text on screen and on the slides
MIN_BLOCK_PX = 1.5  # a block is drawn at least this wide, so a short wait still shows
FONT = "IBM Plex Sans"
KEY_LEFT_IN, KEY_BOTTOM_IN, KEY_COLUMNS = 0.35, 0.3, 3  # the colour key: inches from the left and bottom edge to the middle of its last row


class Recorder:
    """The trace of one run. Called from any thread of the loop; keeps the newest `capacity` spans of each thread."""

    def __init__(self, threads, capacity=4096):
        self._spans = {t: collections.deque(maxlen=capacity) for t in threads}
        self._lock = threading.Lock()

    def __call__(self, thread, kind, t0, t1):
        with self._lock:
            self._spans[thread].append((kind, t0, t1))

    def spans(self, since=float("-inf")):
        """{thread: [(kind, t0, t1), ...]} oldest first, of the spans that end after `since`. A copy."""
        with self._lock:
            return {t: list(itertools.takewhile(lambda s: s[2] > since, reversed(d)))[::-1] for t, d in self._spans.items()}


def _blocks(spans, window, min_width):
    """[(kind, start, width)] of the spans inside the window, oldest first. A block is drawn at least min_width wide.

    A run of blocks narrower than that which touch one another (a load-cell read and a 1 ms sleep, a thousand times a
    second) is one bar, as the eye sees it. Its kind is the one that takes most of the run's time: the row shows what
    the thread mostly does, not a read that fills 0.3 % of it."""
    lo, hi = window
    runs, run = [], None  # run: [start, end, made of narrow blocks only, {kind: seconds}]
    for kind, t0, t1 in spans:  # 3000 spans of the load-cell reader at every tick: plain comparisons, no max() and min()
        if t1 < lo or t0 > hi:
            continue
        if t0 < lo:
            t0 = lo
        if t1 > hi:
            t1 = hi
        if t1 - t0 >= min_width:  # a wait the eye can see by itself
            run = [t0, t1, False, {kind: t1 - t0}]
        elif run is not None and run[2] and t0 - run[1] < min_width:  # narrow, and touching the narrow run before it
            run[1] = max(t1, t0 + min_width)
            run[3][kind] = run[3].get(kind, 0.0) + t1 - t0
            continue
        else:
            run = [t0, t0 + min_width, True, {kind: t1 - t0}]
        runs.append(run)
    return [(max(r[3], key=r[3].get), r[0], r[1] - r[0]) for r in runs]


def kinds_for(version, redraw=False):
    """The kinds of wait that the rows of `version` can contain, in the order of the key. `redraw`: the plot is drawn inside the loop."""
    shown = KINDS_OF[version] + (("redrawing the plot",) if redraw and version in loops.THREADED else ())
    return tuple(k for k in KINDS if k in shown)


def seconds_axis(ax, nbins=4):
    """Time axis labelled "0.5 s", "1 s", ...: the unit sits on the numbers, so the axis needs no label line."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g} s"))


def last_seconds(end, length):
    """The `length` seconds that end at `end`; in the first seconds the window starts at 0."""
    start = max(end - length, 0.0)
    return start, start + length


def window_ending_at(version, end):
    """The WINDOW_S seconds of the version that end at `end`."""
    return last_seconds(end, WINDOW_S[version])


def _frame_mark_style(code):
    """The look of the mark of a frame fate: filled = in the file, ringed = twice, hollow = lost, grey = not known yet.
    Shape and fill tell them apart without the colours."""
    ink, accent = PALETTE["ink"], PALETTE["accent"]
    return {ONCE: dict(facecolors=accent, edgecolors=accent, linewidths=0.0),
            TWICE: dict(facecolors=accent, edgecolors=ink, linewidths=2.0),
            LOST: dict(facecolors="none", edgecolors=ink, linewidths=0.9),
            NOT_KNOWN: dict(facecolors=PALETTE["line"], edgecolors=PALETTE["stone"], linewidths=0.8)}[code]


class Timeline:
    """One panel: a row of coloured blocks for each thread, and the ticks of the rows in the file on the last row.

    The artists are made once, for the threads given; update() only gives them new shapes. Redrawing the panel from
    nothing at every tick of the window took longer than the force plot, the counters and the text together.
    """

    def __init__(self, ax, threads, font_size=LABEL_PT, frames=False):
        """threads: the rows, top first; the last one writes the file (its rows get the ticks). frames: one more row on top,
        the frame strip."""
        ax.clear()
        self.ax, self.threads, self.frames = ax, tuple(threads), frames
        n = len(self.threads)
        rows = n + frames
        for y in range(rows):
            ax.axhline(y + 0.5, color=PALETTE["line"], lw=1, zorder=0)  # the row's baseline: empty = working, not waiting
        self.blocks = PolyCollection([], edgecolors=PALETTE["paper"], linewidths=1, zorder=2)
        self.ticks = LineCollection([], colors=TICK_COLOR, linewidths=2, zorder=3)
        ax.add_collection(self.blocks)
        ax.add_collection(self.ticks)
        self.marks = {code: ax.scatter([], [], zorder=4, **_frame_mark_style(code)) for code in FRAME_LEGEND} if frames else {}
        ax.set_ylim(0, max(rows, 1))
        names = ([FRAME_ROW_LABEL] if frames else []) + [ROW_LABELS[t] for t in self.threads]
        ax.set_yticks([rows - 1 - i + 0.5 for i in range(rows)], names)
        ax.tick_params(axis="y", length=0, labelsize=font_size)
        ax.tick_params(axis="x", labelsize=font_size)
        seconds_axis(ax)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    def update(self, spans, window, ticks=(), frames=None):
        """Show `spans` ({thread: [(kind, t0, t1)]}) in `window` (t_lo, t_hi seconds); ticks: seconds of the rows written.
        frames: (seconds each frame reached the computer, its fate code) as arrays, for a panel made with frames=True."""
        lo, hi = window
        n = len(self.threads)
        min_width = (hi - lo) / self.ax.get_window_extent().width * MIN_BLOCK_PX
        blocks = [(kind, start, width, n - 1 - i) for i, thread in enumerate(self.threads)
                  for kind, start, width in _blocks(spans[thread], window, min_width)]
        x, w, y = np.array([b[1:] for b in blocks], float).reshape(-1, 3).T  # start, width, row of each block
        corners = ((x, y + 0.1), (x + w, y + 0.1), (x + w, y + 0.9), (x, y + 0.9))  # a row: 0.8 high, 0.1 above and below
        self.blocks.set_verts(np.stack([np.stack(c, axis=-1) for c in corners], axis=1))  # (blocks, 4 corners, x y)
        self.blocks.set_facecolor([KIND_RGBA[b[0]] for b in blocks])
        self.ticks.set_segments([((t, 0.05), (t, 0.95)) for t in ticks if lo <= t <= hi])
        if self.frames:
            self._show_frames(*(frames if frames is not None else ([], [])), window)
        self.ax.set_xlim(lo, hi)

    def _show_frames(self, times, fates, window):
        """One mark for each frame in the window, in the strip (the row above the threads)."""
        lo, hi = window
        times, fates = np.asarray(times, float), np.asarray(fates, int)
        inside = (times >= lo) & (times <= hi)
        times, fates = times[inside], fates[inside]
        width_pt = self.ax.get_window_extent().width * 72 / self.ax.figure.dpi
        apart_s = float(np.median(np.diff(times))) if len(times) > 1 else hi - lo  # the frames' period, as the marks show it
        gap_pt = width_pt * apart_s / (hi - lo)  # the space between the middles of two frames, in points
        diameter = float(np.clip(0.8 * gap_pt, *FRAME_MARK_PT))
        y = len(self.threads) + 0.5
        for code, marks in self.marks.items():
            at = times[fates == code]
            marks.set_offsets(np.column_stack([at, np.full(len(at), y)]) if len(at) else np.empty((0, 2)))
            outer = self._outer_pt(code, diameter, gap_pt)
            line = float(_frame_mark_style(code)["linewidths"] if code != TWICE else min(2.0, 0.3 * outer))
            marks.set_linewidths([line])
            marks.set_sizes([(outer - line) ** 2])  # the size is the middle of the edge: the mark, with its edge, is `outer` across

    @staticmethod
    def _outer_pt(code, diameter, gap_pt):
        """Points across, edge included, of the mark of a fate. A frame found in the file twice has the larger mark, but never one
        wider than the space between two frames (unless a plain mark is wider already): the lost frame next to it stays in view."""
        return min(1.5 * diameter, max(gap_pt, diameter)) if code == TWICE else diameter


def add_key(fig, kinds, with_tick=True, font_size=LABEL_PT, frames=(), bottom=KEY_BOTTOM_IN, empty=None):
    """The colour key at the bottom left of `fig`: a swatch and a label for each of `kinds`, then the tick of a row, then the
    marks of the frame strip that `frames` lists (fate codes). `bottom`: inches from the bottom edge to the middle of the last
    row. `empty`: the label of one more swatch, an empty box with an outline (the part of a bar that no wait fills), after the
    kinds. Returns the artists it made, so that the window can take them away and make a key for another version.

    Three columns that fill downward. Plain rectangles and texts, laid out once: a legend lays itself out again at every
    draw, and that was 4 ms of the 20 ms the window's tick takes.
    The window leaves the tick out: its force plot already says that this colour is a point in the file."""
    entries = [(Rectangle((0, 0), 1, 1, facecolor=KINDS[k], edgecolor="none"), k) for k in kinds]
    if empty:
        entries.append((Rectangle((0, 0), 1, 1, facecolor=PALETTE["paper"], edgecolor=PALETTE["ink"], linewidth=1), empty))
    if with_tick:
        entries.append((Line2D([], [], color=TICK_COLOR, lw=2), TICK_LABEL))
    for code in frames:
        style = _frame_mark_style(code)
        entries.append((Line2D([], [], marker="o", ls="", markersize=0.4 * font_size * (1.5 if code == TWICE else 1.0),
                               markerfacecolor=style["facecolors"], markeredgecolor=style["edgecolors"],
                               markeredgewidth=style["linewidths"]), FRAME_LEGEND[code]))
    em, renderer = font_size / 72, fig.canvas.get_renderer()  # inches per em
    rows = -(-len(entries) // KEY_COLUMNS)
    x, made = KEY_LEFT_IN, []
    for column in range(KEY_COLUMNS):
        widest = 0.0
        for r, (swatch, label) in enumerate(entries[column * rows:(column + 1) * rows]):
            y = bottom + (rows - 1 - r) * 1.2 * em
            if isinstance(swatch, Rectangle):
                swatch.set(xy=(x, y - 0.35 * em), width=1.4 * em, height=0.7 * em, transform=fig.dpi_scale_trans)
            elif swatch.get_marker() != "None":  # a mark of the frame strip
                swatch.set_data([x + 0.7 * em], [y])
                swatch.set_transform(fig.dpi_scale_trans)
            else:  # the tick: a short vertical line in the middle of the swatch's width
                swatch.set_data([x + 0.7 * em] * 2, [y - 0.45 * em, y + 0.45 * em])
                swatch.set_transform(fig.dpi_scale_trans)
            fig.add_artist(swatch)
            text = fig.text(x + 2.0 * em, y, label, va="center", fontsize=font_size, color=PALETTE["ink"],
                            transform=fig.dpi_scale_trans)
            made += [swatch, text]
            widest = max(widest, 2.0 * em + text.get_window_extent(renderer).width / fig.dpi)
        x += widest + 1.2 * em
    return made


@functools.lru_cache(maxsize=None)
def font_params():
    """rc settings for the font: IBM Plex Sans when it is installed, else matplotlib's own, with a message (once)."""
    try:
        font_manager.findfont(FONT, fallback_to_default=False)
    except ValueError:
        print(f"{FONT} is not installed: using {matplotlib.rcParams['font.sans-serif'][0]}.", file=sys.stderr)
        return {}
    return {"font.family": FONT}


def timeline_figure(version, spans, window, ticks, font_size=LABEL_PT):
    """The slide figure of one version: 13.33 x 7.5 in, 1920 x 1080 at 144 dpi. Text is font_size pt or more.

    Title, one row per thread, the colour key and the sentence of what the timeline shows."""
    width, height = 40 / 3, 7.5
    rows_h = 2.2 if len(THREADS[version]) == 1 else 0.75 * len(THREADS[version])
    top = 6.4  # the rows start here (inches from the bottom)
    with matplotlib.rc_context({"font.size": font_size, **font_params()}):
        fig = Figure(figsize=(width, height), dpi=144)
        FigureCanvasAgg(fig)  # the key measures its texts on this canvas
        ax = fig.add_axes([2.6 / width, (top - rows_h) / height, 10.4 / width, rows_h / height])
        fig.text(0.2 / width, 7.0 / height, TITLES[version], va="center", fontsize=font_size + 4, fontweight="bold")
        Timeline(ax, spans, font_size).update(spans, window, ticks)
        fig.text(0.2 / width, (top - rows_h - 0.65) / height, textwrap.fill(what_it_shows(version), 82), va="top")
        add_key(fig, kinds_for(version), font_size=font_size)
    return fig
