"""The teaching layer of the window, as functions of data: what each loop lost, and the plain sentence for each cause.

No window in here (design/APP_TEACHING_SPEC.md, sections 3 and 4). The window (app.py) gives these functions what the run
recorded and draws what they return.

  frame_fates      what became of each frame the sensor sent: in the file once, twice, lost, or not known yet
  CAPTIONS         the sentence of each cause (spec 4.1), and Detector, which finds the causes in a run as it grows
  CaptionLine      which sentence the window shows now: each stays a while, repeats are counted
  QUESTIONS        the question before Start and the answer after the run (spec 4.2)
  Tracer, wait_shares   where each thread spent the run's time
  RunRecord, runs_table the table "Runs so far"
All times are seconds on the rig's clock, the first column of the log file.
"""
from __future__ import annotations

import bisect
import collections
import math
import re
import threading
from dataclasses import dataclass

import numpy as np

import loops
import settings
import timeline
from timeline import LOST, NOT_KNOWN, ONCE, TWICE  # the fate codes; the strip draws one mark for each

# --- what became of each frame ------------------------------------------------------------------------------------


def frame_fates(frames, sent, final):
    """The fate of frames 1..sent as an array of NOT_KNOWN / ONCE / TWICE / LOST codes.

    frames: the frame number that each row of the file carries (0: the row came before the first frame).
    A frame that no row has is lost once a later frame is in the file (the loop has read past it), or when the run has ended
    (`final`); until then its fate is not known."""
    frames = np.asarray(frames, int)
    seen = np.bincount(frames[(frames >= 1) & (frames <= sent)], minlength=sent + 1)[1:]
    fates = np.where(seen >= 2, TWICE, np.where(seen == 1, ONCE, NOT_KNOWN))
    newest = frames.max(initial=0)
    gone = final or np.arange(1, sent + 1) < newest
    return np.where((seen == 0) & gone, LOST, fates)


# the fates a version can give a frame, for the key of the strip: a row is written at every tick in V2 and V3, so the same frame
# can be in two rows; V0, V1 and V4 write a row for a new frame only
FATES_OF = {"V0": (ONCE, LOST, NOT_KNOWN), "V1": (ONCE, LOST, NOT_KNOWN), "V2": (ONCE, TWICE, LOST, NOT_KNOWN),
            "V3": (ONCE, TWICE, LOST, NOT_KNOWN), "V4": (ONCE, LOST, NOT_KNOWN)}


def frame_arrivals(sent, period_s, latency_s):
    """Seconds at which frames 1..sent reached the computer: the sensor took frame k at k periods, and it arrives latency later."""
    return np.arange(1, sent + 1) * period_s + latency_s


def loss_pct(fates):
    """% of the frames whose fate is known that are lost; None before any frame is known."""
    known = np.count_nonzero(fates != NOT_KNOWN)
    return float(100 * np.count_nonzero(fates == LOST) / known) if known else None


# --- the captions -------------------------------------------------------------------------------------------------

DISK_STALL_S = 0.015  # C4: a write to disk longer than this is told about
REDRAW_S = 0.005  # C13: so is a redraw longer than this
GAP_LIMIT_S = 0.030  # C10b, C14: a gap between rows longer than this
NO_LOSS_S = 5.0  # C12: told when this long has passed since Start and no frame has been lost
HOLD_S, WORDS_PER_S = 3.0, 3.0  # a caption stays 3 s, or one second per 3 words if it has more than 9
CAPTIONS = {  # id -> sentence; braces are filled from the run. Spec 4.1, with the words the owner fixed after it: a row is one line of the file
    "C1": "The loop waits for the stage to arrive and reads nothing until it does.",
    "C2": "The loop now sleeps for 2 s, and the sensor sends {frames} frames that no one reads.",
    "C3": "This is the only row the file gets from this move.",
    "C4": "Writing to disk took {d} ms, and the loop read nothing in that time.",
    "C5": "Frame {k} reached the computer while the loop was writing to disk, so the loop never read it.",
    "C6": "The loop reads the force, then waits up to {period} ms for the next sensor frame, so the two values in a row are not "
          "from the same moment.",
    "C7": "Frame {next} replaced frame {k} in the register before the loop read it, so frame {k} is lost.",
    "C8": "The loop read frame {k} a second time, so the file has the same sensor value twice, with two different times.",
    "C9": "Your file has no frame numbers, so this loss shows only as a sensor value that repeats.",
    "C10": "Frame {k} is lost, and your file shows it: the frame numbers jump from {before} to {after}.",
    "C10b": "Your file has a {d} ms gap where frame {k} should be.",
    "C11": "The loop missed frame {k} at one tick and caught it at the next, {d} ms later.",
    "C12": "Every frame so far is in the file once.",
    "C13": "Redrawing the plot took {d} ms inside the loop, and the loop read nothing in that time.",
    "C14": "There is no row for {d} ms, because {thread} spent {waits}.",
}


@dataclass(frozen=True)
class Event:
    t: float  # the second of the run in which it happened
    kind: str  # the id in CAPTIONS
    text: str  # the sentence, with the numbers of this event


def _ms(seconds):
    return f"{seconds * 1e3:.0f}"


def thread_words(version):
    """The thread that makes the rows of `version`, as the row of the panel names it, with its article: 'the loop', 'the main loop'."""
    label = timeline.ROW_LABELS[timeline.THREADS[version][0]]
    return label if label.startswith("the ") else f"the {label}"


def _spent(spans, lo, hi):
    """'40 ms writing to disk and 18 ms waiting for the sensor frame': the two waits that took most of lo..hi (C14)."""
    total = collections.Counter()
    for kind, t0, t1 in spans:
        total[kind] += max(min(t1, hi) - max(t0, lo), 0.0)
    top = [(kind, s) for kind, s in total.most_common(2) if s >= 0.0005]
    return " and ".join(f"{_ms(s)} ms {kind}" for kind, s in top)


class Detector:
    """Finds the causes of CAPTIONS in a run as it grows (spec 4.1). Each call to step() gives the events that are new.

    version, the sensor's period and latency and the number of moves of the protocol are those of the run.
    Hand it the rows of the file and the waits of the loop's own thread (V0: the script, V1: the loop, V2-V4: the
    coordinator): rows first, then the waits, so that the waits of a row are never missing. It remembers how far it has read."""

    def __init__(self, version, period_s, latency_s, moves):
        self.version, self.period, self.latency, self.moves = version, period_s, latency_s, moves
        self.cursor = -math.inf  # the end of the newest wait read; ask the recorder for the waits that end after it
        self.rows_read, self.writes = 0, 0
        self._spans, self._ticks = [], []  # the waits of the loop's thread; the times of its ticks (V2-V4)
        self._writes = ([], [])  # when each write to disk of the loop began and ended, in time order (C5)
        self._first_loss_seen = self._first_row_seen = self._no_loss_told = False
        self._any_loss = False  # a frame has been lost since Start: C12 is never told after that
        self._started = False

    def step(self, now, spans, rows_t, frames):
        """The new events up to `now`, oldest first. spans: [(kind, t0, t1)] of the loop's thread that end after self.cursor."""
        events = [] if self._started else self._start(now)
        self._started = True
        for span in spans:
            self.cursor = max(self.cursor, span[2])
            self._spans.append(span)
            if span[0] == "writing to disk":
                self._writes[0].append(span[1])
                self._writes[1].append(span[2])
            events += self._span(*span)
        for i in range(self.rows_read, len(rows_t)):
            events += self._row(i, rows_t, frames)
        self.rows_read = len(rows_t)
        if self.version == "V4" and not self._no_loss_told and not self._any_loss and now >= NO_LOSS_S:
            self._no_loss_told = True
            events.append(Event(now, "C12", CAPTIONS["C12"]))
        return sorted(events, key=lambda e: e.t)

    def _start(self, now):
        """The first move of V0 starts with the run, and nothing tells it."""
        return [Event(now, "C1", CAPTIONS["C1"])] if self.version == "V0" else []

    def _span(self, kind, t0, t1):
        d = t1 - t0
        if self.version == "V0":
            if kind == "moving the stage":
                return [Event(t1, "C2", CAPTIONS["C2"].format(frames=round(loops.V0_SLEEP_S / self.period)))]
            if kind == "writing to disk":
                self.writes += 1
                more = [Event(t1, "C1", CAPTIONS["C1"])] if self.writes < self.moves else []
                return [Event(t1, "C3", CAPTIONS["C3"])] + more
        if self.version == "V1" and kind == "writing to disk" and d > DISK_STALL_S:
            return [Event(t1, "C4", CAPTIONS["C4"].format(d=_ms(d)))]
        if self.version in loops.THREADED:
            if kind == "sleeping":
                self._ticks.append(t1)  # the coordinator wakes up: a tick
            if kind == "redrawing the plot" and d > REDRAW_S:
                return [Event(t1, "C13", CAPTIONS["C13"].format(d=_ms(d)))]
        return []

    def _row(self, i, t, frames):
        """The events of row i, from the frame numbers of the rows i-1 and i."""
        k = int(frames[i])
        if k == 0 or self.version == "V0":  # V0 loses almost every frame, and its causes are C1 to C3
            return []
        events = []
        if self.version == "V1" and not self._first_row_seen:
            events.append(Event(t[i], "C6", CAPTIONS["C6"].format(period=f"{self.period * 1e3:g}")))
        self._first_row_seen = True
        before = int(frames[i - 1]) if i else 0
        if before == 0:
            return events
        lost = list(range(before + 1, k))
        gap = t[i] - t[i - 1]
        if lost:
            self._any_loss = True
            events += self._lost_frames(lost, before, k, t[i], gap)
        elif k == before and self.version == "V2":
            events.append(Event(t[i], "C8", CAPTIONS["C8"].format(k=k)))
        caught = self._caught_late(i, k, t) if self.version == "V4" and not lost else []
        events += caught
        if not lost and not caught and gap > GAP_LIMIT_S and self.version != "V0":  # C14 is the fallback: C11 tells this gap
            events += self._unexplained_gap(t[i - 1], t[i])
        return events

    def _lost_frames(self, lost, before, after, t, gap):
        events = []
        for k in lost:
            if self.version in ("V2", "V3"):
                events.append(Event(t, "C7", CAPTIONS["C7"].format(k=k, next=k + 1)))
            if self.version == "V1" and self._arrived_during_a_write(k):
                events.append(Event(t, "C5", CAPTIONS["C5"].format(k=k)))
            if self.version == "V3":
                events.append(Event(t, "C10", CAPTIONS["C10"].format(k=k, before=before, after=after)))
                if gap > GAP_LIMIT_S:
                    events.append(Event(t, "C10b", CAPTIONS["C10b"].format(d=_ms(gap), k=k)))
        if self.version == "V2" and not self._first_loss_seen:
            events.append(Event(t, "C9", CAPTIONS["C9"]))
        self._first_loss_seen = True
        return events

    def _arrived_during_a_write(self, k):
        """Did frame k reach the computer while the loop was writing to disk? Looks at every write of the run, by time."""
        arrival = k * self.period + self.latency
        starts, ends = self._writes
        i = bisect.bisect_right(starts, arrival) - 1  # the newest write that began at or before the arrival
        return i >= 0 and arrival <= ends[i]

    def _caught_late(self, i, k, t):
        """V4: the tick before this row's came after frame k reached the computer and did not find it (C11)."""
        this = bisect.bisect_right(self._ticks, t[i]) - 1  # the tick of this row: the newest wake-up at or before it
        if this < 1:
            return []
        previous = self._ticks[this - 1]
        if previous > k * self.period + self.latency:
            return [Event(t[i], "C11", CAPTIONS["C11"].format(k=k, d=_ms(t[i] - previous)))]
        return []

    def _unexplained_gap(self, lo, hi):
        """C14, when no other caption tells why this gap is long: a long write (V1) or redraw (V2-V4) has its own."""
        inside = [s for s in self._spans if s[2] > lo and s[1] < hi]
        told = [s for s in inside if (s[0] == "writing to disk" and s[2] - s[1] > DISK_STALL_S and self.version == "V1")
                or (s[0] == "redrawing the plot" and s[2] - s[1] > REDRAW_S)]
        waits = _spent(inside, lo, hi)
        if told or not waits:
            return []
        return [Event(hi, "C14", CAPTIONS["C14"].format(d=_ms(hi - lo), thread=thread_words(self.version), waits=waits))]


class CaptionLine:
    """The one line "What just happened". Spec 3.2: a caption stays at least 3 s, or one second per 3 words if it is longer;
    a repeat of the caption on show is only counted, and the line then ends with "(23 times so far)"; the first event of a kind
    that has not been shown goes before repeats of a kind that has. A new kind waits for the time of the caption on show."""

    def __init__(self):
        self.totals = collections.Counter()  # events of each kind so far, shown or not
        self.shown = set()
        self.current, self.since = None, 0.0
        self._waiting = {}  # kind -> its newest event, in the order the kinds were first told

    def update(self, now, events):
        """Take the events; the text on show at `now` ('' before the first)."""
        for e in events:
            self.totals[e.kind] += 1
            if not self.current or e.kind != self.current.kind:
                self._waiting[e.kind] = e
        if self._waiting and (self.current is None or now - self.since >= self._stays(self.current)):
            self._show_next(now)
        return self.text()

    def _show_next(self, now):
        fresh = [kind for kind in self._waiting if kind not in self.shown]
        if fresh:
            kind = fresh[0]
        else:  # only repeats are waiting: the newest one tells what just happened, the older ones are out of date
            kind = max(self._waiting, key=lambda k: self._waiting[k].t)
            self._waiting = {kind: self._waiting[kind]}
        self.current, self.since = self._waiting.pop(kind), now
        self.shown.add(kind)

    @staticmethod
    def _stays(event):
        return max(HOLD_S, len(event.text.split()) / WORDS_PER_S)

    def text(self):
        if not self.current:
            return ""
        n = self.totals[self.current.kind]
        return f"{self.current.text} ({n} times so far)" if n >= 2 else self.current.text


def what_happened(events, limit=5):
    """The list "What happened in this run": the first event of each kind with its count, in time order, at most `limit` lines."""
    first, count = {}, collections.Counter()
    for e in events:
        first.setdefault(e.kind, e)
        count[e.kind] += 1
    lines = [e.text + (f" It happened {count[e.kind]} times in this run." if count[e.kind] > 1 else "") for e in first.values()]
    return lines[:limit]


# --- the question before Start, and the answer after the run -----------------------------------------------------


@dataclass(frozen=True)
class RunNumbers:
    """What a run counted: the numbers that the answers are made of (all of them also on the counters)."""
    rows: int
    frames_sent: int
    lost_pct: float  # the simulator knows
    lost_seen_pct: float | None  # as the file shows it; None when it has no frame numbers
    repeat_pct: float
    gap_1_in_100_ms: float  # NaN below 100 gaps
    longest_gap_ms: float


@dataclass(frozen=True)
class Question:
    ask: str
    choices: tuple[str, str, str, str]
    correct: str
    answer: str


# The wording of spec 4.2. V3: the spec still says that V3 stamps the frame's own time; V3 is now the fix of pull request #3 (the
# computer's time at the read, and the frame number), so its question and answer say that. A choice is about the same words in all four.
QUESTIONS = {
    "V0": Question(
        "Listing 1 moves, sleeps 2 s and then reads once, for every move. The sensor sends about {F} frames in this run. "
        "How many rows reach the file?",
        ("about {F}", "about {F10}", "about {F100}", "about {moves}"), "D",
        "This run: {rows} rows from {sent} frames. The loop read only after each move and each 2 s sleep, so the file has "
        "one row per move."),
    "V1": Question(
        "This loop reads every 20 ms and writes each row to disk before the next read. 1 gap in 100 is longer than how many ms?",
        ("about 20 ms", "about 25 ms", "about 60 ms", "about 200 ms"), "C",
        "This run: {gap}. Each time a disk write took longer than usual, the loop stopped reading until the write ended."),
    "V2": Question(
        "Each device now has its own thread; the sensor sends 50 frames a second and the loop reads 50 times a second. "
        "How many frames reach the file?",
        ("all of the frames", "about 2 in 3", "about 1 in 10", "almost none of them"), "B",
        "This run: {reached} % of the frames reached the file. Each frame reaches the computer at almost the same moment as the "
        "loop's tick, so delays of a few milliseconds in waking each thread decide which comes first, and a frame that "
        "comes just after one tick can be replaced just before the next."),
    "V3": Question(
        "Each row now carries the frame number and the time the computer read the frame. Compared with V2, how many frames "
        "are lost?",
        ("about the same share", "about half as many", "a few in 100", "no frames at all"), "A",
        "This run: {lost} % lost, and your file shows {seen} % (each run differs a little from the last). The loop still reads at "
        "the same moments, so it loses the same frames, and now the frame numbers show each loss."),
    "V4": Question(
        "The main loop now ticks every 10 ms, twice per frame. How many frames reach the file?",
        ("about 2 in 3", "about 1 in 3", "all of the frames", "about 1 in 10"), "C",
        "This run: {reached} % of the frames reached the file. The ticks are 10 ms apart, so one of the two ticks in each frame "
        "always comes at least 5 ms before or after the moment a frame reaches the computer, and that tick reads the new frame "
        "safely."),
}
V2_OPEN_QUESTION = "Look only at the file: how could you find the lost frames?"
V2_OPEN_ANSWER = ("Count the sensor values that are equal to the one before. When the loop and the sensor run at the same rate, "
                  "each repeat goes with one lost frame: in this run {repeat} % of the rows repeat, and {lost} % of the frames "
                  "are lost.")
INSTRUCTION = ("Choose A, B, C or D, then press Start (the window does not check your choice; it shows the answer after the "
               "run).")


def _two_digits(x):
    """x to two significant digits, as a number to print: 4200, 420, 42, 4.2."""
    return float(f"{x:.2g}")


def question_lines(version, frames_expected, moves):
    """[the question, then its four choices 'A: ...'] before the run. frames_expected: about how many frames the sensor sends
    in it; moves: how many moves the loop makes (V0)."""
    q = QUESTIONS[version]
    fill = {"F": f"{_two_digits(frames_expected):g}", "F10": f"{_two_digits(frames_expected / 10):g}",
            "F100": f"{_two_digits(frames_expected / 100):g}", "moves": moves}
    return [q.ask.format(**fill)] + [f"{letter}: {c.format(**fill)}" for letter, c in zip("ABCD", q.choices)]


def answer_lines(version, run):
    """The answer after a run, made of the numbers of that run (RunNumbers). V2 adds the open question and its answer."""
    gap = (f"1 gap in 100 is longer than {run.gap_1_in_100_ms:.0f} ms" if math.isfinite(run.gap_1_in_100_ms)  # from 100 gaps on
           else f"the longest gap is {run.longest_gap_ms:.0f} ms")
    lines = [QUESTIONS[version].answer.format(
        rows=run.rows, sent=run.frames_sent, gap=gap, reached=f"{100 - run.lost_pct:.0f}", lost=f"{run.lost_pct:.0f}",
        seen="-" if run.lost_seen_pct is None else f"{run.lost_seen_pct:.0f}")]
    if version == "V2":
        lines += [V2_OPEN_QUESTION, V2_OPEN_ANSWER.format(repeat=f"{run.repeat_pct:.0f}", lost=f"{run.lost_pct:.0f}")]
    return lines


# --- where the time went, and the runs so far --------------------------------------------------------------------


class Tracer:
    """The trace= hook of the loops: passes each wait on to `recorder` and adds up the time of each kind of wait of each thread.

    The recorder keeps only its newest spans (the load-cell reader makes two thousand a second); the sums keep the whole run."""

    def __init__(self, recorder):
        self.recorder, self._lock = recorder, threading.Lock()
        self._seconds = collections.defaultdict(float)

    def __call__(self, thread, kind, t0, t1):
        self.recorder(thread, kind, t0, t1)
        with self._lock:
            self._seconds[thread, kind] += t1 - t0

    def seconds(self):
        """{(thread, kind): seconds spent}. A copy."""
        with self._lock:
            return dict(self._seconds)


WORKING = "no wait recorded"  # the rest of a thread's time, the name of the empty part of a bar


def wait_shares(seconds, threads, run_s):
    """{thread: {kind: share of the run}} for each thread; WORKING gets what the waits leave. A thread whose waits add up to more
    than the run (the clocks of two spans overlap) has no WORKING part, and its shares are not rescaled: a bar over 100 % shows it."""
    shares = {}
    for thread in threads:
        mine = {kind: s / run_s for (th, kind), s in seconds.items() if th == thread and s > 0}
        shares[thread] = {**mine, WORKING: max(1.0 - sum(mine.values()), 0.0)}
    return shares


def settings_text(values):
    """'defaults', or the settings that are not the defaults: 'Logging rate: 100 Hz, Slow disk: on'."""
    changed = settings.differing(values)
    parts = []
    for key in changed:
        control = settings.BY_KEY[key]
        name, unit = re.fullmatch(r"(.*?)(?: \((.*)\))?", control.label).groups()
        value = ("on" if values[key] else "off") if control.lo is None else f"{values[key]:g}" + (f" {unit}" if unit else "")
        parts.append(f"{name}: {value}")
    return ", ".join(parts) or "defaults"


@dataclass(frozen=True)
class RunRecord:
    version: str
    settings: str  # settings_text of the run
    numbers: RunNumbers


RUNS_KEPT = 8
RUNS_HEADER = ("Run", "Settings", "Rows", "Frames lost", "Repeats", "1 gap in 100", "Longest gap")


def runs_table(records):
    """The rows of "Runs so far", newest first, at most RUNS_KEPT: one tuple of text for each, in the order of RUNS_HEADER."""
    def ms(x):
        return f"{x:.0f} ms" if math.isfinite(x) else "-"
    return [(r.version, r.settings, str(r.numbers.rows), f"{r.numbers.lost_pct:.0f} %",
             f"{r.numbers.repeat_pct:.0f} %" if math.isfinite(r.numbers.repeat_pct) else "-",
             ms(r.numbers.gap_1_in_100_ms), ms(r.numbers.longest_gap_ms)) for r in reversed(records[-RUNS_KEPT:])]
