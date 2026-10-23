"""The browser interface of the app: a small web server on the real Session of app.py, and the page in web/.

Run:  uv run demo/app.py        (starts this server and opens the page; `--no-browser` only prints the address; `--classic` is
                                 the old matplotlib window)

Two parts, kept apart:
  Engine   what the View of the classic window does, without a window: the chosen version, the settings, the one Session, the
           caption line, the answer of the finished run and the runs so far; state() is the JSON that the page draws. The
           loops, the Session and the redraw switch (LoopRender, a real Agg redraw inside the logging loop) are the ones of app.py.
  Handler  the HTTP edge (Python's standard library only): the page and its files, the GET views, and the POST buttons.
The page asks for /api/state about ten times a second, as the classic window's timer did; every ask builds the state from
one snapshot of the Session, the work of one tick of the classic window except the drawing. The views Guess first (What
happened after a run) and Compare runs ask for /api/guess and /api/runs only when the state says that their content changed
(run_id, phase, version and settings; runs_done).

API (JSON; every POST answers with the new state, or {"error": ...} with 400 / 409):
  GET  /api/meta      the names, colours and texts that do not change (versions, kinds of wait, labels)
  GET  /api/state     phase, status, counters, caption, force plot, timeline, error: all the Run view shows; run_id (runs
                      started) and runs_done (complete runs recorded)
  GET  /api/guess     the view Guess first: the question of the chosen version; after its complete run the answer, what happened
                      and the checks on the file
  GET  /api/runs      the view Compare runs: the complete runs, newest first (at most teach.RUNS_KEPT): where the time went and
                      the frames lost of each, and the rows of the table "Runs so far"
  POST /api/version   {"version": "V4"}      choose the version (not while a run goes)
  POST /api/redraw    {"on": true}           the redraw switch, read at Start
  POST /api/start     {"plot_px": [w, h], "short": true}   start a run; plot_px: the size of the force plot, for the redraw
                      (optional); short: this run with the settings of the Short run (settings.SHORT_RUN), the settings stay
  POST /api/stop                             end the run
  POST /api/short                            set the settings of the Short run (measure_web.py; the page uses "short" of Start)
  POST /api/defaults                         the framework's defaults
  POST /api/settings  {"key": "depth", "value": "0.7"}   one setting: the text typed in its box, or true / false for a switch;
                      a value out of range is refused (400) with the words of settings.parse. Not while a run goes, except the
                      plot window, which works at once
"""
from __future__ import annotations

import dataclasses
import json
import math
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import app
import loops
import settings
import teach
import timeline

WEB_DIR = Path(__file__).parent / "web"
HOST = "127.0.0.1"
LOST_PARTS = 20  # the band "frames lost" under the force plot: the share lost in each of this many parts of the plot window
MAX_POINTS = 1000  # of each series of the force plot: a longer window is thinned (every k-th point)
TIMELINE_PX = 640  # width of the timeline in the page at 1366 x 768: waits narrower than 1.5 px are merged, as the window merges them
PLOT_PX_DEFAULT = (620, 300)  # the force plot at 1366 x 768, when the page does not say (the redraw inside the loop draws this many pixels)
PLOT_PX_RANGE = (100, 8000)
REDRAW_DPI = 100
MAX_BODY_BYTES = 4096
# The words of View._notes in app.py (literals there) for the view Guess first, in the states with no answer
ANSWER_LATER = "The answer and the checks appear when the run has finished."
NO_CHECKS = "Run stopped before the end: no checks."
FAILED = "Something went wrong inside the program, so the run was stopped."
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".ttf": "font/ttf", ".txt": "text/plain; charset=utf-8"}


class Refused(Exception):
    """A request the Engine does not do; `status` is the HTTP code to answer with."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def thin(a, limit=MAX_POINTS):
    """Every k-th row of `a`, k the least whole number that leaves at most `limit` rows."""
    return a[::-(-len(a) // limit)] if len(a) > limit else a


def clean(x):
    """JSON-safe: numpy numbers and arrays to plain ones, NaN to None, floats to 4 decimals."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), 4) if math.isfinite(x) else None
    return x


def encode(x):
    return json.dumps(clean(x), separators=(",", ":"), allow_nan=False).encode()


# --- what the page draws ----------------------------------------------------------------------------------------


def force_view(session, snap, window_s, peak_n):
    """The force plot: the true force and the points of the file in the window that ends now (or at the end of the run), and
    under it the share of the frames lost in each part of that window, so that the loss shows in the plot itself."""
    lo, hi = session.plot_window(snap, window_s)
    i = max(int(np.searchsorted(snap.truth[:, 0], lo)) - 1, 0)  # as app.show_snapshot: one point before the window
    j = max(int(np.searchsorted(snap.rows[:, 0], lo)) - 1, 0)
    return {"peak_n": peak_n, "window": [lo, hi], "truth": thin(snap.truth[i:, :2]), "points": thin(snap.rows[j:, :2]),
            "lost": lost_parts(*session.frame_marks(snap), (lo, hi))}


def lost_parts(arrivals, fates, window):
    """Percent of the frames lost in each of LOST_PARTS equal parts of `window`, counted over the frames whose fate is known
    (a frame is placed at the moment it reached the computer); None for a part with no such frame."""
    edges = np.linspace(window[0], window[1], LOST_PARTS + 1)
    known = np.histogram(arrivals[fates != timeline.NOT_KNOWN], edges)[0]
    lost = np.histogram(arrivals[fates == timeline.LOST], edges)[0]
    return [100.0 * n / k if k else None for n, k in zip(lost, known)]


def timeline_view(session, snap):
    """The panel "Where the time goes": the waits of each thread in the last seconds, merged as the window merges them, one mark
    for each frame the sensor sent (when it reached the computer, and what became of it), and the times of the rows."""
    spans, window, row_times = session.trace_view(snap)
    lo, hi = window
    min_width = (hi - lo) / TIMELINE_PX * 1.5
    rows = {thread: [list(block) for block in timeline._blocks(spans[thread], window, min_width)]
            for thread in timeline.THREADS[session.version]}
    arrivals, fates = session.frame_marks(snap)
    inside = (arrivals >= lo) & (arrivals <= hi)
    return {"window": window, "rows": rows, "frames": {"t": arrivals[inside], "fate": fates[inside]},
            "ticks": row_times[(row_times >= lo) & (row_times <= hi)]}


def status_text(session, start_refused):
    """The line under the buttons: the words of View.status."""
    s = session
    if s.error:
        return "Stopped: something went wrong." if s.done else f"Something went wrong, stopping {s.version} ..."
    if not s.done:
        if s._stop.is_set():
            return f"Still stopping {s.version}: wait." if start_refused else f"Stopping {s.version} ..."
        if start_refused:
            return f"{s.version} is running: press Stop first."
        return f"Running {s.version}. Redraw in the loop: {'on' if s.redraw_in_loop else 'off'}"
    return f"Stopped. {s.version} was cut short." if s.cut else f"{s.version} has finished."


def phase(session):
    """before (no run yet), running, after (a complete run), stopped (cut short) or error."""
    if session is None:
        return "before"
    if not session.done:
        return "running"
    return "error" if session.error else ("stopped" if session.cut else "after")


def question(version, values):
    """The question of Guess first, as View._question builds it: {"ask": ..., "choices": [[letter, text], ...]}."""
    lines = teach.question_lines(version, app.expected_duration_s(version, values) * values["fps"], 2 * (values["cycles"] + 1))
    return {"ask": lines[0], "choices": [[line[0], line[3:]] for line in lines[1:]]}  # each choice is "A: text"


def answer_view(session, snap):
    """What happened, after a complete run (View._answer): the right choice, the share of frames lost (only the simulator knows
    it), the answer made of this run's numbers, the causes
    found in the run, and the checks on the file (app.SENTENCES of the checks it fails; check_sentences when it fails none)."""
    v = session.version
    correct = teach.QUESTIONS[v].correct
    choices = question(v, session.values)["choices"]
    ds, truth = session.sheet
    failed = session.failed_checks()
    numbers = session.numbers(snap)
    return {"version": v, "settings": teach.settings_text(session.values), "correct": correct,
            "choice": next(text for letter, text in choices if letter == correct), "lost_pct": numbers.lost_pct,
            "answer": teach.answer_lines(v, numbers), "what_happened": teach.what_happened(session.events),
            "checks": {"failed": len(failed), "count": len(app.SENTENCES),
                       "sentences": [app.SENTENCES[name](ds, truth) for name in failed] or app.check_sentences(session)}}


def control_view(control):
    """A setting as the Settings view shows it: the label split into its name and its unit, "Depth (mm)" -> Depth, mm."""
    name, unit = re.fullmatch(r"(.*?)(?: \((.*)\))?", control.label).groups()
    return {"key": control.key, "label": control.label, "name": name, "unit": unit or "", "default": control.default,
            "lo": control.lo, "hi": control.hi, "whole": control.whole}


def parse_setting(key, value):
    """The value of setting `key` from what the page sent: true or false for a switch, the text typed in the box for a number
    (settings.parse, in its words). Refused (400) otherwise."""
    control = settings.BY_KEY[key]
    if control.lo is None:
        if not isinstance(value, bool):
            raise Refused(400, f'{control.label}: "value" must be true or false.')
        return value
    if not isinstance(value, str):
        raise Refused(400, f'{control.label}: "value" must be the text typed in the box.')
    try:
        finite = math.isfinite(float(value.replace(",", ".")))
    except ValueError:
        finite = False
    if not finite:  # settings.parse fails on "inf" and "nan" in its whole-number test (OverflowError): refuse them first
        raise Refused(400, f"{control.label}: type a number.")
    try:
        return settings.parse(key, value)
    except ValueError as exc:
        raise Refused(400, str(exc)) from None


def meta():
    """What the page needs once: the versions with their rows and kinds of wait, the settings with their ranges, the colours
    and the texts of app.py, teach.py and timeline.py."""
    versions = [{
        "key": v, "name": app.VERSION_LABELS[v][2:].strip(), "threaded": v in loops.THREADED, "numbered": app._numbered(v),
        "threads": [{"key": t, "label": timeline.ROW_LABELS[t]} for t in timeline.THREADS[v]], "window_s": timeline.WINDOW_S[v],
        "kinds": list(timeline.kinds_for(v)), "kinds_with_redraw": list(timeline.kinds_for(v, True)),
        "fates": list(teach.FATES_OF[v]), "what_it_shows": timeline.what_it_shows(v),
    } for v in loops.VERSIONS]
    return {
        "versions": versions, "kinds": timeline.KINDS, "frame_legend": {str(code): text for code, text in timeline.FRAME_LEGEND.items()},
        "fate_codes": {"not_known": timeline.NOT_KNOWN, "once": timeline.ONCE, "twice": timeline.TWICE, "lost": timeline.LOST},
        "texts": {"caption_title": app.CAPTION_TITLE, "bars_title": app.BARS_TITLE, "bars_note": app.BARS_NOTE,
                  "table_title": app.TABLE_TITLE, "table_note": app.TABLE_NOTE, "hold_note": app.HOLD_NOTE,
                  "not_default_message": settings.MESSAGE_NOT_DEFAULT, "gap_limit_ms": app.GAP_LIMIT_MS},
        "runs_header": teach.RUNS_HEADER, "working": teach.WORKING,
        "share_label_min": app.SHARE_LABEL_MIN, "share_key_min": app.SHARE_KEY_MIN,
        "settings": {"framework": [control_view(c) for c in settings.CONTROLS], "simulated": [control_view(c) for c in settings.SIMULATED]},
    }


class Engine:
    """The chosen version, the settings and the one Session; the caption line; the answer of the finished run and the runs so far.
    One lock: a request does one thing at a time."""

    def __init__(self, out, seed=1):
        self.out, self.seed = Path(out), seed
        self.values = settings.defaults()
        self.version, self.redraw = "V0", False
        self.session, self.caption = None, teach.CaptionLine()
        self._caption_now = ""
        self._start_refused = False  # Start was pressed while a run was going: say so until the run has ended
        self._final = None  # the run view of a finished run, built once
        self._answer = None  # What happened: the answer of a complete run, built with _final
        self.short = False  # the session is a Short run
        self.run_id = 0  # runs started
        self.runs = []  # (teach.RunRecord, shares) of the complete runs, oldest first; the newest teach.RUNS_KEPT
        self.runs_done = 0  # complete runs recorded since the server started (the page asks /api/runs when this changes)
        self._lock = threading.Lock()

    # -- the buttons

    def _running(self):
        return self.session is not None and not self.session.done

    def _settings_free(self):
        if self._running():
            raise Refused(409, "A run is going: press Stop first.")

    def choose_version(self, version):
        with self._lock:
            if version not in loops.VERSIONS:
                raise Refused(400, f"Unknown version: {version!r}.")
            self._settings_free()
            if self.session is not None and self.session.version != version:
                self.session, self._final = None, None  # the picture of a finished run belongs to its version
            self.version = version

    def set_redraw(self, on):
        with self._lock:
            if not isinstance(on, bool):
                raise Refused(400, '"on" must be true or false.')
            self._settings_free()
            self.redraw = on

    def short_run(self):
        with self._lock:
            self._settings_free()
            self.values.update(settings.SHORT_RUN)

    def back_to_defaults(self):
        with self._lock:
            self._settings_free()
            self.values = settings.defaults()
            self._refresh_final_plot()

    def set_value(self, key, value):
        """One setting, from the Settings view. The settings are read at Start; the plot window works at once, also during a run."""
        with self._lock:
            if not isinstance(key, str) or key not in settings.BY_KEY:
                raise Refused(400, f"Unknown setting: {key!r}.")
            if key != "window":
                self._settings_free()
            self.values[key] = parse_setting(key, value)
            self._refresh_final_plot()

    def _refresh_final_plot(self):
        """A finished run in a new plot window (View.refresh: the same run, in the new window)."""
        s = self._final and self.session
        if s:
            self._final["force"] = force_view(s, s.snapshot(), self.values["window"], app.peak_force_n(s.values))

    def start(self, plot_px=None, short=False):
        """plot_px: [width, height] of the force plot in pixels; the redraw inside the loop draws a plot of this size.
        short: a Short run (1 cycle, about 21 s); the settings are not changed by it."""
        with self._lock:
            if self._running():
                self._start_refused = True
                raise Refused(409, status_text(self.session, True))
            if not isinstance(short, bool):
                raise Refused(400, '"short" must be true or false.')
            width, height = PLOT_PX_DEFAULT if plot_px is None else self._plot_px(plot_px)
            values = {**self.values, **settings.SHORT_RUN} if short else self.values
            self._start_refused = False
            on_tick = None
            if self.redraw and self.version in loops.THREADED:
                render = app.LoopRender((width, height, REDRAW_DPI), values["window"], app.peak_force_n(values))

                def on_tick():  # `session` is bound below, before the logging loop starts
                    render.draw(session.snapshot(), (width, height, REDRAW_DPI))
            session = app.Session(self.version, values, self.out, self.seed, on_tick)
            self.short = short
            self.session, self.caption, self._caption_now, self._final, self._answer = session, teach.CaptionLine(), "", None, None
            self.run_id += 1
            session.start()

    @staticmethod
    def _plot_px(plot_px):
        ok = (isinstance(plot_px, list) and len(plot_px) == 2 and all(isinstance(n, int) and not isinstance(n, bool) for n in plot_px)
              and all(PLOT_PX_RANGE[0] <= n <= PLOT_PX_RANGE[1] for n in plot_px))
        if not ok:
            raise Refused(400, f'"plot_px" must be [width, height], whole pixels between {PLOT_PX_RANGE[0]} and {PLOT_PX_RANGE[1]}.')
        return plot_px

    def stop(self):
        with self._lock:
            if self.session:
                self.session.stop()

    def close(self):
        """The server ends: end the run, wait for its threads."""
        self.stop()
        if self.session:
            self.session.join(timeout=self.session.stop_limit_s())

    # -- the state

    def state(self):
        """The JSON bytes of everything the Run view shows, and what tells the page that the other views changed."""
        with self._lock:
            s = self.session
            state = {"phase": phase(s), "version": self.version, "redraw": self.redraw, "values": self.values,
                     "not_default": bool(settings.differing(self.values)), "locked": self._running(), "run_id": self.run_id,
                     "short": self.short and s is not None, "run_s": self._durations()}
            state["redraw_in_loop"] = s.redraw_in_loop if s else self.redraw and self.version in loops.THREADED
            state["error"] = None
            if s is None:
                return encode({**state, **self._idle_view(), "runs_done": self.runs_done})
            view = self._view()
            if s.error:
                state["error"] = {"type": type(s.error).__name__, "message": str(s.error)}
            state["phase"] = phase(s)  # the view may have found an error
            state["status"] = status_text(s, self._start_refused)
            return encode({**state, **view, "runs_done": self.runs_done})

    def guess(self):
        """The JSON bytes of the view Guess first: the question of the chosen version with the line under it; after a complete
        run of it (the rail then says "What happened"), also its answer."""
        with self._lock:
            s, v = self.session, self.version
            if s is not None:
                self._view()  # a run that has just ended: its answer is built with its last picture
            state = phase(s)
            line = {"before": teach.INSTRUCTION, "running": ANSWER_LATER, "stopped": NO_CHECKS, "error": FAILED, "after": None}[state]
            return encode({"version": v, "phase": state, "run_id": self.run_id, "question": question(v, self.values),
                           "what_it_shows": timeline.what_it_shows(v), "run_s": self._durations(),
                           "line": line, "answer": self._answer if state == "after" else None})

    def runs_view(self):
        """The JSON bytes of the view Compare runs: the complete runs, newest first, each with where its time went (the shares of
        each thread) and its frames lost; and the rows of the table "Runs so far"."""
        with self._lock:
            if self.session is not None:
                self._view()
            newest_first = self.runs[::-1]
            return encode({"runs_done": self.runs_done, "table": teach.runs_table([record for record, _ in self.runs]),
                           "runs": [{"version": r.version, "settings": r.settings, "lost_pct": r.numbers.lost_pct, "shares": shares}
                                    for r, shares in newest_first]})

    def _view(self):
        """The run view of the session: the one built at its end, or a new one."""
        s = self.session
        try:
            return self._final or self._run_view()
        except Exception as exc:  # an error here must stop the run and tell the presenter, not freeze the page
            s.fail(exc)
            return {"now_s": 0.0, "total_s": 0.0, "counters": None, "caption": "", "force": None, "timeline": None}

    def _durations(self):
        """About how long a run of the chosen version takes: {"full": s, "short": s} (the two Start buttons)."""
        v, values = self.version, self.values
        return {"full": app.expected_duration_s(v, values), "short": app.expected_duration_s(v, {**values, **settings.SHORT_RUN})}

    def _idle_view(self):
        """No run yet: empty plots with their axes, and what each of the two Start buttons does."""
        v, values = self.version, self.values
        run_s = self._durations()
        duration = run_s["full"]
        status = (f"Start: the whole run, about {run_s['full']:.0f} s.\n"
                  f"Short run: {settings.SHORT_RUN['cycles']} cycle, about {run_s['short']:.0f} s.")
        empty = np.empty((0, 2))
        return {"status": status, "now_s": 0.0, "total_s": duration, "counters": None, "caption": "",
                "force": {"peak_n": app.peak_force_n(values), "window": [0, values["window"]], "truth": empty, "points": empty,
                          "lost": [None] * LOST_PARTS},
                "timeline": {"window": [0, timeline.WINDOW_S[v]], "rows": {t: [] for t in timeline.THREADS[v]},
                             "frames": {"t": [], "fate": []}, "ticks": []}}

    def _run_view(self):
        """What a tick of the classic window does, without the drawing: one snapshot, the counters, the caption, the two plots.
        A run that has ended is built once (`_final`): its picture does not change any more."""
        s = self.session
        done = s.done  # read first: a state built after the run ended is the last one
        snap = s.snapshot()
        counters = s.counters(snap)
        self._caption_now = self.caption.update(snap.now, s.new_events(snap))
        view = {"now_s": s.t_end if done else snap.now,
                "total_s": s.t_end if done else app.expected_duration_s(s.version, s.values),
                "counters": dataclasses.asdict(counters), "caption": self._caption_now,
                "force": force_view(s, snap, self.values["window"], app.peak_force_n(s.values)),
                "timeline": timeline_view(s, snap)}
        if done:
            if s.sheet is not None and not (s.cut or s.error):  # a complete run: its answer, and a row of the runs so far
                self._record(s, snap)
            self._final = view
        return view

    def _record(self, s, snap):
        """What a complete run leaves for the other views (View._show_final): the answer, and its record with its shares."""
        self._answer = answer_view(s, snap)
        record = teach.RunRecord(s.version, teach.settings_text(s.values), s.numbers(snap))
        self.runs = (self.runs + [(record, s.shares(snap))])[-teach.RUNS_KEPT:]
        self.runs_done += 1


# --- the HTTP edge ----------------------------------------------------------------------------------------------


def web_files():
    """{URL path: file} of everything in web/, and "/" for index.html: the only files the server gives out."""
    files = {"/" + p.relative_to(WEB_DIR).as_posix(): p for p in WEB_DIR.rglob("*") if p.is_file()}
    files["/"] = files.pop("/index.html")
    return files


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # the page asks ten times a second: keep the connection

    def log_message(self, *args):  # the terminal is the presenter's: no line for each ask
        pass

    def _send(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _check_host(self):
        """Only the page itself may ask: a page from another site that points at this address is refused (DNS rebinding)."""
        if self.headers.get("Host") not in self.server.hosts:
            raise Refused(403, "Wrong host name.")

    def do_GET(self):
        try:
            self._check_host()
            if self.path == "/api/state":
                return self._send(200, self.server.engine.state())
            if self.path == "/api/guess":
                return self._send(200, self.server.engine.guess())
            if self.path == "/api/runs":
                return self._send(200, self.server.engine.runs_view())
            if self.path == "/api/meta":
                return self._send(200, self.server.meta)
            path = self.server.files.get(self.path)
            if path is None:
                raise Refused(404, "Not found.")
            self._send(200, path.read_bytes(), CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
        except Refused as exc:
            self._send(exc.status, encode({"error": str(exc)}))

    def do_POST(self):
        engine = self.server.engine
        routes = {"/api/version": lambda body: engine.choose_version(body.get("version")),
                  "/api/redraw": lambda body: engine.set_redraw(body.get("on")),
                  "/api/start": lambda body: engine.start(body.get("plot_px"), body.get("short", False)),
                  "/api/stop": lambda body: engine.stop(),
                  "/api/short": lambda body: engine.short_run(),
                  "/api/defaults": lambda body: engine.back_to_defaults(),
                  "/api/settings": lambda body: engine.set_value(body.get("key"), body.get("value"))}
        try:
            self._check_host()
            if self.path not in routes:
                raise Refused(404, "Not found.")
            routes[self.path](self._json_body())
            self._send(200, engine.state())
        except Refused as exc:
            self._send(exc.status, encode({"error": str(exc)}))

    def _json_body(self):
        """The body as a dict. JSON is required, which a page of another site cannot send without asking first."""
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            raise Refused(415, "The body must be JSON.")
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise Refused(413, "The body is too long.")
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            raise Refused(400, "The body is not JSON.") from None
        if not isinstance(body, dict):
            raise Refused(400, "The body must be a JSON object.")
        return body


class AppServer(ThreadingHTTPServer):
    daemon_threads = True  # a connection that stays open does not keep the process alive

    def __init__(self, engine, port=0):
        super().__init__((HOST, port), Handler)
        self.engine, self.files, self.meta = engine, web_files(), encode(meta())
        self.hosts = {f"{name}:{self.server_port}" for name in (HOST, "localhost")}

    @property
    def url(self):
        return f"http://{HOST}:{self.server_port}/"


def main(out, seed=1, open_browser=True):
    """Serve the page until Ctrl+C; then end the run and close the server."""
    engine = Engine(out, seed)
    server = AppServer(engine)
    print(f"The app is at {server.url}  (Ctrl+C stops it)", flush=True)
    if open_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()
        server.server_close()
