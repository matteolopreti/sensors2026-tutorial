"""Tests of the browser interface (webapp.py): the real server, the real Session and the real loops, asked over HTTP.

Run:  cd demo && MPLBACKEND=Agg /usr/bin/python3 -m pytest -q -p no:cacheprovider test_webapp.py   (about 40 s: two runs of V4 go on in real time)
"""
from __future__ import annotations

import dataclasses
import http.client
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import app
import loops
import settings
import webapp

DEMO = Path(__file__).parent
RUN_TIMEOUT_S = 60  # a Short run takes about 21 s
SHORT_VALUES = {**settings.defaults(), **settings.SHORT_RUN}  # the settings of a Short run started at the defaults


class Client:
    """What the page does: GET and POST JSON to the server."""

    def __init__(self, server):
        self.server = server

    def request(self, method, path, body=None, content_type="application/json", host=None):
        """-> (status, body bytes). `host`: the Host header to send instead of the server's own."""
        conn = http.client.HTTPConnection(webapp.HOST, self.server.server_port, timeout=10)
        headers = {"Content-Type": content_type} if method == "POST" else {}
        if host:
            headers["Host"] = host
        conn.request(method, path, json.dumps(body) if body is not None else None, headers)
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, data

    def get(self, path):
        status, data = self.request("GET", path)
        return status, json.loads(data) if data.startswith(b"{") else data

    def post(self, path, body=None):
        status, data = self.request("POST", path, {} if body is None else body)
        return status, json.loads(data)

    def state(self):
        return self.get("/api/state")[1]

    def wait_for(self, phases, timeout_s=RUN_TIMEOUT_S):
        """Poll the state, as the page does, until its phase is one of `phases`."""
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            state = self.state()
            if state["phase"] in phases:
                return state
            time.sleep(0.1)
        raise AssertionError(f"no phase in {phases} after {timeout_s} s; the last state: {state['phase']}, {state['status']}")


@pytest.fixture
def web(tmp_path):
    engine = webapp.Engine(tmp_path)
    server = webapp.AppServer(engine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield Client(server), engine
    engine.close()
    server.shutdown()
    server.server_close()
    thread.join()


# --- the page and its files ------------------------------------------------------------------------------------


def test_page_and_files_are_served(web):
    client, _ = web
    status, page = client.get("/")
    assert status == 200 and b"<title>Rig demo</title>" in page
    for path in ("/style.css", "/app.js", "/fonts/IBMPlexSans-Regular.ttf", "/fonts/IBMPlexMono.ttf"):
        assert client.get(path)[0] == 200, path
    assert client.get("/nothing")[0] == 404
    assert client.get("/../webapp.py")[0] == 404  # only the files of web/ are given out


def test_meta_names_every_version(web):
    client, _ = web
    status, meta = client.get("/api/meta")
    assert status == 200
    assert [v["key"] for v in meta["versions"]] == list(loops.VERSIONS)
    assert set(meta["kinds"]) >= {k for v in meta["versions"] for k in v["kinds"]}


def test_other_hosts_and_other_bodies_are_refused(web):
    client, _ = web
    assert client.request("GET", "/api/state", host="evil.example")[0] == 403
    assert client.request("POST", "/api/stop", {}, host="evil.example")[0] == 403
    assert client.request("POST", "/api/stop", {}, content_type="text/plain")[0] == 415  # what a page of another site can send
    assert client.post("/api/version", {"version": "V9"})[0] == 400
    assert client.post("/api/redraw", {"on": "yes"})[0] == 400
    assert client.post("/api/start", {"plot_px": [10, 10]})[0] == 400
    assert client.post("/api/start", {"plot_px": "wide"})[0] == 400
    assert client.state()["phase"] == "before"


def test_idle_state_has_the_settings_and_empty_plots(web):
    client, _ = web
    state = client.state()
    assert state["phase"] == "before" and state["counters"] is None and not state["locked"]
    assert state["values"] == settings.defaults() and not state["not_default"]
    assert state["force"]["window"] == [0, settings.defaults()["window"]] and state["force"]["points"] == []
    status, state = client.post("/api/short")
    assert status == 200 and state["not_default"] and state["values"]["cycles"] == settings.SHORT_RUN["cycles"]
    assert client.post("/api/defaults")[1]["values"] == settings.defaults()


def test_the_views_guess_compare_and_settings_answer(web):
    """A smoke test of the three views beside Run: each answers before any run, and a setting is taken or refused in words."""
    client, _ = web
    client.post("/api/version", {"version": "V2"})
    status, guess = client.get("/api/guess")
    assert status == 200 and guess["phase"] == "before" and guess["answer"] is None  # no answer before the run
    assert [letter for letter, _ in guess["question"]["choices"]] == ["A", "B", "C", "D"]
    assert client.get("/api/runs") == (200, {"runs_done": 0, "table": [], "runs": []})
    status, state = client.post("/api/settings", {"key": "depth", "value": "0,7"})
    assert status == 200 and state["values"]["depth"] == 0.7 and state["not_default"]
    # refused, in the words of the classic window's settings page (test_app.py)
    assert client.post("/api/settings", {"key": "depth", "value": "21"}) == (400, {"error": "Depth (mm): between 0.1 and 20."})
    assert client.post("/api/settings", {"key": "cycles", "value": "inf"}) == (400, {"error": "Cycles: type a number."})
    assert client.state()["values"]["depth"] == 0.7
    assert client.post("/api/defaults")[1]["values"] == settings.defaults()


# --- a run through the API --------------------------------------------------------------------------------------


def start_short_run(client, version):
    """What the page's Short run button does: a run with the settings of the Short run; the settings stay as they are."""
    assert client.post("/api/version", {"version": version})[0] == 200
    status, state = client.post("/api/start", {"plot_px": [620, 300], "short": True})
    assert status == 200 and state["phase"] == "running" and state["locked"] and state["short"]
    assert state["values"] == settings.defaults() and not state["not_default"]  # no "Not the defaults" for a Short run


def test_a_short_run_of_v4_ends_with_rows_and_the_counters_of_a_session(web):
    client, engine = web
    start_short_run(client, "V4")
    during = client.wait_for({"running"})
    assert during["status"] == "Running V4. Redraw in the loop: off"
    assert client.post("/api/start")[0] == 409  # a second Start while the run goes
    assert client.post("/api/version", {"version": "V2"})[0] == 409
    assert "press Stop first" in client.state()["status"]
    final = client.wait_for({"after"})
    counters = final["counters"]
    assert set(counters) == {f.name for f in dataclasses.fields(app.Counters)}
    assert counters["rows"] > 0 and counters["frames_sent"] > 0
    assert final["status"] == "V4 has finished." and final["error"] is None and not final["locked"]
    assert final["now_s"] == pytest.approx(final["total_s"]) and final["total_s"] == pytest.approx(settings.run_s(SHORT_VALUES), abs=2.0)
    # the page's numbers are the Session's own
    assert counters["rows"] == engine.session.sheet[0]["samples"]
    assert counters["lost_pct"] == pytest.approx(engine.session.sheet[0]["frames_lost_true_pct"], abs=1e-3)
    # the force plot: only the window, thinned
    force = final["force"]
    lo, hi = force["window"]
    assert hi - lo == settings.defaults()["window"]
    assert 0 < len(force["points"]) <= webapp.MAX_POINTS and 0 < len(force["truth"]) <= webapp.MAX_POINTS
    assert all(lo - 1 <= t <= hi for t, _ in force["points"][1:])
    assert final["timeline"]["frames"]["t"] and final["timeline"]["rows"]["file writer"]
    assert client.state() == final  # a finished run does not change


def test_stop_ends_a_run_midway(web):
    client, engine = web
    start_short_run(client, "V4")
    while client.state()["now_s"] < 1.0:
        time.sleep(0.1)
    t0 = time.monotonic()
    assert client.post("/api/stop")[0] == 200
    state = client.wait_for({"stopped"}, timeout_s=engine.session.stop_limit_s() + 2)
    assert time.monotonic() - t0 < engine.session.stop_limit_s() + 1
    assert state["status"] == "Stopped. V4 was cut short." and state["counters"]["rows"] > 0
    assert state["now_s"] < settings.run_s(SHORT_VALUES) - 5


def test_the_redraw_switch_reaches_the_session(web, monkeypatch):
    client, engine = web
    drawn = []
    draw = app.LoopRender.draw

    def counting_draw(self, snap, pixels):
        drawn.append(pixels)
        draw(self, snap, pixels)

    monkeypatch.setattr(app.LoopRender, "draw", counting_draw)
    assert client.post("/api/version", {"version": "V2"})[0] == 200
    assert client.post("/api/redraw", {"on": True})[1]["redraw_in_loop"] is True
    client.post("/api/short")
    state = client.post("/api/start", {"plot_px": [640, 320]})[1]
    assert engine.session.redraw_in_loop and state["status"] == "Running V2. Redraw in the loop: on"
    time.sleep(1.5)
    client.post("/api/stop")
    client.wait_for({"stopped"})
    assert len(drawn) > 10 and set(drawn) == {(640, 320, webapp.REDRAW_DPI)}  # a real draw of the size of the page's plot, at every tick
    # a version without the hook ignores the switch
    client.post("/api/version", {"version": "V0"})
    assert client.state()["redraw_in_loop"] is False


def test_an_error_in_a_run_shows_in_the_state(web, monkeypatch):
    client, engine = web

    def failing_draw(self, snap, pixels):
        raise RuntimeError("the redraw failed")

    monkeypatch.setattr(app.LoopRender, "draw", failing_draw)
    client.post("/api/version", {"version": "V2"})
    client.post("/api/redraw", {"on": True})
    client.post("/api/short")
    client.post("/api/start", {"plot_px": [640, 320]})
    state = client.wait_for({"error"}, timeout_s=10)
    assert state["error"] == {"type": "RuntimeError", "message": "the redraw failed"}
    assert state["status"] == "Stopped: something went wrong."
    assert state["counters"] is not None and not state["locked"]
    assert client.post("/api/start", {"plot_px": [640, 320]})[0] == 200  # the page can start again


# --- the pieces ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n, limit, kept", [(0, 1000, 0), (999, 1000, 999), (1000, 1000, 1000), (1001, 1000, 501), (10_000, 1000, 1000), (15_000, 1000, 1000)])
def test_thin_keeps_at_most_the_limit(n, limit, kept):
    a = np.arange(n * 2, dtype=float).reshape(n, 2)
    out = webapp.thin(a, limit)
    assert len(out) == kept and (n == 0 or out[0, 0] == 0)


def test_clean_makes_strict_json():
    data = {"a": np.array([[1.23456789, np.nan]]), "b": np.int64(3), "c": float("inf"), "d": (1, "x")}
    assert json.loads(webapp.encode(data)) == {"a": [[1.2346, None]], "b": 3, "c": None, "d": [1, "x"]}


# --- the command ------------------------------------------------------------------------------------------------


def test_app_py_no_browser_serves_the_page_and_stops_on_ctrl_c():
    env = {**os.environ, "MPLBACKEND": "Agg"}
    proc = subprocess.Popen([sys.executable, "app.py", "--no-browser"], cwd=DEMO, env=env, stdout=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline()
        assert line.startswith("The app is at http://127.0.0.1:"), line
        url = line.split()[4]
        with urllib.request.urlopen(url, timeout=10) as r:
            assert r.status == 200 and b"<title>Rig demo</title>" in r.read()
        with urllib.request.urlopen(url + "api/state", timeout=10) as r:
            assert json.load(r)["phase"] == "before"
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_classic_and_no_browser_do_not_go_together():
    r = subprocess.run([sys.executable, "app.py", "--classic", "--no-browser"], cwd=DEMO, capture_output=True, text=True, timeout=60,
                       env={**os.environ, "MPLBACKEND": "Agg"})
    assert r.returncode == 2 and "--no-browser belongs to the browser interface" in r.stderr
