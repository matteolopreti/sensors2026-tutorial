"""One acquisition loop, version by version; each version fixes the previous one's bug.

Every version logs rows and writes each row through rig.disk. A row is (t, force, sensor, frame number, sensor time):
V0 to V2 write the first three columns, V3 and V4 all five (the frame number and the time of the sensor value are the
columns that the framework's pull request #3 added).
The protocol is a list of steps (target depth in mm, hold in s after reaching it): the framework's cycle protocol
(naive_vs_threaded.protocol). `cancel`, an Event, ends a run early: V0 sees it between steps, the other versions at once.
"""
import collections
import math
import queue
import threading
import time

import numpy as np


class RateLimiter:
    """Fixed-rate tick on perf_counter deadlines (framework core/timing.py)."""

    def __init__(self, rate_hz, origin=None):
        """origin: a tick lands on origin + n * period (perf_counter s); default: first tick now."""
        self.period, now = 1.0 / rate_hz, time.perf_counter()
        self.next = now if origin is None else origin + math.ceil((now - origin) / self.period) * self.period - self.period

    def wait(self, stop=None):
        self.next += self.period
        delay = self.next - time.perf_counter()
        if delay <= 0:  # overrun: skip whole periods, keep the phase
            self.next += math.floor(-delay / self.period) * self.period
        elif stop is None:
            time.sleep(delay)
        else:
            stop.wait(delay)


V0_SLEEP_S = 2.0  # Listing 1: sleep(2.0) after each move, whatever the hold of the framework protocol


def traced(trace, world, thread, kind, fn, *args):
    """fn(*args). With a trace, also report trace(thread, kind, t0, t1): the start and end of the call on the world clock.

    trace=None is the default everywhere: the call goes straight through and nothing is recorded (timeline.Recorder).
    """
    if trace is None:
        return fn(*args)
    t0 = world.now()
    try:
        return fn(*args)
    finally:
        trace(thread, kind, t0, world.now())


def v0_naive(rig, steps, trace=None, cancel=None):
    """AIS SI Listing 1: blocking move, sleep(2.0), one read pair per step. It knows nothing of the hold of the steps."""
    log, w, cancel = [], rig.world, cancel or threading.Event()
    for z, _hold in steps:
        if cancel.is_set():
            break
        traced(trace, w, "script", "moving the stage", rig.stage.move_to, z)
        traced(trace, w, "script", "sleeping", time.sleep, V0_SLEEP_S)
        f = traced(trace, w, "script", "reading the load cell", rig.load_cell.read)
        s, _, k = traced(trace, w, "script", "waiting for the sensor frame", rig.sensor.read)  # serial.read(): blocks until the next frame
        row = (rig.world.now(), f, s, k)  # k: simulator frame index, never logged by V0-V2
        traced(trace, w, "script", "writing to disk", rig.disk.write, row[:3])
        log.append(row)
    rig.t_stop = rig.world.now()  # logging ends here: frames produced later are not lost frames
    return log


def v1_polling(rig, steps, rate_hz=50.0, trace=None, cancel=None):
    """V0 without the blocking move and sleep: one thread reads every tick through ramps and holds."""
    log, lim, w, cancel = [], RateLimiter(rate_hz), rig.world, cancel or threading.Event()
    for z, hold in steps:
        rig.stage.start(z)  # V0: move_to(z) blocked
        t_hold = None
        while (t_hold is None or rig.world.now() < t_hold + hold) and not cancel.is_set():  # V0: time.sleep(2.0)
            if t_hold is None and rig.stage.position() == z:
                t_hold = rig.world.now()
            f = traced(trace, w, "loop", "reading the load cell", rig.load_cell.read)
            s, _, k = traced(trace, w, "loop", "waiting for the sensor frame", rig.sensor.read)
            row = (rig.world.now(), f, s, k)
            traced(trace, w, "loop", "writing to disk", rig.disk.write, row[:3])  # a disk stall still stops every read
            log.append(row)
            traced(trace, w, "loop", "sleeping", lim.wait)
        if cancel.is_set():
            break
    rig.t_stop = rig.world.now()
    return log


PHASE_MS = 4.0  # pinned start offset (ms) of the tick vs the frame clock: tick coincides with frame arrival -> loss peaks (host-dependent)


NO_FRAME = (math.nan, math.nan, 0, math.nan)  # (value, frame time, frame number, read time) before the first frame


def stamped_on_read(now, forces, sensor):
    """V2 row: both channels stamped at the tick; the sensor value is up to a frame + latency old.

    Before the first frame the sensor value is 0, as the framework wrote it before pull request #3.
    Lag sign: positive = sensor channel older than the force channel.
    """
    value, _t_frame, k, _t_read = sensor
    return now, forces[-1][1], 0.0 if k == 0 else value, k


def stamped_at_read(now, forces, sensor):
    """V3 row, the fix of the framework's pull request #3 (commit c631fde): the sensor value has its own time stamp, the
    computer's time at the moment the frame was read (the frame arrived a little earlier), and the frame number.

    The row time stays the tick's. Before the first frame: value and time NaN, frame number 0: no fabricated zeros."""
    value, _t_frame, k, t_read = sensor
    return (now, forces[-1][1], value, k, t_read) if k else (now, forces[-1][1], math.nan, 0, math.nan)


def stamped_by_sensor(now, forces, sensor):
    """V3 with "the sensor sends its own time": the time stamp of the sensor value is the moment the sensor took the frame."""
    value, t_frame, k, _t_read = sensor
    return (now, forces[-1][1], value, k, t_frame) if k else (now, forces[-1][1], math.nan, 0, math.nan)


def start_and_wait(rig, z, cancel=None):
    """Default move: non-blocking start, wait for the ramp end (or for cancel)."""
    (cancel or threading.Event()).wait(max(rig.stage.start(z) - rig.world.now(), 0.0))


def threaded(rig, steps, make_row, rate_hz=50.0, log_in_dwell=True, phase_ms=PHASE_MS, log_seq=False, new_frames_only=False,
             mover=None, on_tick=None, log=None, trace=None, cancel=None):
    """Device pollers -> latest-value registers; fixed-rate coordinator -> FIFO -> file writer thread.

    phase_ms: coordinator ticks fall on world.t0 + phase + n/rate; sensor frames on n * period (+ 4 ms latency).
    log_seq: the rows carry the frame number and the sensor time (V3+): five columns in the file, else three.
    new_frames_only: a tick writes a row only when a frame came that no row has had yet (V4; V3 writes a row every tick).
    log_in_dwell=False is the planted defect for check-the-checker: no row during a hold.
    Lane hooks (lanes.py): mover(rig, z) makes one move; on_tick() runs first in every coordinator tick; log: caller's list.
    trace(thread, kind, t0, t1): optional, see traced(); threads: coordinator, sensor, load cell, file writer.
    """
    stop, dwelling, fifo, lock = threading.Event(), threading.Event(), queue.Queue(), threading.Lock()
    cancel = cancel or threading.Event()
    mover = mover or (lambda rig, z: start_and_wait(rig, z, cancel))
    w = rig.world
    log = [] if log is None else log
    forces = collections.deque(maxlen=50)  # (t, force) history at 1 kHz
    reg = {"sensor": NO_FRAME}  # latest (value, frame time, frame number, read time); one assignment is GIL-atomic

    def poll_load_cell():
        lim = RateLimiter(1000.0)
        while not stop.is_set():
            with lock:
                forces.append((rig.world.now(), traced(trace, w, "load cell", "reading the load cell", rig.load_cell.read)))
            traced(trace, w, "load cell", "sleeping", lim.wait, stop)  # 1 ms between reads: the row is nearly all sleep

    def poll_sensor():
        while not stop.is_set():
            value, t_frame, k = traced(trace, w, "sensor", "waiting for the sensor frame", rig.sensor.read)
            reg["sensor"] = (value, t_frame, k, w.now())

    def coordinate():
        lim, t_prev, last_frame = RateLimiter(rate_hz, rig.world.t0 + phase_ms * 1e-3), 0.0, 0
        while not stop.is_set():
            if on_tick:
                traced(trace, w, "coordinator", "redrawing the plot", on_tick)
            sensor = reg["sensor"]  # the sensor first, so that its time is never later than the row's (framework _emit)
            t_tick = rig.world.now()
            with lock:
                history = list(forces)
            if history and (log_in_dwell or not dwelling.is_set()):
                row = make_row(t_tick, history, sensor)
                if not new_frames_only or row[3] > last_frame:
                    fifo.put(row)
                    last_frame = max(last_frame, row[3])
            # a frame counts as producible-while-logging once one full tick has passed since it arrived: a frame landing
            # on the last tick races the sensor thread's wake-up and cannot be logged
            rig.t_stop, t_prev = t_prev, t_tick
            traced(trace, w, "coordinator", "sleeping", lim.wait, stop)

    def write():
        for row in iter(fifo.get, None):
            traced(trace, w, "file writer", "writing to disk", rig.disk.write, row if log_seq else row[:3])
            log.append(row)

    workers = [threading.Thread(target=f, daemon=True) for f in (poll_load_cell, poll_sensor, coordinate)]
    writer = threading.Thread(target=write, daemon=True)
    for th in workers + [writer]:
        th.start()
    for z, hold in steps:
        if cancel.is_set():
            break
        mover(rig, z)  # default: non-blocking start, as in V1
        dwelling.set()
        cancel.wait(hold)
        dwelling.clear()
    stop.set()
    for th in workers:
        th.join()
    fifo.put(None)
    writer.join()
    return log


def v2_threaded(rig, steps, **kw):
    """V1 + threads only: each device polled at its native rate, the disk behind a FIFO; rows stamped on read."""
    return threaded(rig, steps, stamped_on_read, **kw)


def v3_stamped_at_read(rig, steps, sensor_time=False, **kw):
    """V2 with the fix of the framework's pull request #3: each row also gets the time the frame was read and its number.

    sensor_time: the simulator setting "The sensor sends its own time": that time stamp is the sensor's (story beat T15)."""
    return threaded(rig, steps, stamped_by_sensor if sensor_time else stamped_at_read, **{"log_seq": True, **kw})


def v4_double_tick(rig, steps, sensor_time=False, **kw):
    """V3 + the coordinator ticks at twice the logging rate, so every frame is seen whatever the phase. A tick writes a
    row only for a frame that is new, so each frame is in the file once."""
    return threaded(rig, steps, stamped_by_sensor if sensor_time else stamped_at_read,
                    **{"rate_hz": 2 * kw.pop("rate_hz", 50.0), "log_seq": True, "new_frames_only": True, **kw})


THREADED = ("V2", "V3", "V4")  # the versions with threads: they take the per-tick hook (on_tick) and the redraw switch
VERSIONS = {"V0": v0_naive, "V1": v1_polling, "V2": v2_threaded, "V3": v3_stamped_at_read, "V4": v4_double_tick}
