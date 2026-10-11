"""One acquisition loop, version by version; each version fixes the previous one's bug.

Every version logs rows (t, force, sensor) and writes each row through rig.disk.
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


def v0_naive(rig, targets, dwell):
    """AIS SI Listing 1: blocking move, sleep through the dwell, one read pair per step."""
    log = []
    for z in targets:
        rig.stage.move_to(z)
        time.sleep(dwell)
        f = rig.load_cell.read()
        s, _, k = rig.sensor.read()  # serial.read(): blocks until the next 50 Hz frame
        row = (rig.world.now(), f, s, k)  # k: simulator frame index, never logged by V0-V2
        rig.disk.write(row[:3])
        log.append(row)
    rig.t_stop = rig.world.now()  # logging ends here: frames produced later are not lost frames
    return log


def v1_polling(rig, targets, dwell, rate_hz=50.0):
    """V0 without the blocking move and sleep: one thread reads every tick through ramps and dwells."""
    log, lim = [], RateLimiter(rate_hz)
    for z in targets:
        rig.stage.start(z)  # V0: move_to(z) blocked
        t_hold = None
        while t_hold is None or rig.world.now() < t_hold + dwell:  # V0: time.sleep(dwell)
            if t_hold is None and rig.stage.position() == z:
                t_hold = rig.world.now()
            f = rig.load_cell.read()
            s, _, k = rig.sensor.read()
            row = (rig.world.now(), f, s, k)
            rig.disk.write(row[:3])  # a disk stall still stops every read
            log.append(row)
            lim.wait()
    rig.t_stop = rig.world.now()
    return log


PHASE_MS = 4.0  # pinned start offset (ms) of the tick vs the frame clock: tick coincides with frame arrival -> loss peaks (host-dependent)


def stamped_on_read(now, forces, sensor):
    """V2 row: both channels stamped at the tick; the sensor value is up to a frame + latency old.

    Lag sign: positive = sensor channel older than the force channel.
    """
    return now, forces[-1][1], sensor[0], sensor[2]


def stamped_at_frame(now, forces, sensor):
    """V3 row: stamped at the sensor frame time (+ frame seq), paired with the force interpolated at that time."""
    t_frame = sensor[1]
    ts, fs = zip(*forces)
    return t_frame, float(np.interp(t_frame, ts, fs)), sensor[0], sensor[2]


def threaded(rig, targets, dwell, make_row, rate_hz=50.0, log_in_dwell=True, phase_ms=PHASE_MS, log_seq=False):
    """Device pollers -> latest-value registers; fixed-rate coordinator -> FIFO -> logger thread.

    phase_ms: coordinator ticks fall on world.t0 + phase + n/rate; sensor frames on n * 20 ms (+ 4 ms latency).
    log_seq: the frame seq is a logged column (V3+); else it stays internal, for the true-loss count.
    log_in_dwell=False is the planted defect for check-the-checker.
    """
    stop, dwelling, fifo, log, lock = threading.Event(), threading.Event(), queue.Queue(), [], threading.Lock()
    forces = collections.deque(maxlen=50)  # (t, force) history at 1 kHz
    reg = {"sensor": (0.0, 0.0, 0)}  # latest (value, frame_time, frame index); a single assignment is GIL-atomic

    def poll_load_cell():
        lim = RateLimiter(1000.0)
        while not stop.is_set():
            with lock:
                forces.append((rig.world.now(), rig.load_cell.read()))
            lim.wait(stop)

    def poll_sensor():
        while not stop.is_set():
            reg["sensor"] = rig.sensor.read()

    def coordinate():
        lim, t_last, t_prev = RateLimiter(rate_hz, rig.world.t0 + phase_ms * 1e-3), -1.0, 0.0
        while not stop.is_set():
            t_tick = rig.world.now()
            with lock:
                history = list(forces)
            if history and (log_in_dwell or not dwelling.is_set()):
                row = make_row(rig.world.now(), history, reg["sensor"])
                if row[0] > t_last:  # one row per timestamp (V3: per new frame)
                    fifo.put(row)
                    t_last = row[0]
            # a frame counts as producible-while-logging once one full tick has passed since it arrived: a frame landing
            # on the last tick races the sensor thread's wake-up and cannot be logged
            rig.t_stop, t_prev = t_prev, t_tick
            lim.wait(stop)

    def write():
        for row in iter(fifo.get, None):
            rig.disk.write(row if log_seq else row[:3])
            log.append(row)

    workers = [threading.Thread(target=f, daemon=True) for f in (poll_load_cell, poll_sensor, coordinate)]
    writer = threading.Thread(target=write, daemon=True)
    for th in workers + [writer]:
        th.start()
    for z in targets:
        t_ramp_end = rig.stage.start(z)  # non-blocking, as in V1
        time.sleep(max(t_ramp_end - rig.world.now(), 0.0))
        dwelling.set()
        time.sleep(dwell)
        dwelling.clear()
    stop.set()
    for th in workers:
        th.join()
    fifo.put(None)
    writer.join()
    return log


def v2_threaded(rig, targets, dwell, **kw):
    """V1 + threads only: each device polled at its native rate, the disk behind a FIFO; rows stamped on read."""
    return threaded(rig, targets, dwell, stamped_on_read, **kw)


def v3_frame_stamped(rig, targets, dwell, **kw):
    """V2 with rows stamped at the sensor frame time."""
    return threaded(rig, targets, dwell, stamped_at_frame, **{"log_seq": True, **kw})


def v4_double_tick(rig, targets, dwell, **kw):
    """V3 + coordinator tick at 2x the frame rate, so every frame is seen whatever the phase."""
    return threaded(rig, targets, dwell, stamped_at_frame, **{"rate_hz": 100.0, "log_seq": True, **kw})


VERSIONS = {"V0": v0_naive, "V1": v1_polling, "V2": v2_threaded, "V3": v3_frame_stamped, "V4": v4_double_tick}
