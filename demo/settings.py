"""The run controls of the app: one table, each control with its default and range, and where the framework has them.

The framework is tactile_characterization_framework at commit 84668db. A control that the framework has starts at the
framework's default; a control that only the simulated rig has (SIMULATED) starts at the value the frozen numbers use.
No window and no matplotlib in here: the app, the tests and the freeze read the same table.
"""
from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass

FRAMEWORK = "tactile_characterization_framework @ 84668db"


@dataclass(frozen=True)
class Control:
    key: str
    label: str  # plain words, as on screen
    default: float | bool
    lo: float | None = None  # None: a tick box
    hi: float | None = None
    whole: bool = False  # a whole number
    source: str = ""  # where the framework has it: file:line (what the default and the range are copied from)


CONTROLS = (
    Control("rate", "Logging rate (Hz)", 50, 1, 200, True, "gui/main_window.py:290-291 (range 1-200, value 50)"),
    # The framework's depth range is 0-20 mm (gui/main_window.py:283). 0 mm makes a run in which the stage never moves:
    # the demo's lowest depth is 0.1 mm.
    Control("depth", "Depth (mm)", 0.5, 0.1, 20.0, False,
            "gui/main_window.py:283 (range 0-20) and config/hardware.json:25 experiment.depth = 0.5"),
    Control("speed", "Speed (mm/s)", 0.1, 0.01, 5.0, False,
            "gui/main_window.py:284 (range 0.01-5) and config/hardware.json:26 experiment.speed = 0.1"),
    Control("hold", "Hold (s)", 0.5, 0.0, 60.0, False,
            "gui/main_window.py:285 (range 0-60) and config/hardware.json:27 experiment.hold_time = 0.5"),
    Control("cycles", "Cycles", 5, 1, 100, True, "gui/main_window.py:298-299 (range 1-100, value 5)"),
    Control("csv", "CSV file", True, source="gui/main_window.py:322 (ticked)"),
    Control("database", "Database file", False, source="gui/main_window.py:324 (not ticked)"),
    Control("window", "Plot window (s)", 10, 5, 300, True, "gui/main_window.py:335-336 (range 5-300, value 10)"),
)
SIMULATED = (  # the simulated rig: not in the framework; the defaults are what NUMBERS.md and LANES.md were measured with
    Control("fps", "Sensor frames per second", 50, 1, 200, True, "none: simulator (naive_vs_threaded.SoftSensor, period 20 ms)"),
    Control("busy", "Computer busy", False, source="none: simulator (other programs use every processor core)"),
    Control("slow_disk", "Slow disk", False, source="none: simulator (the file writer stalls more often and longer)"),
    Control("sensor_time", "The sensor sends its own time", False, source="none: simulator (V3 and V4 only; story beat T15)"),
)
ALL = CONTROLS + SIMULATED
BY_KEY = {c.key: c for c in ALL}

SHORT_RUN = {"cycles": 1, "hold": 0.3}  # the "Short run" button: 1 cycle (2 load-hold-unload legs), real speed, 15 rows in a hold
NORMAL_DISK = {"p_stall": 0.05, "stall_s": (0.02, 0.04)}  # the disk of the frozen numbers (README: an assumption)
SLOW_DISK = {"p_stall": 0.25, "stall_s": (0.04, 0.08)}  # stalls five times as often, twice as long (an assumption)
MESSAGE_NOT_DEFAULT = "Not the defaults: the numbers on the slides do not apply."


def defaults() -> dict:
    return {c.key: c.default for c in ALL}


def differing(values: dict) -> list[str]:
    """Keys of the settings that are not at their default."""
    return [c.key for c in ALL if values[c.key] != c.default]


def parse(key: str, text: str):
    """The value of the control `key` from the text typed in its box. ValueError says in plain words what is wrong."""
    c = BY_KEY[key]
    try:
        value = float(text.replace(",", "."))
    except ValueError:
        raise ValueError(f"{c.label}: type a number.") from None
    if c.whole and value != int(value):
        raise ValueError(f"{c.label}: type a whole number.")
    if not c.lo <= value <= c.hi:
        raise ValueError(f"{c.label}: between {c.lo:g} and {c.hi:g}.")
    return int(value) if c.whole else value


def leg_s(values: dict) -> float:
    """Seconds of one ramp: depth over speed."""
    return values["depth"] / values["speed"]


def run_s(values: dict) -> float:
    """Seconds of a whole run: cycles + 1 legs of load ramp, hold, unload ramp (framework core/coordinator.py:174-176)."""
    return (values["cycles"] + 1) * (2 * leg_s(values) + values["hold"])


def rig_options(values: dict) -> dict:
    """The keyword arguments of naive_vs_threaded.build_rig that these settings change."""
    return {"sensor_fps": values["fps"], "csv": values["csv"], "database": values["database"],
            **(SLOW_DISK if values["slow_disk"] else NORMAL_DISK)}


@contextmanager
def busy_machine(on: bool, processes: int | None = None):
    """While on: one program per processor core that does nothing but calculate, so every wake-up can come late."""
    workers = []
    try:
        if on:
            workers = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(processes or os.cpu_count() or 1)]
        yield workers
    finally:
        for w in workers:
            w.terminate()
        for w in workers:
            w.wait()
