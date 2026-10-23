"""Vector figures for the projector: timeline, sensor-vs-force loops, datasheet table."""
import matplotlib
import numpy as np
from matplotlib.figure import Figure  # no pyplot: importing this module must not choose a backend (the app needs a window)

import metrics

# Palette A "Hide & Kapton" (presentation-kit/palette/palettes.json); sky, stone are its data colours 3 and 4
PALETTE = {"paper": "#F3F4F2", "surface": "#E6E4E0", "ink": "#2B2724", "muted": "#5E5853", "primary": "#3F4A5A",
           "accent": "#A8541A", "line": "#CFCBC5", "sky": "#8FA3B8", "stone": "#8A817A"}
matplotlib.rcParams.update({
    "font.size": 18, "lines.linewidth": 2,
    "figure.facecolor": PALETTE["paper"], "savefig.facecolor": PALETTE["paper"], "axes.facecolor": PALETTE["paper"],
    "text.color": PALETTE["ink"], "axes.labelcolor": PALETTE["ink"], "axes.edgecolor": PALETTE["muted"],
    "xtick.color": PALETTE["ink"], "ytick.color": PALETTE["ink"]})
COLORS = {"V0": PALETTE["stone"], "V1": PALETTE["sky"], "V2": PALETTE["accent"], "V3": PALETTE["primary"], "V4": PALETTE["ink"],
          "A": PALETTE["accent"], "B": PALETTE["primary"]}


def _style(v):
    big = v == "V0"  # naive has ~8 points: draw them large
    return {"color": COLORS.get(v, PALETTE["ink"]), "marker": "o" if big else ".", "ms": 10 if big else 4, "ls": "none"}


def _save(fig, out, stem):
    fig.tight_layout()
    for ext in ("svg", "pdf"):
        fig.savefig(out / f"{stem}.{ext}")


def save_all(out, runs, sheets, truth, bugs):
    """runs: version -> (rig, log); sheets: version -> (datasheet, truth); bugs: version -> failed checks."""
    fig = Figure(figsize=(14, 3 * len(runs) + 1))
    axes = fig.subplots(len(runs), 1, sharex=True, sharey=True, squeeze=False)
    for ax, (v, (rig, log)) in zip(axes[:, 0], runs.items()):
        tr, lg = np.array(rig.world.truth), np.array(log)
        for a, b, _ in rig.world.ramps:
            ax.axvspan(a, b, color=PALETTE["line"])
        ax.plot(tr[:, 0], tr[:, 1], color=PALETTE["muted"], lw=1.5)
        ax.plot(lg[:, 0], lg[:, 1], label=f"{v} ({len(lg)} rows)", **_style(v))
        ax.set_ylabel("force (N)")
        ax.legend(loc="upper right")
    axes[-1, 0].set_xlabel("time (s)   [line = true force, grey = stage ramps]")
    _save(fig, out, "timeline")

    fig = Figure(figsize=(10, 8))
    ax = fig.subplots()
    tr = np.array(next(iter(runs.values()))[0].world.truth)
    ax.plot(tr[:, 1], tr[:, 2], color=PALETTE["muted"], lw=1.5, label="truth (noiseless 1 kHz)")
    for v, (_, log) in reversed(list(runs.items())):
        lg = np.array(log)
        ax.plot(lg[:, 1], lg[:, 2], label=v, **_style(v))
    ax.set(xlabel="force (N)", ylabel="sensor (V)")
    ax.legend()
    _save(fig, out, "loops")

    fig = Figure(figsize=(18, 8))
    ax = fig.subplots()
    ax.axis("off")
    rows = [[metrics.fmt(truth[k])] + [metrics.fmt(ds[k]) for ds, _ in sheets.values()] for k in truth]
    rows.append([""] + [bugs[v].replace(", ", "\n") for v in sheets])
    tab = ax.table(cellText=rows, rowLabels=[*truth, "bug exposed"], colLabels=["truth", *sheets], loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(14)
    tab.scale(1, 1.5)
    for c in range(len(sheets) + 1):
        tab[len(rows), c].set_height(tab[len(rows), c].get_height() * 6)
    _save(fig, out, "datasheet")


def save_lane(out, stem, ylabel, traces, shade=(), vline=None):
    """Lane figure, run A and run B on one axes. traces: (label, t, y, plot style); shade: (t0, t1) spans."""
    fig = Figure(figsize=(14, 5))
    ax = fig.subplots()
    for a, b in shade:
        ax.axvspan(a, b, color=PALETTE["line"])
    for label, t, y, style in traces:
        ax.plot(t, y, label=label, **style)
    if vline is not None:
        ax.axvline(vline, color=PALETTE["ink"], ls="--", label="STOP pressed")
    ax.set(xlabel="time (s)   [grey = stage ramps]", ylabel=ylabel)
    ax.legend(loc="upper right")
    _save(fig, out, stem)
