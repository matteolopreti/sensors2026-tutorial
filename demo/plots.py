"""Vector figures for the projector: timeline, sensor-vs-force loops, datasheet table."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import metrics

plt.rcParams.update({"font.size": 18, "lines.linewidth": 2})
COLORS = {"V0": "tab:red", "V1": "tab:green", "V2": "tab:orange", "V3": "tab:blue", "V4": "tab:purple"}


def _style(v):
    big = v == "V0"  # naive has ~8 points: draw them large
    return {"color": COLORS.get(v, "k"), "marker": "o" if big else ".", "ms": 10 if big else 4, "ls": "none"}


def _save(fig, out, stem):
    fig.tight_layout()
    for ext in ("svg", "pdf"):
        fig.savefig(out / f"{stem}.{ext}")
    plt.close(fig)


def save_all(out, runs, sheets, truth, bugs):
    """runs: version -> (rig, log); sheets: version -> (datasheet, truth); bugs: version -> failed checks."""
    fig, axes = plt.subplots(len(runs), 1, figsize=(14, 3 * len(runs) + 1), sharex=True, sharey=True,
                             squeeze=False)
    for ax, (v, (rig, log)) in zip(axes[:, 0], runs.items()):
        tr, lg = np.array(rig.world.truth), np.array(log)
        for a, b, _ in rig.world.ramps:
            ax.axvspan(a, b, color="0.85")
        ax.plot(tr[:, 0], tr[:, 1], color="0.3", lw=1.5)
        ax.plot(lg[:, 0], lg[:, 1], label=f"{v} ({len(lg)} samples)", **_style(v))
        ax.set_ylabel("force (N)")
        ax.legend(loc="upper right")
    axes[-1, 0].set_xlabel("time (s)   [line = true force, grey = stage ramps]")
    _save(fig, out, "timeline")

    fig, ax = plt.subplots(figsize=(10, 8))
    tr = np.array(next(iter(runs.values()))[0].world.truth)
    ax.plot(tr[:, 1], tr[:, 2], color="0.3", lw=1.5, label="truth (noiseless 1 kHz)")
    for v, (_, log) in reversed(list(runs.items())):
        lg = np.array(log)
        ax.plot(lg[:, 1], lg[:, 2], label=v, **_style(v))
    ax.set(xlabel="force (N)", ylabel="sensor (V)")
    ax.legend()
    _save(fig, out, "loops")

    fig, ax = plt.subplots(figsize=(18, 8))
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
