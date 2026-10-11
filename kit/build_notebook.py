"""Build kit/demo_colab.ipynb from demo/*.py so the notebook cannot drift from the code.

Run:  uv run --with nbformat python kit/build_notebook.py
"""
from pathlib import Path

import nbformat.v4 as nbf
from nbformat import write

ROOT = Path(__file__).resolve().parent.parent
FILES = ("loops.py", "metrics.py", "plots.py", "naive_vs_threaded.py")

INTRO = """# One acquisition loop, five versions: same sensor, different datasheet

A simulated soft sensor is indented by a simulated stage, and the same measurement is recorded by five
versions of the acquisition loop (V0 naive, V1 polling, V2 threaded, V3 frame-stamped, V4 double-rate tick).
Each version fixes the previous one's bug, and the datasheet you would quote changes with it. The demo uses real
threads, so it runs in real time (about 75 s for the fast run).

**Your numbers will differ from the slides: that is the point.** Thread wake-up jitter belongs to the host,
so the exact percentages move from machine to machine. The pattern (which version loses frames, which
hides it) does not.

Run the cells top to bottom. No uploads needed."""

FILES_MD = "## 1. The code\nEach cell writes one source file, byte for byte as `demo/<file>`. They are generated from `demo/*.py`; do not edit them here."

RUN_MD = """## 2. Run all five versions (`--fast`, about 75 s)
Prints the datasheet table (truth vs V0-V4) and the checks each version still fails, then shows the plots."""
RUN = """!python naive_vs_threaded.py --fast --seed 1 --out out
from IPython.display import SVG, display
for name in ("timeline", "loops", "datasheet"):
    display(SVG(filename=f"out/{name}.svg"))"""

SWEEP_MD = """## 3. Phase sweep
Frame loss depends on where the 50 Hz coordinator tick falls relative to the 50 Hz sensor frame clock.
`--phase-sweep --fast` runs V2, V3 and V4 at phases 0 to 18 ms (about 2.5 minutes in total). Look at true vs visible loss:
V2 loses frames its log cannot show; V3 shows its loss; V4 stays near zero at every phase."""
SWEEP = "!python naive_vs_threaded.py --phase-sweep --sweep-versions V2,V3,V4 --fast --seed 1 --out out"

TRY_MD = """## 4. Try this
Change one thing and compare with the run above:
- `--phase-ms 14` moves the tick away from the worst phase (4 ms).
- `--no-stalls` removes the simulated disk stalls (they cause V1's jitter).
- `--seed 2` draws different noise.
- `--versions V2,V3` runs only those versions."""
TRY = "!python naive_vs_threaded.py --fast --versions V1,V2,V3 --phase-ms 14 --no-stalls --seed 2 --out out_try"

HOST_MD = "## 5. Your host\nPrint this next to your numbers when you compare with someone else's."
HOST = """import os, platform
print(platform.platform())
print(platform.python_implementation(), platform.python_version())
print("CPUs:", os.cpu_count())"""


def code(src, cid):
    c = nbf.new_code_cell(src)
    c["id"] = cid
    return c


def md(src, cid):
    c = nbf.new_markdown_cell(src)
    c["id"] = cid
    return c


def build():
    cells = [md(INTRO, "intro"), md(FILES_MD, "files")]
    for f in FILES:
        body = (ROOT / "demo" / f).read_text()
        cells.append(code(f"%%writefile {f}\n" + body, "src-" + f[:-3].replace("_", "-")))
    cells += [md(RUN_MD, "run-md"), code(RUN, "run"), md(SWEEP_MD, "sweep-md"), code(SWEEP, "sweep"),
              md(TRY_MD, "try-md"), code(TRY, "try"), md(HOST_MD, "host-md"), code(HOST, "host")]
    nb = nbf.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}
    return nb


if __name__ == "__main__":
    out = ROOT / "kit" / "demo_colab.ipynb"
    with open(out, "w") as fh:
        write(build(), fh)
    print(f"wrote {out}")
