# One loop, five versions

A sensor is only as good as the loop that reads it.

This is the hands-on demo of the IEEE SENSORS 2026 tutorial *From Naïve Loops to FAIR Datasheets* (Rotterdam, 25 October 2026). It gives you one simulated soft sensor, one indentation test, and five versions of the acquisition loop that records it. Each version fixes the bug of the one before. Each one also leaves its own mark on the datasheet you would publish.

## Run it

1. Get the files: **Code → Download ZIP**, then unzip. Or `git clone` this repository.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once. It brings Python and the packages with it.

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh                                     # macOS, Linux
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
   ```

3. From the repository folder:

   ```bash
   uv run demo/app.py
   ```

   A page opens in your browser. It works offline and talks only to your own computer. Ctrl+C in the terminal stops it.

No uv? [Open the notebook in Colab](https://colab.research.google.com/github/matteolopreti/sensors2026-tutorial/blob/main/kit/demo_colab.ipynb) instead. It needs only a Google account.

## What to try

Pick a version and press **Short run** (about 20 s). Watch *Where the time goes*: one row per thread, one block per wait. Then open **Guess first**, make your prediction before the next run, and see if the file agrees with you.

| | The loop | What goes wrong (full run, default settings) |
|---|---|---|
| V0 | Move, sleep, read: the script a paper's appendix warns against | 12 rows in 84 s. The ramps and holds are never recorded |
| V1 | One thread reads the sensor and writes the file | Slow disk writes open gaps of up to 70 ms |
| V2 | One thread per device and a queue: the framework used for the paper | 35 % of the sensor frames are lost, and the file cannot show it |
| V3 | V2 plus the read time and frame number on each row | Still 35 % lost, but now the file shows it |
| V4 | V3 with the coordinator waking twice per frame | No frame lost in 12 of 13 full runs |

## What is real and what is simulated

The sensor, the test rig and the slow disk are simulated. The loops, Python's threads, the clock and your operating system's scheduler are real: they run on your machine. So your numbers will differ from ours, which come from one Mac (`demo/out/freeze/NUMBERS.md`). The pattern should not.

## More

- `demo/README.md`: every setting, the measured numbers, and how they were frozen.
- `kit/handout/handout.pdf`: the one-page handout with the checks for your own rig.
- Tests: `cd demo && uv run --with pytest --with numpy --with matplotlib python -m pytest -q`
- The paper: [doi.org/10.1002/aisy.70436](https://doi.org/10.1002/aisy.70436). The framework: [tactile_characterization_framework](https://github.com/matteolopreti/tactile_characterization_framework).
