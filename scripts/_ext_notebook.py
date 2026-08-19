"""Shared scaffolding for the `02-ext-research/notebooks/` didactic series.

The five builders (`build_ext_00_notebook.py` … `build_ext_04_notebook.py`) all
emit their `.ipynb` through here, so that the palette, the kernel pin and the
house rules are identical across the series rather than five near-copies.

Nothing scientific lives in this file. It is cell plumbing plus one string:
`PREAMBLE`, the setup cell every notebook opens with.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "02-ext-research" / "notebooks"

KERNEL_NAME = "ext-research"
KERNEL_DISPLAY = "Python (02-ext-research)"


class Notebook:
    """Accumulates cells and writes one `.ipynb`."""

    def __init__(self, filename: str):
        self.path = NB_DIR / filename
        self.cells: list = []

    def md(self, text: str) -> None:
        self.cells.append({
            "cell_type": "markdown", "id": f"md{len(self.cells):03d}",
            "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True),
        })

    def code(self, text: str) -> None:
        self.cells.append({
            "cell_type": "code", "id": f"cd{len(self.cells):03d}",
            "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True),
        })

    def write(self) -> None:
        nb = {
            "cells": self.cells,
            "metadata": {
                "kernelspec": {
                    "display_name": KERNEL_DISPLAY,
                    "language": "python",
                    "name": KERNEL_NAME,
                },
                "language_info": {
                    "name": "python",
                    "pygments_lexer": "ipython3",
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "nbconvert_exporter": "python",
                    "codemirror_mode": {"name": "ipython", "version": 3},
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {self.path.relative_to(ROOT)}  ({len(self.cells)} cells)")


# ---------------------------------------------------------------------------
# The header every notebook in the series carries, and the setup cell.
# ---------------------------------------------------------------------------

HOW_TO_RUN = r"""
### Running this notebook

Pinned to the kernel `ext-research` — the `.venv` beside the project. Register it
once, from the repository root:

```bash
02-ext-research/.venv/bin/python -m ipykernel install \
    --user --name ext-research --display-name "Python (02-ext-research)"
```

**It runs on a clean checkout with no data and no results.** `.gitignore` excludes
`results/` at every level, and the corpora are gigabytes behind author
applications, so nothing here reads `data/` or `results/`. Measured numbers are
**hard-coded as verified constants** in the cell below, each carrying the file it
came from and the line of
`01-info-propagation/overview/EVIDENCE-INDEX.md` that records it. Everything else
is computed live on synthetic data, in front of you.

The notebook is a **build artefact** of `scripts/build_ext_%(nn)s_notebook.py`.
Edit the builder, not the `.ipynb`, or your changes are lost on the next
regeneration.
"""

PREAMBLE = r'''
# --- Setup -----------------------------------------------------------------
# OMP_NUM_THREADS must be set BEFORE the first scikit-learn / numpy-BLAS import.
# On this machine the default OpenMP pool cost 133x wall time on a 4,770 x 12
# fit -- pure scheduling overhead, measured in bitacora/04_p2_temporal.md sect. 6.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

SEED = 42

# --- Palette ---------------------------------------------------------------
# One palette across all five notebooks. Colour-blind safe: validated with the
# dataviz skill's checker (worst adjacent-pair CVD delta-E 9.1, normal-vision
# 22.9, both above their floors). Hues are assigned in fixed slot order and
# never cycled; colour never carries identity alone -- every series is also
# labelled directly or in a legend.
C = {
    "blue":    "#2a78d6",   # slot 1 -- the thing under test
    "orange":  "#eb6834",   # slot 2 -- its comparison / the other class
    "aqua":    "#1baf7a",   # slot 3 -- a third series
    "yellow":  "#eda100",   # slot 4
    "magenta": "#e87ba4",   # slot 5
    "violet":  "#4a3aa7",   # slot 6
    "red":     "#e34948",   # reserved: floors, baselines, failures
    "grey":    "#8a8a86",   # recessive: grids, noise bands, context
    "ink":     "#0b0b0b",
}
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"], C["violet"]]

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c9c9c4", "axes.labelcolor": C["ink"],
    "axes.grid": True, "grid.color": "#e3e3de", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": C["ink"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2.0,
})

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 30)
np.set_printoptions(precision=4, suppress=True, linewidth=110)

print(f"numpy {np.__version__} | pandas {pd.__version__} | matplotlib {mpl.__version__}")
print(f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}  seed={SEED}")
'''


def license_block(licensed: list[str], not_licensed: list[str]) -> str:
    """The two closing sections every notebook in the series ends with."""
    lic = "\n".join(f"- {s}" for s in licensed)
    nlic = "\n".join(f"- {s}" for s in not_licensed)
    return (
        "## What this licenses\n\n" + lic +
        "\n\n## What this does NOT license\n\n" + nlic + "\n"
    )
