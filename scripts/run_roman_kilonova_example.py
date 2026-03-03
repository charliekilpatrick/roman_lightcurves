#!/usr/bin/env python3
"""Run the AT2017gfo-like kilonova example in Roman F129 and F158 and save the plot."""

from pathlib import Path

import numpy as np

from lightcurves.roman_kilonova import run_roman_kilonova

# Output directory for figures (repo root figures/ when run from repo root)
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURES_DIR / "roman_kilonova_example.png"
    times = np.linspace(0.05, 30.0, 300)
    fig = run_roman_kilonova(times=times, save_path=str(save_path))
    print(f"Saved {save_path}")
