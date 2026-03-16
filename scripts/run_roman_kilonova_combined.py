#!/usr/bin/env python3
"""Run the combined AT2017gfo-like and red kilonova figure and save under figures/."""

from pathlib import Path

import numpy as np

from lightcurves.roman_kilonova_combined import run_roman_kilonova_combined

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURES_DIR / "roman_kilonova_combined.png"
    times = np.linspace(0.05, 30.0, 300)
    fig = run_roman_kilonova_combined(times=times, save_path=str(save_path))
    print(f"Saved {save_path}")
