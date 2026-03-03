#!/usr/bin/env python3
"""Run the red (single-component) kilonova example and save the plot.

Uses one_component_kilonova_model with parameters from the red (third) component
of the Villar+2017 AT2017gfo-like model. Output is separate from roman_kilonova_example.
"""

from pathlib import Path

import numpy as np

from lightcurves.roman_kilonova_red import run_roman_kilonova_red

# Output directory for figures (repo root figures/ when run from repo root)
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURES_DIR / "roman_kilonova_red_example.png"
    times = np.linspace(0.05, 30.0, 300)
    fig = run_roman_kilonova_red(times=times, save_path=str(save_path))
    print(f"Saved {save_path}")
