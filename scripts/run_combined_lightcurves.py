#!/usr/bin/env python3
"""Generate combined three-panel light curve figure and save under figures/."""

from pathlib import Path

from lightcurves.combined_lightcurves import run_combined_lightcurves

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURES_DIR / "combined_lightcurves.png"
    fig = run_combined_lightcurves(save_path=str(save_path))
    print(f"Saved {save_path}")
