#!/usr/bin/env python3
"""Generate comparative light curve figure (AGN IR, TXS 0506+056, SN Ic-BL, reddened SN) and save under figures/."""

from pathlib import Path

import numpy as np

from lightcurves.agn_blazar_sn_lightcurves import run_agn_blazar_sn_lightcurves

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURES_DIR / "agn_blazar_sn_lightcurves.png"
    times = np.linspace(-50.0, 365.0, 500)
    fig = run_agn_blazar_sn_lightcurves(times=times, save_path=str(save_path))
    print(f"Saved {save_path}")
