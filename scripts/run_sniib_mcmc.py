#!/usr/bin/env python3
"""
Run MCMC to fit SN IIb r-band light curve to AT2017gfo LSST r-band (days 0--10).
Saves chain, best-fit parameters, and result JSON under mcmc_results/.
"""

from pathlib import Path

from lightcurves.sniib_mcmc_fit import run_mcmc

RESULTS_DIR = Path(__file__).resolve().parent.parent / "mcmc_results"

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    chain, best_theta, result = run_mcmc(
        n_walkers=32,
        n_steps=1500,
        n_burn=300,
        seed=42,
        save_dir=RESULTS_DIR,
    )
    print("Best-fit SN IIb parameters (r-band fit to AT2017gfo r-band, days 0--10):")
    for k, v in result["best_theta_dict"].items():
        print(f"  {k}: {v}")
    print(f"  best_ln_prob: {result['best_ln_prob']}")
