"""
MCMC fit of SN IIb (shock_cooling_and_arnett) LSST r-band light curve to match
AT2017gfo LSST r-band from days 0 to 10.

Uses emcee to sample the SN IIb parameter space; "data" are AT2017gfo r-band
magnitudes over the chosen time range.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# NumPy 2.0 compatibility for redback
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid


# Time range for fitting (days)
T_MIN, T_MAX = 0.0, 10.0
# Number of data points (AT2017gfo r-band evaluated at these times)
N_DATA = 100  # ~0.05 to 10 days

# SN IIb parameters to fit (order matches walker position)
FIT_PARAM_NAMES = [
    "log10_mass",
    "log10_radius",
    "log10_energy",
    "f_nickel",
    "mej",
    "vej",
    "kappa",
    "kappa_gamma",
    "temperature_floor",
]

# Uniform prior bounds [low, high] for each parameter
# Widened where previous MCMC best-fit approached bounds (0–10 day fit)
PRIOR_BOUNDS = {
    "log10_mass": (-3.0, 0.0),            # was -0.5; best ~-1.66 near upper
    "log10_radius": (10.5, 13.0),         # was 11.0; best ~11.28 near lower
    "log10_energy": (49.0, 52.0),         # was 51.5; best ~50.7, room above
    "f_nickel": (0.001, 0.15),            # was 0.005; best ~0.0096 near lower
    "mej": (0.1, 2.5),                    # was 0.2; best ~0.20 at lower
    "vej": (4000.0, 15000.0),             # was 12000; best ~11997 at upper
    "kappa": (0.1, 2.0),                  # was 0.2; best ~0.20 at lower
    "kappa_gamma": (0.001, 0.05),         # was 0.005; best ~0.0051 at lower
    "temperature_floor": (2000.0, 4500.0),  # was 2500; best ~2927, room below
}


def get_at2017gfo_rband_data(
    redshift: float,
    t_min: float = T_MIN,
    t_max: float = T_MAX,
    n_data: int = N_DATA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (times, magnitudes, magnitude_errors) for AT2017gfo LSST r-band
    in the given time range. Used as the "data" to fit with the SN IIb model.
    """
    from lightcurves.roman_kilonova import (
        GW190814_REDSHIFT,
        LIMIT_MAG_5SIGMA,
        get_at2017gfo_like_parameters,
        magnitude_error_poisson,
        _ensure_roman_bands_registered,
    )
    from redback.model_library import all_models_dict

    _ensure_roman_bands_registered()
    z = redshift if redshift is not None else GW190814_REDSHIFT
    params = get_at2017gfo_like_parameters(redshift=z)
    model = all_models_dict["three_component_kilonova_model"]

    times = np.linspace(max(0.05, t_min), t_max, n_data)
    mags = np.atleast_1d(
        model(times, bands="lsstr", output_format="magnitude", **params)
    ).flatten()
    limit_mag = LIMIT_MAG_5SIGMA.get("lsstr", 25.7)
    mag_errs = np.array([magnitude_error_poisson(m, limit_mag) for m in mags])
    return times, mags, mag_errs


def sniib_model_mags(
    times: np.ndarray,
    theta: np.ndarray,
    redshift: float,
    bands: str = "lsstr",
) -> np.ndarray:
    """
    SN IIb (shock_cooling_and_arnett) r-band magnitudes at `times`.
    `theta` is the vector of fitted parameters in order FIT_PARAM_NAMES.
    """
    from redback.model_library import all_models_dict

    model = all_models_dict["shock_cooling_and_arnett"]
    kwargs = dict(
        redshift=redshift,
        output_format="magnitude",
        bands=bands,
        nn=6.0,
        delta=1.0,
    )
    for i, name in enumerate(FIT_PARAM_NAMES):
        kwargs[name] = np.float64(theta[i])
    return np.atleast_1d(model(times, **kwargs)).flatten()


def ln_prior(theta: np.ndarray) -> float:
    """Log uniform prior; -inf if outside bounds."""
    for i, name in enumerate(FIT_PARAM_NAMES):
        low, high = PRIOR_BOUNDS[name]
        if not (low <= theta[i] <= high):
            return -np.inf
    return 0.0


def ln_likelihood(
    theta: np.ndarray,
    times: np.ndarray,
    mag_data: np.ndarray,
    mag_err: np.ndarray,
    redshift: float,
) -> float:
    """Gaussian log-likelihood in magnitude."""
    try:
        mag_model = sniib_model_mags(times, theta, redshift)
    except Exception:
        return -np.inf
    if not np.all(np.isfinite(mag_model)):
        return -np.inf
    chi2 = np.sum(((mag_data - mag_model) / mag_err) ** 2)
    return -0.5 * chi2


def ln_prob(
    theta: np.ndarray,
    times: np.ndarray,
    mag_data: np.ndarray,
    mag_err: np.ndarray,
    redshift: float,
) -> float:
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return lp
    return lp + ln_likelihood(theta, times, mag_data, mag_err, redshift)


def run_mcmc(
    redshift: float | None = None,
    n_walkers: int = 32,
    n_steps: int = 1500,
    n_burn: int = 300,
    seed: int | None = 42,
    save_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Run emcee MCMC to fit SN IIb r-band to AT2017gfo r-band (days 0--10).

    Returns
    -------
    chain : np.ndarray
        Shape (n_steps - n_burn, n_walkers, n_params).
    best_theta : np.ndarray
        Parameter vector with highest ln_prob in the chain.
    result : dict
        Keys: "times", "mag_data", "mag_err", "redshift", "best_theta_dict",
        "mean_theta", "median_theta", "best_ln_prob".
    """
    import emcee
    from lightcurves.roman_kilonova import GW190814_REDSHIFT

    z = redshift if redshift is not None else GW190814_REDSHIFT
    times, mag_data, mag_err = get_at2017gfo_rband_data(z)

    n_params = len(FIT_PARAM_NAMES)
    # Initial positions: center of prior with small scatter
    centers = np.array(
        [0.5 * (PRIOR_BOUNDS[n][0] + PRIOR_BOUNDS[n][1]) for n in FIT_PARAM_NAMES]
    )
    widths = np.array(
        [0.1 * (PRIOR_BOUNDS[n][1] - PRIOR_BOUNDS[n][0]) for n in FIT_PARAM_NAMES]
    )
    rng = np.random.default_rng(seed)
    p0 = centers + widths * rng.standard_normal((n_walkers, n_params))

    def log_prob(theta):
        return ln_prob(theta, times, mag_data, mag_err, z)

    sampler = emcee.EnsembleSampler(n_walkers, n_params, log_prob)
    sampler.run_mcmc(p0, n_steps, progress=True)

    chain = sampler.get_chain(discard=n_burn)
    flat = chain.reshape(-1, n_params)
    ln_probs = sampler.get_log_prob(discard=n_burn).reshape(-1)
    best_idx = np.argmax(ln_probs)
    best_theta = flat[best_idx]
    mean_theta = np.mean(flat, axis=0)
    median_theta = np.median(flat, axis=0)

    result = {
        "times": times.tolist(),
        "mag_data": mag_data.tolist(),
        "mag_err": mag_err.tolist(),
        "redshift": z,
        "best_theta_dict": {n: float(best_theta[i]) for i, n in enumerate(FIT_PARAM_NAMES)},
        "mean_theta_dict": {n: float(mean_theta[i]) for i, n in enumerate(FIT_PARAM_NAMES)},
        "median_theta_dict": {n: float(median_theta[i]) for i, n in enumerate(FIT_PARAM_NAMES)},
        "best_ln_prob": float(ln_probs[best_idx]),
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "sniib_mcmc_chain.npy", chain)
        np.save(save_dir / "sniib_mcmc_best_theta.npy", best_theta)
        with open(save_dir / "sniib_mcmc_result.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved chain and result to {save_dir}")

    return chain, best_theta, result
