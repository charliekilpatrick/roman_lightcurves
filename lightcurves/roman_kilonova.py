"""
Generate an AT2017gfo-like (Villar+2017) three-component kilonova model and plot
light curves in Roman F129/F158 and LSST g/r/i bands.

Uses redback's three_component_kilonova_model and sncosmo-registered bandpasses.
References:
- Villar+2017: https://iopscience.iop.org/article/10.3847/2041-8213/aa9c84/pdf
- Rubin ToO 2024 Gold strategy: https://arxiv.org/abs/2411.04793
"""

from __future__ import annotations

import numpy as np
import sncosmo

from lightcurves.roman_bands import register_roman_bands

# Roman band names as registered with sncosmo
ROMAN_BANDS = ["roman_f129", "roman_f158"]

# LSST band names (sncosmo builtins: lsstg, lsstr, lssti)
LSST_BANDS = ["lsstg", "lsstr", "lssti"]

# Display labels and colors for all bands in plots
BAND_LABELS = {
    "roman_f129": "Roman/F129",
    "roman_f158": "Roman/F158",
    "lsstg": "LSST/g",
    "lsstr": "LSST/r",
    "lssti": "LSST/i",
}
BAND_COLORS = {
    "roman_f129": "orange",
    "roman_f158": "red",
    "lsstg": "blue",
    "lsstr": "green",
    "lssti": "darkgoldenrod",  # similar to gold but darker for contrast
}

# Roman: default observation epochs (days) to mark on the light curve
DEFAULT_OBS_TIMES = [7.0, 12.0, 19.0, 28.0]

# LSST Gold event strategy (Rubin ToO 2024, arxiv.org/abs/2411.04793):
# Night 0 = 3 scans in gri, each scan separated by 95 minutes; Nights 1–3 = one scan in r+i
_MIN_PER_DAY = 60.0 * 24.0
_NIGHT0_FIRST_SCAN_DAY = 0.5  # first scan on Night 0 (observer-frame days)
_NIGHT0_SCAN_SEP_DAY = 95.0 / _MIN_PER_DAY  # 95 minutes in days
_NIGHT0_SCAN_TIMES = [
    _NIGHT0_FIRST_SCAN_DAY,
    _NIGHT0_FIRST_SCAN_DAY + _NIGHT0_SCAN_SEP_DAY,
    _NIGHT0_FIRST_SCAN_DAY + 2.0 * _NIGHT0_SCAN_SEP_DAY,
]
# Days 1–3: observations at ~1.5, 2.5, 3.5 days from merger
LSST_NIGHTS_1_2_3_DAYS = [1.5, 2.5, 3.5]
LSST_GOLD_OBS_TIMES = {
    "lsstg": _NIGHT0_SCAN_TIMES.copy(),  # g only on Night 0 (~0.5 days + 2 more scans)
    "lsstr": _NIGHT0_SCAN_TIMES + LSST_NIGHTS_1_2_3_DAYS,  # Night 0 + days 1–3
    "lssti": _NIGHT0_SCAN_TIMES + LSST_NIGHTS_1_2_3_DAYS,
}
# LSST Gold strategy 5σ limiting magnitudes (AB)
LSST_LIMIT_MAG = {"lsstg": 26.0, "lsstr": 25.7, "lssti": 24.8}
# Roman 5σ limiting magnitudes (AB)
ROMAN_LIMIT_MAG = {"roman_f129": 25.5, "roman_f158": 25.5}
# Combined 5σ limits for all bands (used for upper limits and for Magerr)
LIMIT_MAG_5SIGMA = {**ROMAN_LIMIT_MAG, **LSST_LIMIT_MAG}

# Magnitude error floor (mag)
MAGERR_FLOOR = 0.01


def magnitude_error_poisson(mag: float, limit_mag: float, floor: float = MAGERR_FLOOR) -> float:
    """
    Magnitude uncertainty assuming Poisson-dominated photometry down to a 5σ limit.

    At the limit magnitude, SNR = 5 so σ_m = 1.086/5 ≈ 0.217 mag. For brighter
    sources, SNR ∝ 10^(0.2*(m_lim - m)), so σ_m = (1.086/5) * 10^(-0.2*(m_lim - m)).
    """
    sigma_at_limit = 1.086 / 5.0
    sigma_m = sigma_at_limit * (10.0 ** (-0.2 * (limit_mag - mag)))
    return max(floor, float(sigma_m))

# Redshift chosen so luminosity distance is exactly 240 Mpc (Planck18)
GW190814_REDSHIFT = 0.05212231244007938

# GW190814 observed merger time (2019-08-14 21:10:39 UTC); observer-frame t_obs added for MJD
MERGER_MJD_GW190814 = 58709.88239583

# Band -> (Source, Filter) for light curve serialization
BAND_TO_SOURCE = {
    "roman_f129": "Roman",
    "roman_f158": "Roman",
    "lsstg": "LSST",
    "lsstr": "LSST",
    "lssti": "LSST",
}
BAND_TO_FILTER = {
    "roman_f129": "F129",
    "roman_f158": "F158",
    "lsstg": "g",
    "lsstr": "r",
    "lssti": "i",
}


def _ensure_roman_bands_registered() -> None:
    """Register Roman F129 and F158 with sncosmo if not already present."""
    try:
        sncosmo.get_bandpass(ROMAN_BANDS[0])
    except Exception:
        register_roman_bands(filters=("F129", "F158"))


def get_at2017gfo_like_parameters(redshift: float = GW190814_REDSHIFT) -> dict:
    """
    Return parameters for an AT2017gfo-like kilonova (three-component model).

    From Villar et al. 2017, ApJL, 851, L21
    (https://iopscience.iop.org/article/10.3847/2041-8213/aa9c84/pdf).
    Default redshift is set so luminosity distance is 240 Mpc (Planck18).
    """
    return dict(
        redshift=redshift,
        mej_1=0.020,
        vej_1=0.256,
        kappa_1=0.5,
        temperature_floor_1=3983,
        mej_2=0.047,
        vej_2=0.152,
        kappa_2=3.0,
        temperature_floor_2=1308,
        mej_3=0.011,
        vej_3=0.137,
        kappa_3=10.0,
        temperature_floor_3=3745,
    )


def run_roman_kilonova(
    times: np.ndarray | None = None,
    bands: list[str] | None = None,
    save_path: str | None = None,
    redshift: float | None = None,
    obs_times: list[float] | None = None,
    lightcurve_data_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Generate an AT2017gfo-like (Villar+2017) three-component kilonova light curve
    in Roman F129/F158 and LSST g/r/i bands.

    Parameters
    ----------
    times : np.ndarray, optional
        Time grid in days. Default: 0 to 30 days, 300 points.
    bands : list of str, optional
        Band names (must be registered with sncosmo). Default: Roman + LSST g,r,i.
    save_path : str, optional
        If set, save the figure to this path.
    redshift : float, optional
        Redshift for the model. Default: 240 Mpc luminosity distance.
    obs_times : list of float, optional
        Epochs (days) for Roman bands to mark as observation points. Default: [7, 12, 19, 28].
        LSST bands use Gold event strategy times (arxiv.org/abs/2411.04793). Set to [] to omit.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from astropy.cosmology import Planck18
    from redback.model_library import all_models_dict

    bands = bands or (ROMAN_BANDS + LSST_BANDS)
    _ensure_roman_bands_registered()

    if times is None:
        times = np.linspace(0.05, 30.0, 300)
    if obs_times is None:
        obs_times = DEFAULT_OBS_TIMES

    params = get_at2017gfo_like_parameters(redshift=redshift or GW190814_REDSHIFT)
    model = all_models_dict["three_component_kilonova_model"]

    # Redback expects one band per call for magnitude output; call per band and collect
    mags = {}
    for band in bands:
        out = model(
            times,
            bands=band,
            output_format="magnitude",
            **params,
        )
        mags[band] = np.atleast_1d(out).flatten()

    # Per-band observation times: LSST uses Gold strategy fiducial times, Roman uses obs_times
    def get_obs_times_for_band(b: str) -> list[float]:
        if b in LSST_GOLD_OBS_TIMES:
            return LSST_GOLD_OBS_TIMES[b]
        return list(obs_times) if obs_times else []

    # Roman-style serif font for all text (legend, title, labels, annotations)
    serif_rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
        "axes.titlesize": 14,
        "legend.fontsize": 11,
    }
    # Rows for light curve CSV: (Source, MJD, Filter, Mag, Magerr, Maglimit); t_obs is observer-frame (time dilation in redshift)
    lightcurve_rows = []
    merger_mjd = MERGER_MJD_GW190814

    with plt.rc_context(serif_rc):
        fig, ax = plt.subplots(figsize=(7.5, 7))
        all_obs_t = []
        all_obs_mag = []
        for band in bands:
            label = BAND_LABELS.get(band, band)
            color = BAND_COLORS.get(band)
            (line,) = ax.plot(
                times, mags[band], label=label, lw=2, color=color if color else None
            )
            plot_color = color if color else line.get_color()
            band_obs_times = get_obs_times_for_band(band)
            if len(band_obs_times) > 0:
                t_obs = np.asarray(band_obs_times, dtype=float)
                mag_obs = model(
                    t_obs,
                    bands=band,
                    output_format="magnitude",
                    **params,
                )
                mag_obs = np.atleast_1d(mag_obs).flatten()
                limit_mag = LIMIT_MAG_5SIGMA.get(band)
                for ti, mi in zip(t_obs, mag_obs):
                    all_obs_t.append(ti)
                    if limit_mag is not None and mi > limit_mag:
                        all_obs_mag.append(limit_mag)
                        all_obs_mag.append(limit_mag + 0.35)  # arrow tip for y-range
                    else:
                        all_obs_mag.append(mi)
                source = BAND_TO_SOURCE.get(band, "Unknown")
                filter_name = BAND_TO_FILTER.get(band, band)
                for ti, mi in zip(t_obs, mag_obs):
                    # MJD = merger + observer-frame days (time dilation already in t_obs)
                    mjd = merger_mjd + ti
                    if limit_mag is not None and mi > limit_mag:
                        lightcurve_rows.append((source, mjd, filter_name, None, None, limit_mag))
                        # Upper limit: arrow at limit magnitude, pointing down
                        ax.annotate(
                            "",
                            xy=(ti, limit_mag + 0.35),
                            xytext=(ti, limit_mag),
                            arrowprops=dict(
                                arrowstyle="->",
                                color=plot_color,
                                lw=2,
                            ),
                            zorder=5,
                        )
                    else:
                        magerr = magnitude_error_poisson(mi, limit_mag) if limit_mag is not None else MAGERR_FLOOR
                        lightcurve_rows.append((source, mjd, filter_name, mi, magerr, None))
                        ax.scatter(
                            [ti],
                            [mi],
                            color=plot_color,
                            marker="o",
                            s=60,
                            zorder=5,
                            edgecolors="white",
                            linewidths=1.5,
                        )
        # Plot range: include all observation circles with padding; curves may extend beyond
        if all_obs_t and all_obs_mag:
            pad_t = 2.0
            pad_mag = 0.8
            x_min = max(0, min(all_obs_t) - pad_t)
            x_max = max(all_obs_t) + pad_t
            mag_min = min(all_obs_mag) - pad_mag
            mag_max = max(all_obs_mag) + pad_mag
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(mag_max, mag_min)  # brighter up
        else:
            ax.set_ylim(ax.get_ylim()[::-1])  # brighter up
        # Axis and tick label sizes (30% larger than 20 pt)
        label_fontsize = 26
        tick_fontsize = 26
        # Redback uses observer-frame time in days (not rest-frame)
        ax.set_xlabel(
            "Observer-frame days from merger",
            fontsize=label_fontsize,
            fontfamily="serif",
        )
        ax.set_ylabel(
            "Magnitude (AB)",
            fontsize=label_fontsize,
            fontfamily="serif",
        )
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily("serif")
        # Legend inside plot frame, left; clear of luminosity distance (right)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.28, 0.98),
            frameon=True,
            fontsize=21,
            prop={"family": "serif"},
            ncol=1,
            framealpha=0.95,
            facecolor="white",
        )
        ax.set_title(
            "AT2017gfo-like kilonova (Villar+2017, 3-component): Roman + LSST",
            fontfamily="serif",
        )
        ax.grid(True, alpha=0.3)

        # Luminosity distance (top-right, inside frame); larger font
        z = params["redshift"]
        d_L_Mpc = Planck18.luminosity_distance(z).to("Mpc").value
        d_L_rounded = round(d_L_Mpc / 10) * 10
        ax.text(
            0.97,
            0.97,
            f"Luminosity distance: {d_L_rounded:.0f} Mpc\n(z = {z:.3f})",
            transform=ax.transAxes,
            fontsize=15,
            fontfamily="serif",
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
        )
        # LSST Gold event strategy: just below luminosity distance, inside frame
        ax.text(
            0.97,
            0.84,
            "LSST: Gold strategy\n"
            "Night 0: 3× gri (95 min apart);\n"
            "Nights 1–3: r+i. Rubin ToO 2024.",
            transform=ax.transAxes,
            fontsize=10,
            fontfamily="serif",
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
        )
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # Serialize light curve to CSV in data directory
    if lightcurve_rows:
        out_path = lightcurve_data_path
        if out_path is None and save_path:
            from pathlib import Path
            data_dir = Path(save_path).resolve().parent.parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(save_path).stem
            out_path = str(data_dir / f"{stem}_lightcurve.csv")
        if out_path:
            from pathlib import Path
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                f.write("Source, MJD, Filter, Mag, Magerr, Maglimit\n")
                for (src, mjd, flt, mag, magerr, maglimit) in lightcurve_rows:
                    mag_s = "" if mag is None else f"{mag:.4f}"
                    err_s = "" if magerr is None else f"{magerr:.4f}"
                    lim_s = "" if maglimit is None else f"{maglimit:.4f}"
                    f.write(f'"{src}", {mjd:.5f}, "{flt}", {mag_s}, {err_s}, {lim_s}\n')
            print(f"Saved light curve data to {out_path}")

    # Write model light curves as ASCII (rest-frame phase, absolute magnitude per band)
    if save_path and bands:
        from pathlib import Path
        z = params["redshift"]
        d_L_pc = Planck18.luminosity_distance(z).to("pc").value
        phase_rest = times / (1.0 + z)
        data_dir = Path(save_path).resolve().parent.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(save_path).stem
        model_path = data_dir / f"{stem}_model_lightcurves.txt"
        with open(model_path, "w") as f:
            filter_names = [BAND_TO_FILTER.get(b, b) for b in bands]
            f.write("# Rest-frame phase (days) and absolute magnitude (AB) per band\n")
            f.write("phase_rest_day " + " ".join(filter_names) + "\n")
            for i in range(len(times)):
                M_list = [mags[b][i] - 5.0 * np.log10(d_L_pc / 10.0) for b in bands]
                f.write(f"{phase_rest[i]:.6f} " + " ".join(f"{M:.4f}" for M in M_list) + "\n")
        print(f"Saved model light curves to {model_path}")

    return fig
