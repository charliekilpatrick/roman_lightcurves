"""
Generate a single-component red kilonova model and plot light curves in Roman
F129/F158 and LSST g/r/i bands.

Uses redback's one_component_kilonova_model with parameters corresponding to
the red (third) component of the Villar+2017 AT2017gfo-like model (mej_3, vej_3,
kappa_3, temperature_floor_3). Kept separate from roman_kilonova.py (three-component).
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
    "lssti": "darkgoldenrod",  # dark yellow
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
LSST_GOLD_OBS_TIMES = {
    "lsstg": _NIGHT0_SCAN_TIMES.copy(),  # g only on Night 0 (3 scans)
    "lsstr": _NIGHT0_SCAN_TIMES + [2.0, 3.0, 4.0],  # Night 0 (3 scans) + Nights 1–3
    "lssti": _NIGHT0_SCAN_TIMES + [2.0, 3.0, 4.0],
}
# LSST Gold strategy 5σ limiting magnitudes (AB)
LSST_LIMIT_MAG = {"lsstg": 26.0, "lsstr": 25.7, "lssti": 25.0}

# Redshift chosen so luminosity distance is exactly 240 Mpc (Planck18)
GW190814_REDSHIFT = 0.05212231244007938

# Red component: Villar+2017 third-component parameters for one_component_kilonova_model
RED_MEJ = 0.02    # ejecta mass (solar masses)
RED_VEJ = 0.1     # ejecta velocity (c)
RED_KAPPA = 10.0  # kappa_3
RED_TEMPERATURE_FLOOR = 3500  # temperature floor (K)


def _ensure_roman_bands_registered() -> None:
    """Register Roman F129 and F158 with sncosmo if not already present."""
    try:
        sncosmo.get_bandpass(ROMAN_BANDS[0])
    except Exception:
        register_roman_bands(filters=("F129", "F158"))


def get_red_kilonova_parameters(redshift: float = GW190814_REDSHIFT) -> dict:
    """
    Return parameters for a single-component red kilonova (one_component_kilonova_model).

    Values match the red (third) component of the Villar+2017 AT2017gfo-like model:
    mej_3, vej_3, kappa_3, temperature_floor_3.
    """
    return dict(
        redshift=redshift,
        mej=RED_MEJ,
        vej=RED_VEJ,
        kappa=RED_KAPPA,
        temperature_floor=RED_TEMPERATURE_FLOOR,
    )


def run_roman_kilonova_red(
    times: np.ndarray | None = None,
    bands: list[str] | None = None,
    save_path: str | None = None,
    redshift: float | None = None,
    obs_times: list[float] | None = None,
) -> "matplotlib.figure.Figure":
    """
    Generate a single-component red kilonova light curve in Roman F129/F158 and
    LSST g/r/i bands.

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
        LSST bands use Gold event strategy times. Set to [] to omit.

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

    params = get_red_kilonova_parameters(redshift=redshift or GW190814_REDSHIFT)
    model = all_models_dict["one_component_kilonova_model"]

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
                limit_mag = LSST_LIMIT_MAG.get(band)
                for ti, mi in zip(t_obs, mag_obs):
                    all_obs_t.append(ti)
                    if limit_mag is not None and mi > limit_mag:
                        all_obs_mag.append(limit_mag)
                        all_obs_mag.append(limit_mag + 0.35)  # arrow tip for y-range
                    else:
                        all_obs_mag.append(mi)
                for ti, mi in zip(t_obs, mag_obs):
                    if limit_mag is not None and mi > limit_mag:
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
        # Axis and tick label sizes
        label_fontsize = 20
        tick_fontsize = 20
        ax.set_xlabel(
            "Observer-frame time (days)",
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
            "Red kilonova (1-component, Villar+17 red): Roman + LSST",
            fontfamily="serif",
        )
        ax.grid(True, alpha=0.3)

        # Luminosity distance (top-right, inside frame)
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
            0.62,
            "LSST circles: Gold event strategy\n"
            "(Night 0: 3× gri scans, 95 min apart; Nights 1–3: r+i)\n"
            "Rubin ToO 2024 (arxiv.org/abs/2411.04793)",
            transform=ax.transAxes,
            fontsize=12,
            fontfamily="serif",
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
        )
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
