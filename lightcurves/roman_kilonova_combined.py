"""
Plot AT2017gfo-like (three-component) kilonova light curves with Rubin and Roman
search phase regions. Adopts code from lightcurves.roman_kilonova.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lightcurves.roman_kilonova import (
    BAND_COLORS,
    BAND_LABELS,
    DEFAULT_OBS_TIMES,
    GW190814_REDSHIFT,
    LSST_BANDS,
    LSST_GOLD_OBS_TIMES,
    LIMIT_MAG_5SIGMA,
    ROMAN_BANDS,
    _ensure_roman_bands_registered,
    get_at2017gfo_like_parameters,
)


def _load_sniib_params_from_mcmc(result_path: Path) -> dict | None:
    """Load best-fit KN Impostor (shock cooling + Arnett) parameters from sniib_mcmc_result.json; None if missing/invalid."""
    if not result_path.is_file():
        return None
    try:
        with open(result_path) as f:
            data = json.load(f)
        best = data.get("best_theta_dict")
        if not best:
            return None
        return dict(best)
    except (json.JSONDecodeError, OSError):
        return None


def run_roman_kilonova_combined(
    times: np.ndarray | None = None,
    bands: list[str] | None = None,
    save_path: str | None = None,
    redshift: float | None = None,
    obs_times: list[float] | None = None,
    sniib_mcmc_result_path: str | Path | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.figure.Figure":
    """
    Plot AT2017gfo-like kilonova and KN Impostor light curves with Roman F129/F158 and LSST g/r/i,
    with Rubin and Roman search phase shaded regions and phase labels.
    KN Impostor (shock cooling + Arnett) uses best-fit parameters from sniib_mcmc_result.json if present (path optional).
    """
    import matplotlib.pyplot as plt
    # NumPy 2.0 removed np.trapz; redback still uses it
    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid
    from redback.model_library import all_models_dict

    bands = bands or (ROMAN_BANDS + LSST_BANDS)
    _ensure_roman_bands_registered()

    if times is None:
        times = np.linspace(0.05, 30.0, 300)
    if obs_times is None:
        obs_times = DEFAULT_OBS_TIMES

    z = redshift or GW190814_REDSHIFT
    params_blue = get_at2017gfo_like_parameters(redshift=z)
    model_blue = all_models_dict["three_component_kilonova_model"]

    mags_blue = {}
    for band in bands:
        mags_blue[band] = np.atleast_1d(
            model_blue(times, bands=band, output_format="magnitude", **params_blue)
        ).flatten()

    # KN Impostor (shock_cooling_and_arnett): use MCMC best-fit to AT2017gfo r-band if available
    _pkg_root = Path(__file__).resolve().parent.parent
    _default_mcmc_path = _pkg_root / "mcmc_results" / "sniib_mcmc_result.json"
    mcmc_path = Path(sniib_mcmc_result_path) if sniib_mcmc_result_path else _default_mcmc_path
    best_theta = _load_sniib_params_from_mcmc(mcmc_path)
    mags_sniib = {}
    sniib_available = False
    try:
        if best_theta is not None:
            params_sniib = {
                "output_format": "magnitude",
                "redshift": z,
                "nn": 6.0,
                "delta": 1.0,
                **{k: np.float64(v) for k, v in best_theta.items()},
            }
        else:
            # Fallback: previous default parameters
            params_sniib = {
                "output_format": "magnitude",
                "redshift": z,
                "log10_mass": np.log10(1.5e-2),
                "log10_radius": np.log10(2.16e12),
                "log10_energy": np.log10(2.88e50),
                "nn": 6.0,
                "delta": 1.0,
                "f_nickel": np.float64(0.03),
                "mej": np.float64(1.0),
                "vej": np.float64(8100),
                "kappa": 1.0,
                "kappa_gamma": np.float64(0.01),
                "temperature_floor": np.float64(3550),
            }
        model_sniib = all_models_dict["shock_cooling_and_arnett"]
        mags_sniib["lsstr"] = np.atleast_1d(
            model_sniib(times, **params_sniib, bands="lsstr")
        ).flatten()
        mags_sniib["roman_f158"] = np.atleast_1d(
            model_sniib(times, **params_sniib, bands="roman_f158")
        ).flatten()
        sniib_available = True
    except Exception as e:
        import warnings
        warnings.warn(f"KN Impostor model unavailable (redback/numpy?), skipping: {e}")

    def get_obs_times_for_band(b: str) -> list[float]:
        if b in LSST_GOLD_OBS_TIMES:
            return LSST_GOLD_OBS_TIMES[b]
        return list(obs_times) if obs_times else []

    # Validate LSST observation times: day 0 ~0.5 days from merger; days 1–3 at 1.5, 2.5, 3.5
    _g = LSST_GOLD_OBS_TIMES["lsstg"]
    _r = LSST_GOLD_OBS_TIMES["lsstr"]
    assert _g[0] == 0.5, "LSST day 0 first scan should be at 0.5 days from merger"
    assert _r[-3:] == [1.5, 2.5, 3.5], "LSST days 1–3 should be at 1.5, 2.5, 3.5 days from merger"

    serif_rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
        "axes.titlesize": 14,
        "legend.fontsize": 12,
    }
    all_obs_t: list[float] = []
    all_obs_mag: list[float] = []
    blue_lines: list = []
    sniib_lines: list = []

    with plt.rc_context(serif_rc):
        if ax is None:
            fig, ax = plt.subplots(figsize=(7.5, 7))
        else:
            fig = ax.figure

        # --- AT2017gfo-like: solid for Roman, dashed for LSST
        for band in bands:
            label = BAND_LABELS.get(band, band)
            color = BAND_COLORS.get(band)
            ls = "--" if band in LSST_BANDS else "-"
            (line,) = ax.plot(
                times,
                mags_blue[band],
                label=label,
                lw=2,
                color=color if color else None,
                linestyle=ls,
            )
            blue_lines.append(line)
            plot_color = color if color else line.get_color()
            band_obs_times = get_obs_times_for_band(band)
            if band_obs_times:
                t_obs = np.asarray(band_obs_times, dtype=float)
                mag_obs = np.atleast_1d(
                    model_blue(t_obs, bands=band, output_format="magnitude", **params_blue)
                ).flatten()
                limit_mag = LIMIT_MAG_5SIGMA.get(band)
                for ti, mi in zip(t_obs, mag_obs):
                    all_obs_t.append(ti)
                    if limit_mag is not None and mi > limit_mag:
                        all_obs_mag.append(limit_mag)
                        all_obs_mag.append(limit_mag + 0.35)
                    else:
                        all_obs_mag.append(mi)

        # --- KN Impostor: LSST r and Roman F158, dotted (if model available)
        if sniib_available:
            (line,) = ax.plot(
                times,
                mags_sniib["lsstr"],
                label="LSST/r (KN Impostor)",
                lw=3.5,
                color="black",
                linestyle=":",
                zorder=2,
            )
            sniib_lines.append(line)
            (line_f158,) = ax.plot(
                times,
                mags_sniib["roman_f158"],
                label="F158 (KN Impostor)",
                lw=2.5,
                color=BAND_COLORS.get("roman_f158", "red"),
                linestyle=":",
                zorder=2,
            )
            sniib_lines.append(line_f158)

        if all_obs_t and all_obs_mag:
            pad_t = 2.0
            x_min = max(0, min(all_obs_t) - pad_t)
            x_max = max(all_obs_t) + pad_t
            ax.set_xlim(x_min, x_max)
        # Fixed magnitude range: 21 mag (bright, top) to 27 mag (faint, bottom)
        ax.set_ylim(27.0, 21.0)

        # Shaded blue gradient: LSST counterpart search time range (extends over LSST optical obs)
        y_bot, y_top = ax.get_ylim()
        lsst_search_x_end = 5.5  # days from merger (overlaps most LSST optical observations)
        n_steps = 80
        gradient = np.linspace(1, 0, n_steps).reshape(1, -1)
        rgba = np.ones((1, n_steps, 4))
        rgba[0, :, 0] = 0.2
        rgba[0, :, 1] = 0.35
        rgba[0, :, 2] = 0.75
        rgba[0, :, 3] = 0.38 * gradient.squeeze()
        ax.imshow(
            rgba,
            extent=[0, lsst_search_x_end, y_bot, y_top],
            aspect="auto",
            interpolation="bilinear",
            zorder=0,
        )

        # Shaded red gradient: Roman observation time scale (fades in around 7 days, extends to right)
        x_min_plot, x_max_plot = ax.get_xlim()
        roman_search_x_start = 7.0  # days from merger (Roman default first epoch)
        grad_red = np.linspace(0, 1, n_steps).reshape(1, -1)  # fade in from left
        rgba_red = np.ones((1, n_steps, 4))
        rgba_red[0, :, 0] = 0.85
        rgba_red[0, :, 1] = 0.2
        rgba_red[0, :, 2] = 0.2
        rgba_red[0, :, 3] = 0.32 * grad_red.squeeze()
        ax.imshow(
            rgba_red,
            extent=[roman_search_x_start, x_max_plot, y_bot, y_top],
            aspect="auto",
            interpolation="bilinear",
            zorder=0,
        )

        # Horizontal lines and labels for 5σ limiting magnitude per band (inside frame)
        plot_width = x_max_plot - x_min_plot
        line_length = 0.08 * plot_width
        line_x_end = x_max_plot
        line_x_start = x_max_plot - line_length
        label_x = x_max_plot - line_length - 0.02 * plot_width
        roman_limit_label_done = False
        used_at_mag: dict[float, int] = {}
        for band in bands:
            limit_mag = LIMIT_MAG_5SIGMA.get(band)
            if limit_mag is None:
                continue
            color = BAND_COLORS.get(band)
            if color is None:
                color = "gray"
            ax.hlines(
                limit_mag,
                line_x_start,
                line_x_end,
                color=color,
                linestyle=":",
                alpha=0.7,
                lw=1.5,
                zorder=1,
            )
            # Roman: single label "Roman limits" for both F129 and F158
            if band in ROMAN_BANDS:
                if not roman_limit_label_done:
                    ax.text(
                        label_x,
                        limit_mag,
                        "Roman limits  ",
                        color="red",
                        va="center",
                        ha="right",
                        fontsize=10,
                        fontfamily="serif",
                        zorder=6,
                    )
                    roman_limit_label_done = True
                continue
            # LSST: one label per band
            n_at = used_at_mag.get(limit_mag, 0)
            label_y = limit_mag - 0.06 * n_at
            used_at_mag[limit_mag] = n_at + 1
            band_label = BAND_LABELS.get(band, band)
            ax.text(
                label_x,
                label_y,
                f"{band_label} limit  ",
                color=color,
                va="center",
                ha="right",
                fontsize=10,
                fontfamily="serif",
                zorder=6,
            )

        label_fontsize = 26
        xlabel_fontsize = 20
        tick_fontsize = 26
        ax.set_xlabel(
            "Observer-frame days from merger",
            fontsize=xlabel_fontsize,
            fontfamily="serif",
        )
        ax.set_ylabel(
            "Apparent Magnitude (AB)",
            fontsize=label_fontsize,
            fontfamily="serif",
        )
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.set_xticks([0, 10, 20])
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily("serif")

        # Legend: AT2017gfo bands + KN Impostor if available
        legend_fontsize_base = 14
        legend_fontsize = legend_fontsize_base * 0.7  # decrease by 30%
        band_order = [1, 0, 4, 3, 2]
        legend_handles = [blue_lines[i] for i in band_order]
        legend_labels = [BAND_LABELS.get(bands[i], bands[i]) for i in band_order]
        if sniib_available:
            legend_handles += list(sniib_lines)
            legend_labels += ["LSST/r (KN Impostor)", "F158 (KN Impostor)"]
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            bbox_to_anchor=(0.97, 0.83),
            frameon=True,
            fontsize=legend_fontsize*0.6,
            prop={"family": "serif"},
            framealpha=0.95,
            facecolor="white",
        )
        ax.set_title(
            "AT2017gfo-like kilonova and KN Impostor (Roman + LSST)" if sniib_available else "Kilonova light curves: AT2017gfo-like (Roman + LSST)",
            fontfamily="serif",
        )
        ax.grid(True, alpha=0.3)

        # AT2017gfo label centered near 10 days, 22 mag
        ax.text(
            10.0,
            22.0,
            "AT2017gfo",
            color="blue",
            fontsize=22,
            fontfamily="serif",
            ha="center",
            va="center",
            transform=ax.transData,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="blue", alpha=0.95),
            zorder=10,
        )

        # Search Phases labels near top of plot (above light curves), line break so text stays within frame
        ax.text(
            0.12,
            0.99,
            "Rubin\nSearch Phases",
            color="blue",
            fontsize=12,
            fontfamily="serif",
            ha="center",
            va="top",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="blue", alpha=0.95),
            zorder=10,
        )
        # Roman Search Phases label centered near 19 days (data coordinates)
        ax.text(
            19.0,
            21.4,
            "Roman\nSearch Phases",
            color="red",
            fontsize=12,
            fontfamily="serif",
            ha="center",
            va="top",
            transform=ax.transData,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="red", alpha=0.95),
            zorder=10,
        )
        if ax is None:
            fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
