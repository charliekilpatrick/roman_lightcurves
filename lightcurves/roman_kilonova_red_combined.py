"""
Plot red (single-component) kilonova light curves with Rubin and Roman search
phase regions. Same styling as roman_kilonova_combined (gradients, labels, legend)
but no KN Impostor. Uses redback one_component_kilonova_model.
"""

from __future__ import annotations

import numpy as np

from lightcurves.roman_kilonova_red import (
    BAND_COLORS,
    BAND_LABELS,
    DEFAULT_OBS_TIMES,
    GW190814_REDSHIFT,
    LSST_BANDS,
    LSST_GOLD_OBS_TIMES,
    LIMIT_MAG_5SIGMA,
    ROMAN_BANDS,
    _ensure_roman_bands_registered,
    get_red_kilonova_parameters,
)


def run_roman_kilonova_red_combined(
    times: np.ndarray | None = None,
    bands: list[str] | None = None,
    save_path: str | None = None,
    redshift: float | None = None,
    obs_times: list[float] | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
    draw_search_phases: bool = True,
    draw_search_phase_labels: bool = True,
) -> "matplotlib.figure.Figure":
    """
    Plot red kilonova light curves with Roman F129/F158 and LSST g/r/i,
    with Rubin and Roman search phase shaded regions and phase labels.
    """
    import matplotlib.pyplot as plt
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
    params = get_red_kilonova_parameters(redshift=z)
    model = all_models_dict["one_component_kilonova_model"]

    mags = {}
    for band in bands:
        mags[band] = np.atleast_1d(
            model(times, bands=band, output_format="magnitude", **params)
        ).flatten()

    def get_obs_times_for_band(b: str) -> list[float]:
        if b in LSST_GOLD_OBS_TIMES:
            return LSST_GOLD_OBS_TIMES[b]
        return list(obs_times) if obs_times else []

    _g = LSST_GOLD_OBS_TIMES["lsstg"]
    _r = LSST_GOLD_OBS_TIMES["lsstr"]
    assert _g[0] == 0.5, "LSST day 0 first scan should be at 0.5 days from merger"
    assert _r[-3:] == [1.5, 2.5, 3.5], "LSST days 1–3 should be at 1.5, 2.5, 3.5 days from merger"

    serif_rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
        "axes.titlesize": 14,
        "legend.fontsize": 14,
    }
    all_obs_t: list[float] = []
    all_obs_mag: list[float] = []
    red_lines: list = []

    with plt.rc_context(serif_rc):
        if ax is None:
            fig, ax = plt.subplots(figsize=(7.5, 7))
        else:
            fig = ax.figure

        LSST_T_CUTOFF = 1.0  # mask first 1 day of LSST g, r, i evolution
        bands_to_plot = [b for b in bands if b != "lsstg"]
        for band in bands_to_plot:
            label = BAND_LABELS.get(band, band)
            color = BAND_COLORS.get(band)
            ls = "--" if band in LSST_BANDS else "-"
            t_plot = times
            m_plot = mags[band]
            if band in LSST_BANDS:
                mask = times >= LSST_T_CUTOFF
                t_plot = times[mask]
                m_plot = mags[band][mask]
            (line,) = ax.plot(
                t_plot,
                m_plot,
                label=label,
                lw=2,
                color=color if color else None,
                linestyle=ls,
            )
            red_lines.append(line)
            plot_color = color if color else line.get_color()
            band_obs_times = get_obs_times_for_band(band)
            if band_obs_times:
                t_obs = np.asarray(band_obs_times, dtype=float)
                mag_obs = np.atleast_1d(
                    model(t_obs, bands=band, output_format="magnitude", **params)
                ).flatten()
                limit_mag = LIMIT_MAG_5SIGMA.get(band)
                for ti, mi in zip(t_obs, mag_obs):
                    all_obs_t.append(ti)
                    if limit_mag is not None and mi > limit_mag:
                        all_obs_mag.append(limit_mag)
                        all_obs_mag.append(limit_mag + 0.35)
                    else:
                        all_obs_mag.append(mi)

        if all_obs_t and all_obs_mag:
            pad_t = 2.0
            x_min = max(0, min(all_obs_t) - pad_t)
            x_max = max(all_obs_t) + pad_t
            ax.set_xlim(x_min, x_max)
        ax.set_ylim(27.0, 21.0)

        x_min_plot, x_max_plot = ax.get_xlim()
        y_bot, y_top = ax.get_ylim()
        if draw_search_phases:
            lsst_search_x_end = 5.5
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
            roman_search_x_start = 7.0
            grad_red = np.linspace(0, 1, n_steps).reshape(1, -1)
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

        plot_width = x_max_plot - x_min_plot
        line_length = 0.08 * plot_width
        line_x_end = x_max_plot
        line_x_start = x_max_plot - line_length
        label_x = x_max_plot - line_length - 0.02 * plot_width
        roman_limit_label_done = False
        used_at_mag: dict[float, int] = {}
        # Roman limits at 25.4 mag in middle panel (same as Roman Medium in right panel)
        ROMAN_LIMIT_MAG_MIDDLE = 25.4
        for band in bands_to_plot:
            limit_mag = ROMAN_LIMIT_MAG_MIDDLE if band in ROMAN_BANDS else LIMIT_MAG_5SIGMA.get(band)
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
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily("serif")

        # Order without lsstg: F158, F129, lssti, lsstr -> red_lines indices 1, 0, 3, 2
        band_order_red = [1, 0, 3, 2]
        legend_handles = [red_lines[i] for i in band_order_red]
        legend_labels = [BAND_LABELS.get(bands_to_plot[i], bands_to_plot[i]) for i in band_order_red]
        ax.legend(
            legend_handles,
            legend_labels,
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            frameon=True,
            fontsize=17,
            prop={"family": "serif"},
            framealpha=0.95,
            facecolor="white",
        )
        # "Red Kilonova" annotation above the legend (raised by 2 mag in axes ≈ +0.07)
        ax.text(
            0.5,
            0.69,
            "Red Kilonova",
            color="red",
            fontsize=24,
            fontfamily="serif",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.95),
            zorder=10,
        )
        ax.set_title(
            "Red kilonova (Roman + LSST)",
            fontfamily="serif",
        )
        ax.grid(True, alpha=0.3)

        if draw_search_phases and draw_search_phase_labels:
            ax.text(
                4.2,
                24.08,
                "Rubin\nSearch Phases",
                color="blue",
                fontsize=12,
                fontfamily="serif",
                ha="center",
                va="top",
                transform=ax.transData,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="blue", alpha=0.95),
                zorder=10,
            )
            ax.text(
                19.0,
                24.35,
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
