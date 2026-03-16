"""
Plot comparative light curves: blazar TXS 0506+056 (Fermi-style + RATAN-600 digitized),
SN Ic-BL (1998bw template), and a highly reddened SN II-P at 20 kpc (A_V=30).
Phase axis -50 to +365 days. RATAN-600 curve from digitized fig (2.3–22.3 GHz) in magnitudes.
"""

from __future__ import annotations

import os
from urllib.request import urlopen

import numpy as np

# Phase reference for TXS 0506+056: IceCube-170922A neutrino alert (2017-09-22), JD 2458032.5
TXS0506_JD_REF = 2458032.5


def _txs0506_fermi_style_light_curve(phase_days: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fermi-LAT-style gamma-ray light curve for TXS 0506+056 (similar to ResearchGate fig:
    Radio and gamma-ray light curves, black dots Fermi-LAT). Flux in ph/cm^2/s converted
    to magnitude: m = m0 - 2.5*log10(F / F0) so variability shape is preserved.
    """
    # Typical Fermi light curve: baseline ~1e-7, flares to ~3–6e-7 ph/cm^2/s
    F0 = 1e-7  # ph/cm^2/s reference
    baseline = 1.2e-7
    # Two flares: one near phase 50, one near 200
    flare1 = 4.0e-7 * np.exp(-((phase_days - 50.0) ** 2) / (2 * 25.0**2))
    flare2 = 2.5e-7 * np.exp(-((phase_days - 210.0) ** 2) / (2 * 30.0**2))
    flux = baseline + flare1 + flare2
    flux = np.maximum(flux, 0.5e-7)
    # Convert to magnitude: m = m0 - 2.5*log10(F/F0); choose m0 so mag in ~14–18
    m0 = 17.0
    mag = m0 - 2.5 * np.log10(flux / F0)
    # Simple fractional error from Poisson-like scaling
    mag_err = 0.08 * (1.0 + 0.3 * np.abs(phase_days - 50) / 100.0)
    return phase_days, mag, mag_err
# AGN K-band: phase = MJD - 52200 so a segment fits in [-50, 365] (Minezak+ 2019 data span)
AGN_MJD_REF = 52200.0

# Vizier catalog URLs (ASU plain text)
VIZIER_TXS0506 = "https://vizier.cds.unistra.fr/viz-bin/asu-txt?-source=J/ApJ/896/L19&-out.form=ASCII"
VIZIER_AGN_LC = "https://vizier.cds.unistra.fr/viz-bin/asu-txt?-source=J/ApJ/886/150/table2&-out.form=ASCII&-out.max=3000"


def _fetch_txs0506_light_curve() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Fetch TXS 0506+056 MASTER-NET clear-band light curve from Vizier (Lipunov+ 2020, J/ApJ/896/L19).
    Returns (phase_days, mag, mag_err) or None on failure. Phase = JD - JD_ref (IceCube-170922A).
    """
    try:
        with urlopen(VIZIER_TXS0506, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except Exception:
        return None
    phase, mag, err = [], [], []
    for line in text.splitlines():
        if line.strip().startswith("#END"):
            break
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 3 and parts[0].replace(".", "").isdigit():
            try:
                jd = float(parts[0])
                m = float(parts[1])
                e = float(parts[2])
            except (ValueError, IndexError):
                continue
            if 2453000 < jd < 2460000:  # plausible JD range
                phase.append(jd - TXS0506_JD_REF)
                mag.append(m)
                err.append(e)
    if not phase:
        return None
    return np.array(phase), np.array(mag), np.array(err)


def _load_ratan600_digitized(
    data_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Load digitized RATAN-600 light curve (TXS 0506+056, 2.3–22.3 GHz, 20100–20225)
    from data/txs0506_ratan600_digitized.csv. Convert flux (Jy) to magnitude:
    m = m0 - 2.5*log10(S/S0) so the curve sits in ~14–18 mag.
    Returns (phase_days, mag, mag_err) or None.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    path = os.path.join(data_dir, "txs0506_ratan600_digitized.csv")
    if not os.path.isfile(path):
        return None
    try:
        data = np.genfromtxt(path, delimiter=",", skip_header=3, usecols=(0, 1))
    except Exception:
        return None
    if data.size == 0 or data.ndim != 2:
        return None
    phase = data[:, 0]
    flux_jy = np.maximum(data[:, 1], 0.01)
    S0 = 0.5  # Jy reference
    m0 = 16.0
    mag = m0 - 2.5 * np.log10(flux_jy / S0)
    mag_err = np.full_like(mag, 0.06)  # ~6% flux -> ~0.06 mag
    return phase, mag, mag_err


def _fetch_agn_ir_light_curve(agn_id: str = "SDSS J0007-0054") -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Fetch K-band light curve for one AGN from Vizier (Minezaki+ 2019, J/ApJ/886/150).
    Returns (phase_days, mag_AB, mag_err) or None. Phase = MJD - AGN_MJD_REF.
    Flux in uJy converted to AB mag: m_AB = 23.59 - 2.5*log10(F_uJy).
    """
    try:
        with urlopen(VIZIER_AGN_LC, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except Exception:
        return None
    phase, mag, err = [], [], []
    for line in text.splitlines():
        if line.strip().startswith("#END"):
            break
        if line.startswith("#") or not line.strip():
            continue
        # Format: ID ... Band (V/R/I/K) MJD Flux e_Flux
        if " K " not in line or agn_id not in line:
            continue
        parts = line.split()
        try:
            k_idx = parts.index("K")
        except ValueError:
            continue
        if k_idx + 3 >= len(parts):
            continue
        try:
            mjd = float(parts[k_idx + 1])
            flux = float(parts[k_idx + 2])
            e_flux = float(parts[k_idx + 3])
        except (ValueError, IndexError):
            continue
        if flux <= 0:
            continue
        m_ab = 23.59 - 2.5 * np.log10(flux)
        e_mag = 2.5 / np.log(10) * (e_flux / flux) if flux > 0 else 0.1
        phase.append(mjd - AGN_MJD_REF)
        mag.append(m_ab)
        err.append(e_mag)
    if not phase:
        return None
    return np.array(phase), np.array(mag), np.array(err)


def run_agn_blazar_sn_lightcurves(
    times: np.ndarray | None = None,
    save_path: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.figure.Figure":
    """
    Plot AGN (IR), TXS 0506+056, SN Ic-BL model, and reddened SN (15 kpc, Av=25)
    on one light curve figure. Phase -50 to +365 days.
    """
    import matplotlib.pyplot as plt
    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid
    from redback.model_library import all_models_dict

    if times is None:
        times = np.linspace(-50.0, 365.0, 500)

    # RATAN-600 digitized (one band, 2.3–22.3 GHz), flux -> mag; shift to z=2.5 (blazar is at z≈0.33)
    ratan = _load_ratan600_digitized()
    t_ratan, mag_ratan, mag_ratan_err = None, None, None
    if ratan is not None:
        from astropy.cosmology import Planck18 as cosmo
        z_blazar = 0.33
        z_display = 2.5
        dm_blazar = cosmo.distmod(z_blazar).value
        dm_display = cosmo.distmod(z_display).value
        delta_mag = dm_display - dm_blazar  # how much fainter at z=2.5
        pr, mr, er = ratan
        mr = mr + delta_mag
        mask = (pr >= -25) & (pr <= 250)
        if np.any(mask):
            mr_masked = mr[mask]
            # Shift so blazar light curve peaks at 23.7 mag
            peak_mag = np.min(mr_masked)
            mr = mr + (23.7 - peak_mag)
            t_ratan, mag_ratan, mag_ratan_err = pr[mask], mr[mask], er[mask]

    # SN Ic-BL: redback SN 1998bw template at z=1; template valid to ~90 rest-frame days (180 obs at z=1), extend decline to 365 d
    model_sn = all_models_dict["sn1998bw_template"]
    t_sn = times[times >= 0.05]
    mag_sn_icbl_full = np.full_like(times, np.nan, dtype=float)
    if len(t_sn) > 0:
        mag_sn = np.atleast_1d(
            model_sn(
                t_sn,
                redshift=1.0,
                amplitude=1.0,
                output_format="magnitude",
                bands="lsstr",
            )
        ).flatten()
        # Extend beyond template validity (~180 d observer at z=1): continue linear decline (Co decay tail ~0.015 mag/d)
        t_flat = 180.0
        decay_rate = 0.015  # mag per day
        extend = t_sn > t_flat
        if np.any(extend):
            idx_at_flat = np.searchsorted(t_sn, t_flat, side="right") - 1
            if idx_at_flat >= 0:
                mag_at_flat = mag_sn[idx_at_flat]
            else:
                mag_at_flat = mag_sn[0]
            mag_sn[extend] = mag_at_flat + decay_rate * (t_sn[extend] - t_flat)
        # Shift so broad-line Ic peaks at ~24 mag at z=1
        peak_mag = np.nanmin(mag_sn)
        mag_sn = mag_sn + (24.0 - peak_mag)
        mag_sn_icbl_full[times >= 0.05] = mag_sn
    mag_sn_icbl = mag_sn_icbl_full

    # SN II-P at 20 kpc, A_V=30 mag: Nugent Type II-P template (SN2004et-like), scaled to 20 kpc + A_R
    import sncosmo
    model_iip = sncosmo.Model(source="nugent-sn2p")  # Nugent SN II-P, similar to SN2004et
    t_iip = times[times >= 0.5]
    mag_iip_template = np.full_like(times, np.nan, dtype=float)
    if len(t_iip) > 0:
        mag_iip = model_iip.bandmag("lsstr", "ab", t_iip)
        # Scale to 20 kpc + A_V=30: m = M_R + DM(20 kpc) + A_R. II-P peak M_R ~ -16.5.
        DM_20kpc = 5.0 * np.log10(20000.0 / 10.0)
        A_R_Av30 = 0.8 * 30.0
        M_peak_iip = -16.5
        m_peak_20kpc_av30 = M_peak_iip + DM_20kpc + A_R_Av30
        mag_iip_shifted = mag_iip + (m_peak_20kpc_av30 - np.nanmin(mag_iip))
        mag_iip_template[times >= 0.5] = mag_iip_shifted
    mag_sn_reddened = mag_iip_template

    serif_rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
        "axes.titlesize": 14,
        "legend.fontsize": 12,
    }

    with plt.rc_context(serif_rc):
        if ax is None:
            fig, ax = plt.subplots(figsize=(7.5, 7))
        else:
            fig = ax.figure

        if t_ratan is not None:
            order = np.argsort(t_ratan)
            tr, mr = t_ratan[order], mag_ratan[order]
            blazar_color = "darkblue"
            ax.plot(
                tr,
                mr,
                color=blazar_color,
                lw=1.5,
                linestyle="-",
                zorder=2,
                alpha=0.9,
                label="Blazar (z=2.5, H-band)",
            )
        ax.plot(
            times,
            mag_sn_icbl,
            color="green",
            lw=2,
            label="SN Ic-BL (z=1, H)",
            linestyle="-",
        )
        ax.plot(
            times,
            mag_sn_reddened,
            color="red",
            lw=2,
            label="SN II-P (20 kpc, A$_V$=30 mag, H)",
            linestyle="--",
        )

        # y-axis: fixed range 27 mag (bottom) to 23.5 mag (top, bright)
        ax.set_ylim(27.0, 23.5)

        # Roman limiting magnitudes (horizontal lines across full x-range + labels; Roman Shallow removed)
        roman_limits = [
            (26.8, "Roman ToO Deep", 0.14),
            (25.4, "Roman ToO Wide", 0.14),
        ]
        x_min_plot, x_max_plot = -25, 250
        ax.set_xlim(x_min_plot, x_max_plot)
        plot_width = x_max_plot - x_min_plot
        label_x = x_max_plot - 0.04 * plot_width  # inside frame, near right edge
        limit_line_color = "0.25"  # deeper gray
        for limit_mag, label, label_offset in roman_limits:
            ax.hlines(
                limit_mag,
                x_min_plot,
                x_max_plot,
                color=limit_line_color,
                linestyle=":",
                alpha=0.9,
                lw=1.2,
                zorder=1,
            )
            ax.text(
                label_x,
                limit_mag - label_offset,
                f"  {label}",
                color=limit_line_color,
                va="center",
                ha="right",
                fontsize=15,
                fontfamily="serif",
                zorder=6,
            )

        ax.set_xlabel(
            "Observer-frame days from Multi-messenger Event",
            fontsize=14,
            fontfamily="serif",
        )
        ax.set_ylabel(
            "Apparent magnitude (AB)",
            fontsize=18,
            fontfamily="serif",
        )
        ax.tick_params(axis="both", labelsize=14)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily("serif")

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 0.88),
            frameon=True,
            fontsize=14.25,
            prop={"family": "serif"},
            framealpha=0.95,
            facecolor="white",
            ncol=1,
        )
        ax.set_title(
            "Blazar, SN Ic-BL, and reddened SN light curves",
            fontfamily="serif",
            fontsize=11,
            pad=6,
            loc="center",
        )
        ax.grid(True, alpha=0.3)
        if ax is None:
            fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
