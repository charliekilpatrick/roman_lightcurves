"""
Combined figure: three panels left-to-right (Roman kilonova, red kilonova, blazar/SN).
All three panels share the same y-axis scale (magnitude), labeled on the left.
"""

from __future__ import annotations

from lightcurves.agn_blazar_sn_lightcurves import run_agn_blazar_sn_lightcurves
from lightcurves.roman_kilonova_combined import run_roman_kilonova_combined
from lightcurves.roman_kilonova_red_combined import run_roman_kilonova_red_combined


def run_combined_lightcurves(
    save_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Create a three-panel figure (left to right): (1) AT2017gfo-like + KN Impostor,
    (2) Red kilonova, (3) Blazar + SN Ic-BL + SN II-P. All panels share the same
    y-axis scale (27 to 21 mag), labeled "Apparent Magnitude (AB)" on the left.
    """
    import matplotlib.pyplot as plt

    # Shared y-axis range (faint to bright)
    y_min_shared = 27.0
    y_max_shared = 21.0

    fig, axes = plt.subplots(
        1,
        3,
        sharey=True,
        figsize=(18, 6),
        gridspec_kw={"wspace": 0},
    )

    # Left: AT2017gfo-like kilonova + KN Impostor
    run_roman_kilonova_combined(ax=axes[0])

    # Middle: Red kilonova (blue/red gradients, no text labels; no LSST g)
    run_roman_kilonova_red_combined(ax=axes[1], draw_search_phases=True, draw_search_phase_labels=False)

    # Right: Blazar, SN Ic-BL, SN II-P
    run_agn_blazar_sn_lightcurves(ax=axes[2])

    # Force shared y-axis scale for all panels
    for ax in axes:
        ax.set_ylim(y_min_shared, y_max_shared)

    # Share y-axis in the frames: only left panel shows y-axis spine and ticks
    for i in (1, 2):
        axes[i].spines["left"].set_visible(False)
        axes[i].tick_params(axis="y", left=False, labelleft=False)

    # Single y-axis label on the left (first panel only)
    axes[0].set_ylabel("Apparent Magnitude (AB)", fontsize=18, fontfamily="serif")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    # No plot titles
    for ax in axes:
        ax.set_title("")

    # Panel labels "a", "b", "c" in upper right, bold black in bounding box
    for ax, letter in zip(axes, ("a", "b", "c")):
        ax.text(
            0.98,
            0.98,
            letter,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=14,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=1.2),
            zorder=20,
        )

    # Left panel: remove "30" tick label to avoid overlap with middle tick labels
    for lbl in axes[0].get_xticklabels():
        t = lbl.get_text().strip()
        if t == "30" or t == "30.0" or (t.startswith("30") and not t.startswith("300")):
            lbl.set_text("")
            break

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
