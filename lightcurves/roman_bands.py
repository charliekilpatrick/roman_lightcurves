"""
Register Roman WFI imaging bandpasses (F129, F158) with sncosmo for use with redback.

Throughputs are taken from the Roman technical information repo (EffectiveAreas ECSV).
Wave is in microns; effective area in m² is normalized to 0–1 for bandpass transmission.
"""

from __future__ import annotations

from typing import Sequence
from urllib.request import urlopen

import numpy as np
import sncosmo

# Default: one SCA; same format as other SCAs for filter curves
ROMAN_ECSV_URL = (
    "https://raw.githubusercontent.com/RomanSpaceTelescope/roman-technical-information"
    "/refs/tags/v1.2/data/WideFieldInstrument/Imaging/EffectiveAreas/"
    "Roman_effarea_v8_SCA08_20240301.ecsv"
)


def load_roman_effective_area_ecsv(
    url: str = ROMAN_ECSV_URL,
) -> "astropy.table.Table":
    """Load Roman effective area ECSV from URL into an astropy Table."""
    from astropy.table import Table

    with urlopen(url) as resp:
        lines = resp.read().decode().splitlines()
    # Find the header line (column names) and data; ECSV comment block can confuse strict parser
    data_lines = []
    names = None
    for line in lines:
        if line.startswith("#"):
            continue
        if names is None:
            names = [c.strip() for c in line.split(",")]
            continue
        data_lines.append(line)
    from io import BytesIO
    table_str = "\n".join([",".join(names)] + data_lines)
    return Table.read(BytesIO(table_str.encode("utf-8")), format="csv")


def build_roman_bandpass(
    table: "astropy.table.Table",
    filter_name: str,
    name: str | None = None,
) -> sncosmo.Bandpass:
    """
    Build an sncosmo Bandpass from the effective area table for one filter.

    Parameters
    ----------
    table : astropy.table.Table
        Table with 'Wave' (micron) and a column matching filter_name (e.g. 'F129').
    filter_name : str
        Column name for the filter, e.g. 'F129', 'F158'.
    name : str, optional
        Name for the bandpass (used by sncosmo). Defaults to 'roman_<filter_name.lower>'.

    Returns
    -------
    sncosmo.Bandpass
        Wavelength in Angstroms, transmission in [0, 1].
    """
    wave_um = np.asarray(table["Wave"], dtype=float)
    eff_area = np.asarray(table[filter_name], dtype=float)
    # sncosmo expects wavelength in Angstroms
    wave_angstrom = wave_um * 1e4
    # Normalize to 0–1 for bandpass transmission
    transmission = eff_area / np.nanmax(eff_area)
    transmission = np.clip(transmission, 0.0, None)
    band_name = name or f"roman_{filter_name.lower()}"
    return sncosmo.Bandpass(wave_angstrom, transmission, name=band_name)


def register_roman_bands(
    filters: Sequence[str] = ("F129", "F158"),
    ecsv_url: str = ROMAN_ECSV_URL,
) -> list[str]:
    """
    Load Roman effective area ECSV, build bandpasses for the given filters, and register with sncosmo.

    Parameters
    ----------
    filters : sequence of str
        Filter column names, e.g. ('F129', 'F158').
    ecsv_url : str
        URL to the Roman EffectiveAreas ECSV file.

    Returns
    -------
    list of str
        Registered band names (e.g. ['roman_f129', 'roman_f158']).
    """
    table = load_roman_effective_area_ecsv(ecsv_url)
    registered = []
    for f in filters:
        band = build_roman_bandpass(table, f)
        sncosmo.register(band)
        registered.append(band.name)
    return registered
