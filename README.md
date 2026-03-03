# roman_lightcurves

**Roman WFI imaging-band light curves for transients** — synthetic light curves in Roman filter bandpasses (e.g. F129, F158) using [redback](https://github.com/nikhil-sarin/redback) kilonova models and throughputs from the [roman-technical-information](https://github.com/RomanSpaceTelescope/roman-technical-information) EffectiveAreas ECSV (v1.2). Use this repo to generate and plot AT2017gfo-like (Villar+2017) three-component kilonova light curves in Roman bands and to register Roman bandpasses with [sncosmo](https://sncosmo.readthedocs.io/) for other transient work.

---

## Repository layout

| Path | Description |
|------|-------------|
| `lightcurves/` | Python package: Roman bandpass registration and kilonova light-curve generation |
| `lightcurves/roman_bands.py` | Fetches Roman EffectiveAreas ECSV, builds sncosmo bandpasses (Wave in µm → Å, transmission 0–1), registers e.g. `roman_f129`, `roman_f158` |
| `lightcurves/roman_kilonova.py` | AT2017gfo-like three-component kilonova (Villar+2017) and plotting; uses redback + registered Roman bands; default redshift GW190814 (0.05) |
| `scripts/run_roman_kilonova_example.py` | Example script: generates a light-curve figure and saves it under `figures/` |
| `figures/` | Output directory for saved plots (created automatically when running the example script) |
| `environment.yml` | Conda environment (Python 3.11, numba, pip install of this package) |
| `pyproject.toml` | Project metadata and Python dependencies; install with `pip install -e .` |
| `requirements.txt` | Legacy pip requirements (same deps as in `pyproject.toml`) |

---

## Setup

**Python:** Use Python 3.10 or 3.11 (recommended). Python 3.13 is not yet supported by `numba`/`redback`; if `pip install` fails, use the conda environment below.

**Conda (recommended):**

```bash
conda env create -f environment.yml
conda activate roman_lightcurves
```

**Or pip only:**

```bash
pip install -e .
# or: pip install -r requirements.txt  (then use lightcurves from this repo)
```

Generated figures are written to the `figures/` directory (created automatically when running the example script).

---

## Roman bands

The `lightcurves` package registers **F129** and **F158** with sncosmo using the ECSV effective area tables (SCA08, v8, 20240301). Wavelength is in µm; effective area (m²) is normalized to 0–1 for bandpass transmission. You can call `register_roman_bands()` from `lightcurves` to ensure these bandpasses are available before running models.

---

## AT2017gfo-like kilonova (Villar+2017)

The model is the **three-component kilonova** from [Villar et al. 2017, ApJL, 851, L21](https://iopscience.iop.org/article/10.3847/2041-8213/aa9c84/pdf) (AT2017gfo-like). The default redshift is GW190814 (0.05) as a dummy for the template.

Generate and plot the light curve in Roman F129 and F158:

```python
from lightcurves.roman_kilonova import run_roman_kilonova
import numpy as np

times = np.linspace(0.2, 15, 100)
fig = run_roman_kilonova(times=times, save_path="figures/roman_kilonova_example.png")
```

Or run the example script from the repo root (writes to `figures/`):

```bash
python scripts/run_roman_kilonova_example.py
```

Or use defaults (0.1–20 days, 200 points, Roman F129/F158, redshift=0.05):

```python
from lightcurves.roman_kilonova import run_roman_kilonova
run_roman_kilonova(save_path="figures/roman_kilonova_example.png")
# Optional: pass redshift=0.023 for another distance
```

---

## License

See [LICENSE](LICENSE).
