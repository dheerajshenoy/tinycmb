# tinycmb

Boilerplate code for simulating CMB maps, foregrounds, and noise, as well as utilities for downgrading maps and computing power spectra.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Features

- **Realistic CMB simulations** using [CAMB](https://camb.info/) with Planck 2018 cosmology
- **Foreground modeling** with [PySM3](https://pysm3.readthedocs.io/) (synchrotron, dust, etc.)
- **White noise generation** with configurable instrument sensitivities
- **Multi-frequency simulations** for realistic CMB experiments
- **Proper map downgrading** using harmonic space methods
- **Beam smoothing** with Gaussian beams
- **Simple API** - just a few lines of code to get started

---

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/yourusername/tinycmb.git
cd tinycmb
pip install -e .
```

Dependencies

- `numpy` - Numerical computing
- `healpy` - HEALPix spherical harmonics
- `camb` - Cosmological power spectra
- `pysm3` - Foreground modeling
- `astropy` - Units and constants
- `matplotlib (optional)` - Visualization

All dependencies are automatically installed.

# Quick Start

## Generate simple CMB map

```python
from tinycmb import simulate_cmb, PLANCK_2018_PARAMS
import healpy as hp

# Simulate CMB with Planck 2018 cosmology
cmb = simulate_cmb(nside=256, cosmic_params=PLANCK_2018_PARAMS, seed=42)

# cmb is a (3, npix) array containing [T, Q, U] in μK
print(f"CMB temperature RMS: {cmb[0].std():.2f} μK")

# Visualize
hp.mollview(cmb[0], title="CMB Temperature", unit="μK")
```

## Quick simulation with everything

```python
from tinycmb import quick_sim
import healpy as hp

# One-line simulation with CMB + foregrounds + noise + beam
total_map = quick_sim(
    nside=64,
    freq_ghz=145,
    sensitivity=5.0,    # μK·arcmin
    beam_fwhm=8.0,      # arcmin
    seed=42
)

hp.mollview(total_map[0], title="Total Sky at 145 GHz", unit="μK")
```

## Multi-frequency simulation

```python
from tinycmb import simulate_full_sky, save_maps, PLANCK_2018_PARAMS

# Configure frequency channels
freq_configs = [
    {"freq": 94, "sens": 7.0, "beam_fwhm": 12.0},
    {"freq": 145, "sens": 5.2, "beam_fwhm": 8.0},
    {"freq": 217, "sens": 12.4, "beam_fwhm": 5.0},
]

# Run full simulation
results = simulate_full_sky(
    nside=64,
    lmax=191,
    cosmic_params=PLANCK_2018_PARAMS,
    freq_configs=freq_configs,
    fg_models=["s1", "d1"],  # Synchrotron + dust
    apply_beam=True,
    cmb_seed=42,
    noise_seed=123,
)

# Access results for specific frequency
data_145 = results[145]
print(f"CMB RMS: {data_145['cmb'][0].std():.2f} μK")
print(f"Foreground RMS: {data_145['foreground'][0].std():.2f} μK")
print(f"Noise RMS: {data_145['noise'][0].std():.2f} μK")

# Save all maps to FITS files
save_maps(results, output_dir="simulation_output")
```

References

This package builds on:

- Planck 2018 Cosmology: Planck Collaboration (2020)
- CAMB: Lewis, A., Challinor, A., & Lasenby, A. (2000). ApJ, 538, 473
- HEALPix: Górski et al. (2005). ApJ, 622, 759
- PySM3: Thorne et al. (2017). MNRAS, 469, 2821

