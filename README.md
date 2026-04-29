# tinycmb

A lightweight Python package for simulating CMB (Cosmic Microwave Background) maps, foregrounds, and noise for multi-frequency experiments.

## Installation

```bash
pip install -e .
```

**Dependencies:** `healpy`, `numpy`, `pysm3`, `matplotlib`

## Quick Start

```python
import numpy as np
from tinycmb import Simulator, Config, CosmoConfig

# Define cosmological parameters
cosmo_config = CosmoConfig(
    H0=67.5, ombh2=0.022, omch2=0.122,
    As=2e-9, ns=0.965, r=0
)

# Configure the simulation
config = Config(
    nside_out=1024,
    nside_cmb=512,
    nside_fg=512,
    fg_models=["s0", "d0"],
    lmax=3000,
    cosmo_params=cosmo_config,
    spectra_type="total",
    freqs=np.array([50.0, 78.0, 100.0, 119.0, 140.0, 166.0,
                    195.0, 235.0, 280.0, 337.0]),
    sens=np.array([32.78, 18.59, 12.93, 9.79, 9.55, 5.81,
                   7.12, 15.16, 17.98, 24.99]),
    beam_arcmin=15.0,
    output_unit="uK_CMB",
    add_noise=True,
)

simulator = Simulator(config)
```

## Simulations

### CMB Maps

```python
cmb_maps = simulator.simulate_cmb()
# cmb_maps[freq_idx] -> array of shape (3, npix) for T, Q, U
```

### Foreground Maps

```python
fg_maps = simulator.simulate_foregrounds()
```

Foreground models are specified via `fg_models` in `Config`. PySM3 model strings are supported, e.g. `"s0"` (synchrotron), `"d0"` (thermal dust).

### Noise Maps

```python
noise_maps = simulator.simulate_noise()
```

Noise is generated from the per-frequency sensitivities (`sens`) provided in `Config`. Set `add_noise=False` in `Config` to disable.

### Combined Maps

```python
total_maps = cmb_maps + fg_maps + noise_maps
```

## Map Downgrading

Use `downgrade_map_harmonic` to reproject maps to a lower resolution with optional beam smoothing:

```python
from tinycmb.utils import downgrade_map_harmonic

cmb_maps_downgraded = downgrade_map_harmonic(
    cmb_maps,
    nside_in=config.nside_cmb,
    nside_out=config.nside_out,
    beam_fwhm_arcmin=config.beam_arcmin,
    extra_smoothing_fwhm_arcmin=0.0,
    lmax_out=config.lmax,
    iter=0
)
```

## Visualization

```python
import healpy as hp
import matplotlib.pyplot as plt
from tinycmb.utils import plot_cmb_spectra

# Mollweide projection of CMB temperature map
hp.mollview(cmb_maps[0][0], title="Simulated CMB T", unit="muK")
plt.show()

# Plot TT, EE, BB, TE power spectra
plot_cmb_spectra(cmb_maps[0], lmax=config.lmax, dpi=300)
plt.show()

# Save to file
plot_cmb_spectra(cmb_maps[0], lmax=config.lmax, show=False,
                 save="cmb_spectra.png", dpi=300)
```

## Configuration Reference

### `CosmoConfig`

| Parameter | Description |
|-----------|-------------|
| `H0` | Hubble constant (km/s/Mpc) |
| `ombh2` | Baryon density |
| `omch2` | Cold dark matter density |
| `As` | Scalar amplitude |
| `ns` | Spectral index |
| `r` | Tensor-to-scalar ratio |

### `Config`

| Parameter | Description |
|-----------|-------------|
| `nside_out` | HEALPix Nside for output maps |
| `nside_cmb` | HEALPix Nside for CMB simulation |
| `nside_fg` | HEALPix Nside for foreground simulation |
| `fg_models` | List of PySM3 foreground model strings |
| `lmax` | Maximum multipole |
| `cosmo_params` | `CosmoConfig` instance |
| `spectra_type` | Power spectrum type (`"total"`, `"unlensed"`, etc.) |
| `freqs` | Array of frequency channels in GHz |
| `sens` | Per-channel noise sensitivity in µK·arcmin |
| `beam_arcmin` | Gaussian beam FWHM in arcminutes |
| `output_unit` | Output map unit (e.g. `"uK_CMB"`) |
| `extra_smoothing_fwhm_arcmin` | Additional smoothing applied during downgrading |
| `add_noise` | Whether to add noise maps |

## Package Structure

```
tinycmb/
├── __init__.py       # Exports Simulator, Config, CosmoConfig
├── simulator.py      # Simulator class
└── utils.py          # plot_cmb_spectra, downgrade_map_harmonic
```
