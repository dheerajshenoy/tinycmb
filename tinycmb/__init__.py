"""tinycmb - Simple CMB simulation toolkit."""

from .main import (
    simulate_cmb,
    simulate_foreground_single_freq,
    simulate_full_sky,
    generate_white_noise,
    downgrade_map,
    apply_beam_smoothing,
    save_maps,
    compute_power_spectra,
    quick_sim,
    PLANCK_2018_PARAMS,
)

__version__ = "0.1.0"

__all__ = [
    "simulate_cmb",
    "simulate_foreground_single_freq",
    "simulate_full_sky",
    "generate_white_noise",
    "downgrade_map",
    "apply_beam_smoothing",
    "save_maps",
    "compute_power_spectra",
    "quick_sim",
    "PLANCK_2018_PARAMS",
]
