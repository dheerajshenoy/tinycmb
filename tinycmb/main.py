"""
TinyCMB: A simple CMB simulation toolkit based on CAMB, HEALPy, and PySM3.
This contains boilerplate code for simulating CMB maps, foregrounds, and noise, as well as utilities for downgrading maps and computing power spectra.
Author: Dheeraj Vittal Shenoy
"""

from typing import Dict, List

import astropy.units as u
import camb
import healpy as hp
import numpy as np
import pysm3

# Planck 2018 cosmological parameters (TT,TE,EE+lowE+lensing)
# Reference: Planck Collaboration 2018, Table 2
# https://arxiv.org/abs/1807.06209
PLANCK_2018_PARAMS = {
    "H0": 67.36,  # Hubble constant [km/s/Mpc]
    "ombh2": 0.02237,  # Baryon density
    "omch2": 0.1200,  # CDM density
    "tau": 0.0544,  # Optical depth
    "As": 2.1e-9,  # Scalar amplitude
    "ns": 0.9649,  # Spectral index
    "mnu": 0.06,  # Neutrino mass sum [eV]
    "omk": 0.0,  # Curvature
}


def downgrade_map(
    input_map: np.ndarray, input_nside: int, target_nside: int
) -> np.ndarray:
    """
    Correctly downgrade HEALPix map using harmonic space method.
    Handles both single maps and polarized [T, Q, U] maps.
    """
    lmax_in = 3 * input_nside - 1
    lmax_out = 3 * target_nside - 1

    # Check if polarized (3, npix) or single map (npix,)
    if input_map.ndim == 2 and input_map.shape[0] == 3:
        # Polarized: convert to alm_T, alm_E, alm_B
        alm_T, alm_E, alm_B = hp.map2alm(input_map, lmax=lmax_in, iter=3)
        downgraded_map = hp.alm2map(
            [alm_T, alm_E, alm_B],
            nside=target_nside,
            lmax=lmax_out,
        )
    else:
        # Single map
        alm = hp.map2alm(input_map, lmax=lmax_in, iter=3)
        downgraded_map = hp.alm2map(alm, nside=target_nside, lmax=lmax_out)

    return downgraded_map


def simulate_cmb(
    nside: int, cosmic_params: dict, lmax: int | None = None, seed: int | None = None
) -> np.ndarray:
    """
    Simulate CMB map using CAMB + HEALPy.

    Returns:
        Array of shape (3, npix) containing [T, Q, U] maps in μK
    """
    if seed is not None:
        np.random.seed(seed)

    if lmax is None:
        lmax = 3 * nside - 1

    pars = camb.set_params(
        H0=cosmic_params["H0"],
        ombh2=cosmic_params["ombh2"],
        omch2=cosmic_params["omch2"],
        tau=cosmic_params["tau"],
        As=cosmic_params["As"],
        ns=cosmic_params["ns"],
        mnu=cosmic_params.get("mnu", 0.06),
        omk=cosmic_params.get("omk", 0.0),
        lmax=lmax,
    )

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, lmax=lmax, CMB_unit="muK", raw_cl=True)

    total = powers["total"]
    cls = [total[:, 0], total[:, 1], total[:, 2], total[:, 3]]  # TT, EE, BB, TE

    maps = hp.synfast(cls, nside=nside, lmax=lmax, pol=True, new=True)
    return np.array(maps)  # Shape: (3, npix)


def simulate_foreground_single_freq(
    nside: int, freq_ghz: float, fg_models: List[str]
) -> np.ndarray:
    """
    Simulate foreground map at a single frequency using PySM3.

    Returns:
        Array of shape (3, npix) containing [I, Q, U] maps in μK_CMB
    """
    fg = pysm3.Sky(nside=nside, preset_strings=fg_models, output_unit="uK_CMB")
    emissions = fg.get_emission(freq_ghz * u.GHz)
    return emissions.value  # Shape: (3, npix)


def generate_white_noise(
    nside: int,
    sensitivities: float | List[float],
    is_half_split: bool = False,
    polarized: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate white noise maps.

    Returns:
        For single frequency: (3, npix) if polarized, (npix,) otherwise
        For multiple frequencies: (n_freq, 3, npix) if polarized
    """
    if seed is not None:
        np.random.seed(seed)

    sensitivities = np.atleast_1d(sensitivities)
    n_freq = len(sensitivities)
    npix = hp.nside2npix(nside)

    # Pixel area in arcmin²
    pix_amin2 = (4.0 * np.pi / npix) * (180.0 * 60.0 / np.pi) ** 2

    # Noise per pixel
    sigma_pix_I = np.sqrt(sensitivities**2 / pix_amin2)

    if is_half_split:
        sigma_pix_I *= np.sqrt(2.0)

    if polarized:
        noise = np.random.randn(n_freq, 3, npix)
        noise *= sigma_pix_I[:, None, None]
    else:
        noise = np.random.randn(n_freq, npix)
        noise *= sigma_pix_I[:, None]

    return noise.squeeze()


def apply_beam_smoothing(map_in: np.ndarray, beam_fwhm_arcmin: float) -> np.ndarray:
    """
    Apply Gaussian beam smoothing to map(s).

    Parameters:
        map_in: Shape (3, npix) for [T, Q, U] or (npix,) for single map
        beam_fwhm_arcmin: Beam FWHM in arcminutes

    Returns:
        Smoothed map with same shape as input
    """
    beam_fwhm_rad = np.radians(beam_fwhm_arcmin / 60.0)

    if map_in.ndim == 2 and map_in.shape[0] == 3:
        # Polarized maps
        smoothed = np.zeros_like(map_in)
        for i in range(3):
            smoothed[i] = hp.smoothing(map_in[i], fwhm=beam_fwhm_rad)
    else:
        # Single map
        smoothed = hp.smoothing(map_in, fwhm=beam_fwhm_rad)

    return smoothed


def simulate_full_sky(
    nside: int,
    lmax: int,
    cosmic_params: dict,
    freq_configs: List[Dict],
    fg_models: List[str],
    apply_beam: bool = True,
    cmb_seed: int = 42,
    noise_seed: int = 123,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Simulate full sky maps (CMB + foregrounds + noise) for multiple frequencies.

    Returns:
        Dictionary mapping frequency (GHz) to dict containing:
            'cmb': CMB maps [T, Q, U]
            'foreground': Foreground maps [I, Q, U]
            'noise': Noise maps [T, Q, U]
            'total': Total maps [T, Q, U]
            'beam_fwhm': Beam FWHM in arcmin
            'sensitivity': Noise sensitivity in μK·arcmin
    """
    print(f"\n{'=' * 70}")
    print("FULL SKY SIMULATION")
    print(f"{'=' * 70}")
    print(f"NSIDE = {nside}, LMAX = {lmax}")
    print(f"Number of frequencies: {len(freq_configs)}")
    print(f"Foreground models: {fg_models}")
    print(f"Apply beam smoothing: {apply_beam}")

    # Generate CMB once (frequency-independent)
    print(f"\n1. Generating CMB map (seed={cmb_seed})...")
    cmb_map = simulate_cmb(nside, cosmic_params, lmax, seed=cmb_seed)
    print(f"   CMB T RMS: {np.std(cmb_map[0]):.2f} μK")

    # Process each frequency
    results = {}

    for i, config in enumerate(freq_configs):
        freq = config["freq"]
        sens = config["sens"]
        beam = config.get("beam_fwhm", 10.0)

        print(
            f"\n{i + 1}. Processing {freq} GHz (sens={sens:.2f} μK·arcmin, beam={beam:.1f}')"
        )

        # CMB (same for all frequencies)
        cmb_freq = cmb_map.copy()

        # Foreground (frequency-dependent)
        print("   Generating foreground...")
        fg_freq = simulate_foreground_single_freq(nside, freq, fg_models)
        print(f"   FG I RMS: {np.std(fg_freq[0]):.2f} μK")

        # Noise (frequency-dependent)
        print(f"   Generating noise (seed={noise_seed + i})...")
        noise_freq = generate_white_noise(
            nside, sens, polarized=True, seed=noise_seed + i
        )
        print(f"   Noise T RMS: {np.std(noise_freq[0]):.2f} μK")

        # Combine
        total_freq = cmb_freq + fg_freq + noise_freq

        # Apply beam smoothing
        if apply_beam:
            print(f"   Applying beam smoothing (FWHM={beam:.1f}')...")
            total_freq = apply_beam_smoothing(total_freq, beam)

        print(f"   Total T RMS: {np.std(total_freq[0]):.2f} μK")

        # Store results
        results[freq] = {
            "cmb": cmb_freq,
            "foreground": fg_freq,
            "noise": noise_freq,
            "total": total_freq,
            "beam_fwhm": beam,
            "sensitivity": sens,
        }

    print(f"\n{'=' * 70}")
    print(f"✓ Simulation complete for {len(results)} frequencies")
    print(f"{'=' * 70}\n")

    return results


def save_maps(results: Dict, output_dir: str = "."):
    """
    Save all simulated maps to FITS files.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    for freq, data in results.items():
        # Save individual components
        hp.write_map(
            f"{output_dir}/cmb_freq{freq}GHz.fits", data["cmb"], overwrite=True
        )
        hp.write_map(
            f"{output_dir}/foreground_freq{freq}GHz.fits",
            data["foreground"],
            overwrite=True,
        )
        hp.write_map(
            f"{output_dir}/noise_freq{freq}GHz.fits", data["noise"], overwrite=True
        )
        hp.write_map(
            f"{output_dir}/total_freq{freq}GHz.fits", data["total"], overwrite=True
        )

    print(f"\n✓ Saved all maps to {output_dir}/")


def compute_power_spectra(
    maps: np.ndarray, lmax: int | None = None
) -> Dict[str, np.ndarray]:
    """
    Compute power spectra from maps.

    Parameters
    ----------
    maps : np.ndarray
        Input maps, shape (3, npix) for [T, Q, U] or (npix,) for T only
    lmax : int, optional
        Maximum multipole

    Returns
    -------
    dict
        Dictionary with keys 'TT', 'EE', 'BB', 'TE', 'EB', 'TB' (if polarized)
        or just 'TT' (if temperature only)
    """
    if maps.ndim == 2 and maps.shape[0] == 3:
        cls = hp.anafast(maps, lmax=lmax)
        return {
            "TT": cls[0],
            "EE": cls[1],
            "BB": cls[2],
            "TE": cls[3],
            "EB": cls[4],
            "TB": cls[5],
        }
    else:
        cl = hp.anafast(maps, lmax=lmax)
        return {"TT": cl}


def quick_sim(
    nside: int = 64,
    freq_ghz: float = 145,
    sensitivity: float = 5.0,
    beam_fwhm: float = 8.0,
    include_foregrounds: bool = True,
    include_noise: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """
    Quick simulation with sensible defaults.

    Parameters
    ----------
    nside : int, default=64
        HEALPix resolution
    freq_ghz : float, default=145
        Observation frequency in GHz
    sensitivity : float, default=5.0
        Noise sensitivity in μK·arcmin
    beam_fwhm : float, default=8.0
        Beam FWHM in arcminutes
    include_foregrounds : bool, default=True
        Whether to include foregrounds (s1, d1)
    include_noise : bool, default=True
        Whether to include noise
    seed : int, default=42
        Random seed

    Returns
    -------
    np.ndarray
        Total map [T, Q, U] in μK

    Examples
    --------
    >>> from tinycmb import quick_sim
    >>> total_map = quick_sim(nside=128, freq_ghz=94)
    """
    lmax = 3 * nside - 1

    # CMB
    cmb = simulate_cmb(nside, PLANCK_2018_PARAMS, lmax, seed=seed)
    total = cmb.copy()

    # Foregrounds
    if include_foregrounds:
        fg = simulate_foreground_single_freq(nside, freq_ghz, ["s1", "d1"])
        total += fg

    # Noise
    if include_noise:
        noise = generate_white_noise(nside, sensitivity, seed=seed + 100)
        total += noise

    # Beam
    total = apply_beam_smoothing(total, beam_fwhm)

    return total

