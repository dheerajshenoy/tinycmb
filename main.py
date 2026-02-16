from typing import Dict, List, Tuple

import astropy.units as u
import camb
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pysm3

matplotlib.use("qtAgg")

# Configuration
NSIDE = 256
LMAX = 3 * NSIDE - 1
FG_MODELS = ["s1", "d1"]

# Frequency channels (GHz) and sensitivities (μK·arcmin)
# Based on typical CMB experiment specifications
FREQ_CONFIGS = [
    {"freq": 40, "sens": 122.723, "beam_fwhm": 53.4},
    {"freq": 61, "sens": 28.572, "beam_fwhm": 37.8},
    {"freq": 50, "sens": 43.878, "beam_fwhm": 42.5},
    {"freq": 77, "sens": 11.707, "beam_fwhm": 29.9},
    {"freq": 94, "sens": 7.019, "beam_fwhm": 22.6},
    {"freq": 145, "sens": 5.198, "beam_fwhm": 15.6},
    {"freq": 118, "sens": 4.576, "beam_fwhm": 18.5},
    {"freq": 182, "sens": 5.176, "beam_fwhm": 13.1},
    {"freq": 217, "sens": 12.365, "beam_fwhm": 10.3},
    {"freq": 334, "sens": 29.613, "beam_fwhm": 7.6},
    {"freq": 280, "sens": 16.105, "beam_fwhm": 8},
    {"freq": 402, "sens": 134.075, "beam_fwhm": 6.3},
]

# Planck 2018 cosmological parameters
COSMIC_PARAMS = {
    "H0": 67.36,
    "ombh2": 0.02237,
    "omch2": 0.1200,
    "tau": 0.0544,
    "As": 2.100e-9,
    "ns": 0.9649,
    "mnu": 0.06,
    "omk": 0.0,
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
            [alm_T, alm_E, alm_B], nside=target_nside, lmax=lmax_out, verbose=False
        )
    else:
        # Single map
        alm = hp.map2alm(input_map, lmax=lmax_in, iter=3)
        downgraded_map = hp.alm2map(
            alm, nside=target_nside, lmax=lmax_out, verbose=False
        )

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
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")

    total = powers["total"]
    cls = [total[:, 0], total[:, 1], total[:, 2], total[:, 3]]  # TT, EE, BB, TE

    maps = hp.synfast(cls, nside=nside, lmax=lmax, pol=True, new=True, verbose=False)
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
            smoothed[i] = hp.smoothing(map_in[i], fwhm=beam_fwhm_rad, verbose=False)
    else:
        # Single map
        smoothed = hp.smoothing(map_in, fwhm=beam_fwhm_rad, verbose=False)

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
    print(f"FULL SKY SIMULATION")
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
        print(f"   Generating foreground...")
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


if __name__ == "__main__":
    # Run full simulation
    results = simulate_full_sky(
        nside=NSIDE,
        lmax=LMAX,
        cosmic_params=COSMIC_PARAMS,
        freq_configs=FREQ_CONFIGS,
        fg_models=FG_MODELS,
        apply_beam=True,
        cmb_seed=42,
        noise_seed=123,
    )

    # Plot comparison across all frequencies
    # plot_frequency_comparison(
    #     results, nside=NSIDE, lmax=LMAX, save_name="simulation_multifreq.png"
    # )

    # Save all maps
    save_maps(results, output_dir="simulated_maps")

    # plt.show()
