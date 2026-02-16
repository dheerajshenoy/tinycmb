from re import PatternError
from typing import List

import astropy.units as u
import camb
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pysm3

matplotlib.use("qtAgg")

NSIDE = 64
FG_MODELS = ["s1", "d1"]
LMAX = 3 * NSIDE - 1

# Planck 2018 best-fit parameters (TT,TE,EE+lowE+lensing)
# Reference: Planck Collaboration 2018, Table 2 (last column)
# https://arxiv.org/abs/1807.06209
COSMIC_PARAMS = {
    "H0": 67.36,  # Hubble constant at z=0 in km/s/Mpc
    "ombh2": 0.02237,  # Physical baryon density parameter
    "omch2": 0.1200,  # Physical cold dark matter density parameter
    "tau": 0.0544,  # Reionization optical depth
    "As": 2.100e-9,  # Amplitude of the primordial curvature perturbations
    "ns": 0.9649,  # Spectral index of the primordial power spectrum
}


def simulate_cmb(nside: int, cosmic_params: dict) -> np.ndarray:
    """
    Simulate a CMB map using the CAMB library + Healpy based on the provided cosmological parameters.

    Returns a CMB healpy map in microkelvin (muK) units.
    """
    pars = camb.set_params(
        H0=cosmic_params["H0"],
        ombh2=cosmic_params["ombh2"],
        omch2=cosmic_params["omch2"],
        tau=cosmic_params["tau"],
        As=cosmic_params["As"],
        ns=cosmic_params["ns"],
    )
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")

    cls = powers["total"]
    TT, EE, BB, TE = cls[:, 0], cls[:, 1], cls[:, 2], cls[:, 3]

    return hp.synfast([TT, EE, BB, TE], nside=nside, lmax=LMAX, pol=True, new=True)


def simulate_foreground(
    nside: int, freqs: u.Quantity[u.GHz], fg_models: List[str], output_unit
) -> np.ndarray:
    """
    Simulate a foreground map using the PySM3 library based on the provided foreground models.

    Returns a foreground healpy map in microkelvin (muK) units.
    """

    fg = pysm3.Sky(nside=nside, preset_strings=fg_models, output_unit=output_unit)
    emissions = fg.get_emission(freqs)

    # Construct a healpy map with the same shape as the CMB map (nside, 3 pol components)
    fg_map = np.zeros((3, hp.nside2npix(nside)))
    fg_map[0] = emissions[0]  # Intensity (I)
    fg_map[1] = emissions[1]  # Q polarization
    fg_map[2] = emissions[2]  # U polarization
    return fg_map


def simulate_noise(): ...


if __name__ == "__main__":
    cmb_map = simulate_cmb(NSIDE, COSMIC_PARAMS)
    fg_map = simulate_foreground(NSIDE, 150 * u.GHz, FG_MODELS, output_unit="uK_CMB")

    # Add the CMB and foreground maps together
    total_map = cmb_map + fg_map

    hp.mollview(
        total_map[0], title="Simulated CMB + Foreground Intensity (I)", unit="uK"
    )
    hp.mollview(
        total_map[1], title="Simulated CMB + Foreground Q Polarization", unit="uK"
    )
    hp.mollview(
        total_map[2], title="Simulated CMB + Foreground U Polarization", unit="uK"
    )

    plt.show()
    noise_map = simulate_noise()
