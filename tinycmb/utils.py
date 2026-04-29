import numpy as np
from typing import Optional
import healpy as hp
import matplotlib.pyplot as plt

def downgrade_map_harmonic(
    maps: np.ndarray,
    nside_in: int,
    nside_out: int,
    beam_fwhm_arcmin: Optional[np.ndarray] = None,
    extra_smoothing_fwhm_arcmin: Optional[float] = None,
    lmax_out: Optional[int] = None,
    iter: int = 0,
) -> np.ndarray:
    """Downgrade HEALPix maps using harmonic space operations.

    This method properly accounts for pixel window functions when
    downgrading, avoiding the artifacts introduced by hp.ud_grade.

    The procedure:
    1. Transform maps to alm
    2. Deconvolve the input pixel window function
    3. Convolve with the output pixel window function
    4. Transform back to maps at lower nside

    Parameters
    ----------
    maps : np.ndarray
        Input maps of shape (n_freq, 3, npix_in) or (3, npix_in)
        where 3 = T, Q, U Stokes parameters
    nside_in : int
        Input HEALPix nside
    nside_out : int
        Output HEALPix nside (must be <= nside_in)
    beam_fwhm_arcmin : np.ndarray or float, optional
        Gaussian beam FWHM(s) in arcminutes to apply during downgrade.
    extra_smoothing_fwhm_arcmin : float, optional
        Additional Gaussian smoothing FWHM in arcminutes applied during downgrade.
    lmax_out : int, optional
        Maximum multipole for downgrade (defaults to 3*nside_out-1).
    iter : int
        Iterations for map2alm (default 0 for speed on band-limited maps).

    Returns
    -------
    np.ndarray
        Downgraded maps of shape (n_freq, 3, npix_out) or (3, npix_out)
    """
    if nside_out > nside_in:
        raise ValueError(f"nside_out ({nside_out}) must be <= nside_in ({nside_in})")

    if (
        nside_out == nside_in
        and beam_fwhm_arcmin is None
        and extra_smoothing_fwhm_arcmin is None
    ):
        return maps.copy()

    # Get pixel window functions
    # pixwin returns (T_window, P_window) for pol=True
    pixwin_in = hp.pixwin(nside_in, pol=True)
    pixwin_out = hp.pixwin(nside_out, pol=True)

    if lmax_out is None:
        lmax_out = 3 * nside_out - 1

    # Handle different input shapes
    single_freq = maps.ndim == 2
    if single_freq:
        maps = maps[np.newaxis, ...]  # Add frequency dimension

    n_freq = maps.shape[0]
    npix_out = hp.nside2npix(nside_out)
    result = np.zeros((n_freq, 3, npix_out), dtype=np.float32)

    extra_beam = None
    if extra_smoothing_fwhm_arcmin is not None:
        extra_fwhm_rad = extra_smoothing_fwhm_arcmin * np.pi / 10800.0
        extra_beam = hp.gauss_beam(extra_fwhm_rad, lmax=lmax_out)

    for i in range(n_freq):
        # Get T, Q, U maps
        map_tqu = maps[i]  # Shape (3, npix_in)

        # Transform to alm (returns alm_T, alm_E, alm_B)
        # Compute directly at lmax_out to avoid ambiguous filter length behavior.
        alm = hp.map2alm(map_tqu, pol=True, lmax=lmax_out, iter=iter)
        alm_T, alm_E, alm_B = alm[0], alm[1], alm[2]

        # Deconvolve input pixel window and convolve output pixel window
        # For T: use pixwin[0] (temperature)
        # For E, B: use pixwin[1] (polarization)
        fl_T = pixwin_out[0][:lmax_out + 1] / np.clip(pixwin_in[0][:lmax_out + 1], 1e-10, None)
        fl_P = pixwin_out[1][:lmax_out + 1] / np.clip(pixwin_in[1][:lmax_out + 1], 1e-10, None)

        if beam_fwhm_arcmin is not None:
            if np.ndim(beam_fwhm_arcmin) == 0:
                fwhm_arcmin = float(beam_fwhm_arcmin)
            else:
                fwhm_arcmin = float(beam_fwhm_arcmin[i])
            fwhm_rad = fwhm_arcmin * np.pi / 10800.0
            beam = hp.gauss_beam(fwhm_rad, lmax=lmax_out)
            fl_T = fl_T * beam
            fl_P = fl_P * beam

        if extra_beam is not None:
            fl_T = fl_T * extra_beam
            fl_P = fl_P * extra_beam

        # Apply to alm
        alm_T_out = hp.almxfl(alm_T, fl_T)
        alm_E_out = hp.almxfl(alm_E, fl_P)
        alm_B_out = hp.almxfl(alm_B, fl_P)

        # Transform back to map space at lower nside
        result[i] = hp.alm2map(
            [alm_T_out, alm_E_out, alm_B_out],
            nside=nside_out,
            lmax=lmax_out,
            pol=True,
        )

    if single_freq:
        return result[0]
    return result

def plot_cmb_spectra(maps, lmax=None, save="cmb_spectra.png", **kwargs):
    """Plot the power spectra of the input maps."""
    # anafast returns (TT, EE, BB, TE, EB, TB) if pol=True
    cl = hp.anafast(maps, lmax=lmax, pol=True)
    ell = np.arange(cl.shape[1])

    fig, ax = plt.subplots(figsize=(12, 8), nrows=2, ncols=2)
    labels = ["TT", "EE", "BB", "TE"]

    for i in range(4):
        I, J = i // 2, i % 2

        # Calculate D_ell = ell(ell+1)Cl / 2pi
        # Use np.maximum to avoid log issues with zero or noise
        dl = cl[i] * ell * (ell + 1) / (2 * np.pi)

        if labels[i] == "TE":
            # TE can be negative; linear scale often better, or plot abs(TE)
            ax[I, J].plot(ell, dl, label=labels[i])
            ax[I, J].axhline(0, color='k', linestyle='--', alpha=0.3)
        else:
            ax[I, J].plot(ell, dl, label=labels[i])
            # ax[I, J].set_yscale("log")
        ax[I, J].set_yscale("linear")

        ax[I, J].set_title(f"{labels[i]} Spectrum")
        ax[I, J].set_xlabel(r"Multipole $\ell$")
        ax[I, J].set_ylabel(r"$D_\ell$ [$\mu K^2$]") # Standard units
        ax[I, J].legend()
        ax[I, J].set_xlim(4, lmax if lmax is not None else ell[-1])
        if labels[i] == "TT":
            ax[I, J].set_ylim(10, 1e4)   # adjust lower bound to taste

    plt.tight_layout()

    if save is not None:
        plt.savefig(save, **kwargs)

