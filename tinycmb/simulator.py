import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import camb
import astropy.units as u
from typing import List, Optional
from dataclasses import dataclass
import pysm3

@dataclass
class CosmoConfig:
    H0: float  # Hubble constant
    ombh2: float  # Baryon density
    omch2: float  # Cold dark matter density
    As: float  # Amplitude of the primordial power spectrum
    ns: float  # Spectral index
    r: float  # Tensor-to-scalar ratio

@dataclass
class Config:
    nside_out: int
    nside_cmb: int
    nside_fg: int
    fg_models: List[str]
    lmax: int | None
    cosmo_params: CosmoConfig
    freqs: np.ndarray
    sens: np.ndarray
    beam_arcmin: float | np.ndarray
    spectra_type: str
    output_unit: str
    extra_smoothing_fwhm_arcmin: float
    add_noise: bool
    mask: str | None
    """
    nside_out: HEALPix nside for output maps (e.g., 64)
    nside_cmb: HEALPix nside for CMB simulation (e.g., 512)
    nside_fg: HEALPix nside for foreground simulation (e.g., 512)
    fg_models: List of foreground model preset strings for PySM3 (e.g., ["s5", "d10", "a2"])
    lmax: Maximum multipole for CMB power spectra (if None, defaults to 3*nside_out-1)
    cosmo_params: Cosmological parameters for CAMB
    freqs: Array of frequency channels in GHz (e.g., [50, 78, 100, 119, 140, 166, 195, 235, 280, 337])
    sens: Array of noise sensitivities in uK-arcmin for each frequency channel (e.g., [32.78, 18.59, 12.93, 9.79, 9.55, 5.81, 7.12, 15.16, 17.98, 24.99])
    beam_arcmin: Beam FWHM in arcminutes for Gaussian smoothing (can be a single float or an array of length n_freq)
    spectra_type: Type of CAMB power spectra to use (e.g., "total", "unlensed_scalar", "lensed_scalar", "tensor")
    output_unit: Unit for output maps (e.g., "uK_CMB")
    extra_smoothing_fwhm_arcmin: Additional Gaussian smoothing FWHM in arcminutes to apply to maps (e.g., 0.0 for no extra smoothing)
    add_noise: Whether to add noise to the simulated maps (True or False)
    mask: File path to a HEALPix mask map (optional, can be None for no mask)
    """

class Simulator:
    def __init__(self, config):
        self.config = config
        if self.config.lmax is None:
            self.config.lmax = 3 * self.config.nside_out - 1

    def __simulate_single_cmb(self):
        """
        Generates a simulated CMB map based on the provided configuration.

        Returns:
            np.ndarray: Simulated CMB map in HEALPix format.
        """
        cosparams = self.config.cosmo_params
        lmax = self.config.lmax
        # Set up CAMB parameters
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=cosparams.H0, ombh2=cosparams.ombh2, omch2=cosparams.omch2)
        pars.InitPower.set_params(As=cosparams.As, ns=cosparams.ns, r=cosparams.r)
        pars.WantTensors = True
        pars.set_for_lmax(lmax)

        # Get the power spectra
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(
                params=pars,
                lmax=lmax,
                CMB_unit="muK",
                raw_cl=True,
                )

        if self.config.spectra_type not in powers:
            raise ValueError(
                    f"spectra_type='{self.config.spectra_type}' not in CAMB outputs: {list(powers.keys())}"
                    )

        cls = powers[self.config.spectra_type]
        ell = np.arange(cls.shape[0])

        # Return as [ell, TT, EE, BB, TE]
        spectra = np.vstack([ell, cls[:, 0], cls[:, 1], cls[:, 2], cls[:, 3]])
        cl_tt = spectra[1, :lmax+1]
        cl_ee = spectra[2, :lmax+1]
        cl_bb = spectra[3, :lmax+1]
        cl_te = spectra[4, :lmax+1]

        # Generate maps using synfast
        # synfast expects [TT, EE, BB, TE, (TB=0, EB=0)]
        cmb_maps = hp.synfast(
                [cl_tt, cl_ee, cl_bb, cl_te],
                nside=self.config.nside_out,
                pol=True,
                new=True,
               )

        return np.array(cmb_maps, dtype=np.float32)

    def simulate_cmb(self):
        cmb_single = self.__simulate_single_cmb()

        npix = hp.nside2npix(self.config.nside_out)
        n_freq = len(self.config.freqs)
        cmb_maps = np.zeros((n_freq, 3, npix), dtype=np.float32)

        for i in range(n_freq):
            cmb_maps[i] = cmb_single

        return cmb_maps

    def simulate_foregrounds(self):
        sky = pysm3.Sky(nside=self.config.nside_out, preset_strings=self.config.fg_models, output_unit=self.config.output_unit)

        n_freq = len(self.config.freqs)
        npix = hp.nside2npix(self.config.nside_out)
        foregrounds = np.zeros((n_freq, 3, npix), dtype=np.float32)

        for i, freq in enumerate(self.config.freqs):
            emission = sky.get_emission(freq * u.GHz)
            foregrounds[i] = emission.value.astype(np.float32)

        return foregrounds

    def simulate_noise(self, seed = None):
        if self.config.add_noise is False:
            return np.zeros((len(self.config.freqs), 3, hp.nside2npix(self.config.nside_out)), dtype=np.float32)
        else:
            if seed is not None:
                np.seed(seed)
            n_freq = len(self.config.freqs)
            extra_smoothing_fwhm = self.config.extra_smoothing_fwhm_arcmin
            if self.config.nside_out < self.config.nside_cmb:
                noise_out = self.__generate_noise_native_then_downgrade(
                        nside_cmb=self.config.nside_cmb,
                        nside_out=self.config.nside_out,
                        )
            else:
                noise_out = self.__generate_noise(
                        nside=self.config.nside_out,
                        )

            if self.config.extra_smoothing_fwhm_arcmin is not None and self.config.extra_smoothing_fwhm_arcmin > 0.0:
                noise_out = self.__apply_beam_multifreq(
                        maps = noise_out,
                        fwhm_arcmin=np.full(n_freq, extra_smoothing_fwhm),
                        )
            return noise_out

    def __apply_beam_multifreq(
            self,
            maps: np.ndarray,
            fwhm_arcmin: np.ndarray,
            ) -> np.ndarray:
        """Apply different Gaussian beams to each frequency channel.

        Parameters
        ----------
        maps : np.ndarray
            Input maps of shape (n_freq, 3, npix)
        fwhm_arcmin : np.ndarray
            Beam FWHM in arcminutes for each frequency
        nside : int
            HEALPix nside

        Returns
        -------
        np.ndarray
            Beam-convolved maps
        """
        n_freq = maps.shape[0]
        result = np.zeros_like(maps)

        for i in range(n_freq):
            fwhm_rad = fwhm_arcmin[i] * np.pi / 10800.0
            result[i] = hp.smoothing(maps[i], fwhm=fwhm_rad, pol=True)

        return result

    def __generate_noise(
            self,
            nside: int,
            half_split: bool = True,
            seed: Optional[int] = None,
            ) -> np.ndarray:
        """Generate instrument noise maps at multiple frequencies.

        Generates white noise maps based on the instrument noise sensitivity
        specifications. The noise is generated independently for each Stokes
        parameter (T, Q, U) at each frequency.

        Parameters
        ----------
        instrument : LiteBIRDConfig
            Instrument configuration with noise specifications
        nside : int
            HEALPix nside for output maps
        half_split : bool
            If True, increase noise by sqrt(2) for half-mission split maps.
            This simulates the noise increase when using only half the data.
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        np.ndarray
            Noise maps of shape (n_freq, 3, npix) in uK_CMB
            where 3 corresponds to T, Q, U Stokes parameters
        """
        if seed is not None:
            np.random.seed(seed)

        n_freq = len(self.config.freqs)
        npix = hp.nside2npix(nside)

        # Pixel solid angle in arcmin^2
        pix_amin2 = 4.0 * np.pi / npix * (180.0 * 60.0 / np.pi) ** 2

        # Noise standard deviation per pixel for each frequency
        # noise_pol is in uK-arcmin, we convert to uK per pixel
        sigma_pix = np.sqrt(self.config.sens**2 / pix_amin2)

        if half_split:
            sigma_pix *= np.sqrt(2)

        # Generate noise maps
        noise = np.random.randn(n_freq, 3, npix).astype(np.float32)
        noise *= sigma_pix[:, None, None]

        return noise


    def __generate_noise_native_then_downgrade(
            self,
            nside_cmb: int,
            nside_out: int,
            half_split: bool = True,
            seed: Optional[int] = None,
            ) -> np.ndarray:
        """Generate white noise at native resolution and downgrade.

        This matches the simulator pattern used for sky signal components:
            create maps at ``nside_native`` and then downgrade to ``nside_out``.

        Important: the noise is NOT beam-smoothed.

        Downgrading is performed with ``hp.ud_grade`` (pixel-averaging), which is
        appropriate for uncorrelated per-pixel white noise.

        Returns
        -------
        np.ndarray
            Noise maps of shape (n_freq, 3, npix_out) in uK_CMB.
        """
        if nside_out > nside_cmb:
            raise ValueError(f"nside_out ({nside_out}) must be <= nside_cmb ({nside_cmb})")

        if nside_out == nside_cmb:
            return self.__generate_noise(
                    nside=nside_out,
                    half_split=half_split,
                    seed=seed,
                    )

        if seed is not None:
            np.random.seed(seed)

        n_freq = len(self.config.freqs)
        npix_native = hp.nside2npix(nside_cmb)
        npix_out = hp.nside2npix(nside_out)

        # Pixel solid angle in arcmin^2
        pix_amin2_native = 4.0 * np.pi / npix_native * (180.0 * 60.0 / np.pi) ** 2

        # noise_pol is in uK-arcmin, we convert to uK per pixel at native resolution
        sigma_pix_native = np.sqrt(self.config.sens**2 / pix_amin2_native)
        if half_split:
            sigma_pix_native *= np.sqrt(2)

        noise_out = np.empty((n_freq, 3, npix_out), dtype=np.float32)
        for i in range(n_freq):
            noise_native = (np.random.randn(3, npix_native).astype(np.float32) * sigma_pix_native[i])
            noise_out[i] = hp.ud_grade(noise_native, nside_out).astype(np.float32)

        return noise_out


