from tinycmb.utils import plot_cmb_spectra
from tinycmb import Simulator, Config, CosmoConfig
from tinycmb.utils import downgrade_map_harmonic
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

cosmo_config = CosmoConfig(
        H0=67.5, ombh2=0.022, omch2=0.122,
        As=2e-9, ns=0.965, r=0)

config = Config(
        nside_out = 64,
        nside_cmb = 512,
        nside_fg = 512,
        fg_models = ["s0", "d0"],
        lmax = None,
        cosmo_params = cosmo_config,
        spectra_type="total",
        freqs = np.array([
            50.0, 78.0,  # LFT
            100.0, 119.0, 140.0, 166.0,  # MFT
            195.0, 235.0, 280.0, 337.0 #, 402.0  # HFT
            ]),
        sens = np.array([
            32.78, 18.59, 12.93, 9.79, 9.55, 5.81, 7.12, 15.16, 17.98, 24.99
            ]),
        beam_arcmin = 15.0,
        output_unit = "uK_CMB",
        extra_smoothing_fwhm_arcmin = 0.0,
        add_noise = True,
        mask = None,
        )

simulator = Simulator(config)

cmb_maps = simulator.simulate_cmb()

fg_maps = simulator.simulate_foregrounds()

cmb_maps = downgrade_map_harmonic(
        cmb_maps,
        nside_in=config.nside_cmb,
        nside_out=config.nside_out,
        beam_fwhm_arcmin=config.beam_arcmin,
        extra_smoothing_fwhm_arcmin=config.extra_smoothing_fwhm_arcmin,
        lmax_out=config.lmax,
        iter=0)

hp.write_map(f"cmb_map_{config.nside_out}.fits", cmb_maps[0], overwrite=True)

fg_maps = downgrade_map_harmonic(
        fg_maps,
        nside_in=config.nside_fg,
        nside_out=config.nside_out,
        beam_fwhm_arcmin=config.beam_arcmin,
        extra_smoothing_fwhm_arcmin=2.5,
        lmax_out=config.lmax,
        iter=0)

noise_maps = simulator.simulate_noise()
print("Noise = ", noise_maps.shape)

total_maps = cmb_maps + fg_maps + noise_maps

for i in range(noise_maps.shape[0]):
    hp.write_map(f"fg_{config.nside_out}_{"_".join(config.fg_models)}_{str(config.freqs[i]).replace(".", "_")}GHz.fits", fg_maps[i], overwrite=True)
    hp.write_map(f"noise_{config.nside_out}_{str(config.freqs[i]).replace(".", "_")}GHz.fits", noise_maps[i], overwrite=True)

print("Total = ", total_maps.shape)
