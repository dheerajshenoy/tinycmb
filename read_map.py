from tinycmb.utils import plot_cmb_spectra
from tinycmb import Simulator, Config, CosmoConfig
from tinycmb.utils import downgrade_map_harmonic, downgrade_mask
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.use("qtagg")  # Use a non-interactive backend for matplotlib

cosmo_config = CosmoConfig(
        H0=67.5, ombh2=0.022, omch2=0.122,
        As=2e-9, ns=0.965, r=0)

config = Config(
        nside_out = 512,
        nside_cmb = 2048,
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

total_map = hp.read_map("total.fits", field=(0, 1, 2))

mask = hp.read_map("HFI_Mask_GalPlane-apo2_2048_R2.00.fits", field=4).astype(bool)

mask_downgraded = downgrade_mask(mask, nside_out=config.nside_out)

total_map_masked = hp.ma(total_map)
total_map_masked.mask = np.logical_not(mask_downgraded)

# hp.mollview(total_map_masked.filled(), title="Simulated CMB T", unit="muK", min=-300, max=300)
# plt.savefig("cmb_map.png", dpi=300)

# plot_cmb_spectra(total_map, lmax=config.lmax, save="cmb_spectra.png", dpi=300)
# plot_cmb_spectra(total_map_masked, lmax=config.lmax, save="cmb_spectra_masked.png", dpi=300)
# plot_cmb_spectra((total_maps[0]), lmax=config.lmax, dpi=300)

plt.show()
