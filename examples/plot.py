import tinycmb
import matplotlib.pyplot as plt
from typing import Dict
import healpy as hp
import numpy as np

from tinycmb.main import PLANCK_2018_PARAMS

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


def plot_component_comparison(
    results: Dict, freq: int, nside: int, lmax: int, save_prefix: str = ""
):
    """
    Plot comparison of CMB, foreground, noise, and total maps for a single frequency.
    """
    data = results[freq]

    fig = plt.figure(figsize=(16, 12))

    # Extract T maps
    cmb_T = data["cmb"][0]
    fg_T = data["foreground"][0]
    noise_T = data["noise"][0]
    total_T = data["total"][0]

    # Plot maps
    vmin, vmax = -300, 300

    hp.mollview(
        cmb_T,
        title=f"CMB Temperature ({freq} GHz)",
        sub=(3, 3, 1),
        cmap="RdBu_r",
        min=vmin,
        max=vmax,
        unit="μK",
        hold=True,
    )

    hp.mollview(
        fg_T,
        title=f"Foreground ({freq} GHz)",
        sub=(3, 3, 2),
        cmap="viridis",
        unit="μK",
        hold=True,
    )

    hp.mollview(
        noise_T,
        title=f"Noise ({freq} GHz, {data['sensitivity']:.1f} μK·arcmin)",
        sub=(3, 3, 3),
        cmap="gray",
        unit="μK",
        hold=True,
    )

    hp.mollview(
        total_T,
        title=f"Total Map ({freq} GHz)",
        sub=(3, 3, 4),
        cmap="RdBu_r",
        min=vmin,
        max=vmax,
        unit="μK",
        hold=True,
    )

    # Q polarization
    hp.mollview(
        data["total"][1],
        title=f"Q Polarization ({freq} GHz)",
        sub=(3, 3, 5),
        cmap="RdBu_r",
        unit="μK",
        hold=True,
    )

    # U polarization
    hp.mollview(
        data["total"][2],
        title=f"U Polarization ({freq} GHz)",
        sub=(3, 3, 6),
        cmap="RdBu_r",
        unit="μK",
        hold=True,
    )

    # Power spectra
    plt.subplot(3, 3, 7)

    # Compute power spectra
    cl_cmb = hp.anafast(data["cmb"], lmax=lmax)[2]  # BB
    cl_fg = hp.anafast(data["foreground"], lmax=lmax)[0]
    cl_total = hp.anafast(data["total"], lmax=lmax)[0]

    ell = np.arange(len(cl_cmb))
    dl_cmb = cl_cmb * ell * (ell + 1) / (2 * np.pi)
    dl_fg = cl_fg * ell * (ell + 1) / (2 * np.pi)
    dl_total = cl_total * ell * (ell + 1) / (2 * np.pi)

    plt.semilogy(ell[2:], dl_cmb[2:], label="CMB", linewidth=2)
    plt.semilogy(ell[2:], dl_fg[2:], label="Foreground", linewidth=2)
    plt.semilogy(ell[2:], dl_total[2:], label="Total", linewidth=2, alpha=0.7)

    plt.xlabel(r"Multipole $\ell$", fontsize=12)
    plt.ylabel(r"$D_\ell$ [$\mu K^2$]", fontsize=12)
    plt.title(f"BB Power Spectrum ({freq} GHz)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Histogram
    plt.subplot(3, 3, 8)
    plt.hist(cmb_T, bins=100, alpha=0.5, label="CMB", density=True)
    plt.hist(total_T, bins=100, alpha=0.5, label="Total", density=True)
    plt.xlabel("Temperature [μK]", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.title("Temperature Distribution", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Statistics table
    plt.subplot(3, 3, 9)
    plt.axis("off")
    stats_text = f"""
    Simulation Statistics ({freq} GHz)
    {"=" * 35}

    NSIDE: {nside}
    LMAX: {lmax}
    Beam FWHM: {data["beam_fwhm"]:.1f} arcmin
    Sensitivity: {data["sensitivity"]:.2f} μK·arcmin

    RMS Values:
    CMB T:        {np.std(cmb_T):8.2f} μK
    Foreground:   {np.std(fg_T):8.2f} μK
    Noise:        {np.std(noise_T):8.2f} μK
    Total:        {np.std(total_T):8.2f} μK

    CMB Q:        {np.std(data["cmb"][1]):8.2f} μK
    CMB U:        {np.std(data["cmb"][2]):8.2f} μK
    """
    plt.text(
        0.1,
        0.5,
        stats_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
    )

    plt.tight_layout()

    if save_prefix:
        filename = f"{save_prefix}_freq{freq}GHz.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved: {filename}")


def plot_frequency_comparison(
    results: Dict, nside: int, lmax: int, save_name: str = ""
):
    """
    Plot comparison across different frequencies.
    """
    freqs = sorted(results.keys())
    n_freq = len(freqs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Power spectra comparison
    ax = axes[0, 0]
    for freq in freqs:
        cl = hp.anafast(results[freq]["total"], lmax=lmax)[0]
        ell = np.arange(len(cl))
        dl = cl * ell * (ell + 1) / (2 * np.pi)
        ax.semilogy(ell[2:], dl[2:], label=f"{freq} GHz", alpha=0.7)

    ax.set_xlabel(r"Multipole $\ell$", fontsize=12)
    ax.set_ylabel(r"$D_\ell$ [$\mu K^2$]", fontsize=12)
    ax.set_title("BB Power Spectra (All Frequencies)", fontsize=13)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # RMS vs Frequency
    ax = axes[0, 1]
    cmb_rms = [np.std(results[f]["cmb"][0]) for f in freqs]
    fg_rms = [np.std(results[f]["foreground"][0]) for f in freqs]
    noise_rms = [np.std(results[f]["noise"][0]) for f in freqs]
    total_rms = [np.std(results[f]["total"][0]) for f in freqs]

    ax.plot(freqs, cmb_rms, "o-", label="CMB", linewidth=2)
    ax.plot(freqs, fg_rms, "s-", label="Foreground", linewidth=2)
    ax.plot(freqs, noise_rms, "^-", label="Noise", linewidth=2)
    ax.plot(freqs, total_rms, "d-", label="Total", linewidth=2)

    ax.set_xlabel("Frequency [GHz]", fontsize=12)
    ax.set_ylabel("RMS [μK]", fontsize=12)
    ax.set_title("RMS vs Frequency", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Sensitivity vs Frequency
    ax = axes[1, 0]
    sensitivities = [results[f]["sensitivity"] for f in freqs]
    ax.semilogy(freqs, sensitivities, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Frequency [GHz]", fontsize=12)
    ax.set_ylabel("Sensitivity [μK·arcmin]", fontsize=12)
    ax.set_title("Instrument Sensitivity", fontsize=13)
    ax.grid(True, alpha=0.3)

    # Beam FWHM vs Frequency
    ax = axes[1, 1]
    beams = [results[f]["beam_fwhm"] for f in freqs]
    ax.plot(freqs, beams, "o-", linewidth=2, markersize=8, color="orange")
    ax.set_xlabel("Frequency [GHz]", fontsize=12)
    ax.set_ylabel("Beam FWHM [arcmin]", fontsize=12)
    ax.set_title("Beam Size", fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_name:
        plt.savefig(save_name, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_name}")


if __name__ == "__main__":
    # Run full simulation
    results = tinycmb.simulate_full_sky(
        nside=NSIDE,
        lmax=LMAX,
        cosmic_params=PLANCK_2018_PARAMS,
        freq_configs=FREQ_CONFIGS,
        fg_models=FG_MODELS,
        apply_beam=True,
        cmb_seed=42,
        noise_seed=123,
    )

    # Plot comparison across all frequencies
    plot_frequency_comparison(
        results, nside=NSIDE, lmax=LMAX, save_name="simulation_multifreq.png"
    )

    # Save all maps
    # tinycmb.save_maps(results, output_dir="simulated_maps")

    plt.show()
