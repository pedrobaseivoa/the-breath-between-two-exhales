import os
import numpy as np
import matplotlib.pyplot as plt

from thermo_entropy import entropy_series
from cosmology_memory import H_series
from complex_breathing import complex_series
from neural_thresholds import ltp_response


def ensure_figdir(path: str = "figures") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def plot_entropy(figdir: str):
    cases = [(0.9, 0.0, "alpha=0.9, beta=0.0"), (1.0, 1.0, "alpha=1.0, beta=1.0"), (1.1, 0.0, "alpha=1.1, beta=0.0")]
    plt.figure(figsize=(8, 5))
    for alpha, beta, label in cases:
        out = entropy_series(150000, alpha, beta, lam=1e-4, rho=1.0)
        n = out["n"]
        S = out["S"]
        plt.plot(n, S, label=label)
    plt.xlabel("n")
    plt.ylabel("S(n)")
    plt.title("Entropy Accumulation across Regimes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "entropy_S.png"), dpi=150)
    plt.close()


def plot_cosmology(figdir: str):
    H = H_series(200000, alpha=1.0, beta=1.0, lam=1e-4, rho=1.0, H0=1.0)
    t = np.arange(2, 200000 + 2)
    plt.figure(figsize=(8, 5))
    plt.plot(t, H)
    plt.xlabel("t (index)")
    plt.ylabel("H(t) (model units)")
    plt.title("Cosmology: Memory-Weighted Expansion H(t)")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "cosmology_H.png"), dpi=150)
    plt.close()


def plot_complex_breathing(figdir: str):
    N = 200000
    n, a, phase, M = complex_series(N=N,
                                    alpha_r=1.0,
                                    beta_r=1.0,
                                    alpha_i=2.8,
                                    beta_i=2.9,
                                    lam=1e-4,
                                    rho=1.0)
    mag = np.abs(a)
    plt.figure(figsize=(8, 4))
    plt.plot(n[:2000], mag[:2000])
    plt.xlabel("n")
    plt.ylabel("|a_n|")
    plt.title("Complex Breathing: Magnitude (first 2k samples)")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "breathing_magnitude.png"), dpi=150)
    plt.close()

    # Simple spectrum of magnitude (for visualization only)
    mag_centered = mag - np.mean(mag)
    fft = np.fft.rfft(mag_centered)
    freqs = np.fft.rfftfreq(len(mag_centered), d=1.0)
    psd = np.abs(fft)
    plt.figure(figsize=(8, 4))
    plt.semilogy(freqs[1:5000], psd[1:5000])
    plt.xlabel("frequency (index^-1)")
    plt.ylabel("|FFT(|a_n|)|")
    plt.title("Complex Breathing: Spectrum of |a_n|")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "breathing_spectrum.png"), dpi=150)
    plt.close()


def plot_neural(figdir: str):
    out = ltp_response(200000, alpha=1.0, beta=1.0, lam=1e-3, rho=1.0, stim=1.0)
    n = out["n"]
    W = out["W"]
    plt.figure(figsize=(8, 5))
    plt.plot(n, W)
    plt.xlabel("n")
    plt.ylabel("A(n) ~ W(n)")
    plt.title("Awareness Accumulation (Model)")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "awareness_accumulation.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    figdir = ensure_figdir()
    plot_entropy(figdir)
    plot_cosmology(figdir)
    plot_complex_breathing(figdir)
    plot_neural(figdir)
    print({"saved": [
        "figures/entropy_S.png",
        "figures/cosmology_H.png",
        "figures/breathing_magnitude.png",
        "figures/breathing_spectrum.png",
        "figures/awareness_accumulation.png",
    ]})


