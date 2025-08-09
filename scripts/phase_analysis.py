import numpy as np
import matplotlib.pyplot as plt
import os

from complex_breathing import complex_series


def ensure_figdir(path: str = "figures") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def analyze_phase(N: int = 300000,
                  alpha_r: float = 1.0,
                  beta_r: float = 1.0,
                  alpha_i: float = 2.8,
                  beta_i: float = 2.9,
                  lam: float = 1e-4,
                  rho: float = 1.0,
                  figdir: str = "figures"):
    n, a, phase, M = complex_series(N, alpha_r, beta_r, alpha_i, beta_i, lam, rho)
    # Unwrap phase to avoid 2pi jumps
    ph = np.unwrap(phase)
    # Local slope d(phase)/d(ln n) ~ alpha_i + beta_i / ln n
    ln_n = np.log(n)
    # Numerical derivative wrt ln n
    dphi = np.gradient(ph, ln_n)

    # Plot phase vs ln n and derivative
    plt.figure(figsize=(8, 4))
    plt.plot(ln_n[:5000], ph[:5000])
    plt.xlabel("ln n")
    plt.ylabel("phase")
    plt.title("Phase vs ln n (first 5k)")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "phase_vs_lnn.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(ln_n[100:10000], dphi[100:10000])
    plt.xlabel("ln n")
    plt.ylabel("d phase / d ln n")
    plt.title("Local slope of phase (indicative of alpha_i + beta_i/ln n)")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "phase_slope.png"), dpi=150)
    plt.close()

    # Spectrum of complex series (real/imag)
    # Use real/imag parts separately for FFT
    a_real = np.real(a)
    a_imag = np.imag(a)
    a_real_c = a_real - np.mean(a_real)
    a_imag_c = a_imag - np.mean(a_imag)
    A_r = np.fft.rfft(a_real_c)
    A_i = np.fft.rfft(a_imag_c)
    freqs = np.fft.rfftfreq(len(a_real_c), d=1.0)
    plt.figure(figsize=(8, 4))
    plt.semilogy(freqs[1:5000], np.abs(A_r[1:5000]), label='real')
    plt.semilogy(freqs[1:5000], np.abs(A_i[1:5000]), label='imag')
    plt.xlabel("frequency (index^-1)")
    plt.ylabel("|FFT| of a_n components")
    plt.title("Spectrum of complex series a_n (real/imag)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "complex_series_spectrum.png"), dpi=150)
    plt.close()

    # Robust linear fit of phase vs ln n over a window
    k = 20000
    x = ln_n[100:100 + k]
    y = ph[100:100 + k]
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    with open(os.path.join(figdir, "phase_fit.txt"), "w", encoding="utf-8") as f:
        f.write(f"slope_over_window≈{m}\nintercept≈{c}\n")


if __name__ == "__main__":
    figdir = ensure_figdir()
    analyze_phase(figdir=figdir)
    print({"saved": [
        "figures/phase_vs_lnn.png",
        "figures/phase_slope.png",
        "figures/complex_series_spectrum.png",
    ]})


