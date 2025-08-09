import numpy as np
from typing import Tuple


def baseline_term(n: np.ndarray, alpha_r: float, beta_r: float) -> np.ndarray:
    ln_n = np.log(n)
    ln_n = np.where(ln_n == 0.0, np.finfo(float).tiny, ln_n)
    return 1.0 / (np.power(n, alpha_r) * np.power(ln_n, beta_r))


def f_memory(M: np.ndarray, lam: float, rho: float) -> np.ndarray:
    return np.exp(-lam * np.power(M, rho))


def complex_series(N: int,
                   alpha_r: float,
                   beta_r: float,
                   alpha_i: float,
                   beta_i: float,
                   lam: float,
                   rho: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = np.arange(2, N + 2, dtype=float)
    b = baseline_term(n, alpha_r, beta_r)
    a = np.empty(N, dtype=np.complex128)
    phase = alpha_i * np.log(n) + beta_i * np.log(np.log(n))
    M_real = np.empty(N, dtype=float)
    accum = 0.0
    for i in range(N):
        fm = f_memory(np.array([accum]), lam, rho)[0]
        a_i = b[i] * np.exp(-1j * phase[i]) * fm
        a[i] = a_i
        accum += np.abs(a_i)
        M_real[i] = accum
    return n, a, phase, M_real


def breathing_metrics(a: np.ndarray,
                      n: np.ndarray,
                      scale_frac: float = 0.5) -> dict:
    # Use relative scale target based on typical magnitude
    mag = np.abs(a)
    target = np.median(mag[mag > 0])
    tol = scale_frac * target
    near = np.abs(mag - target) < tol
    residence = int(np.sum(near))
    amp = float(np.std(mag[near])) if residence > 0 else float(np.std(mag))
    return {
        "residence": residence,
        "amplitude_std": amp,
        "mean_mag": float(np.mean(mag)),
    }


if __name__ == "__main__":
    n, a, phase, M = complex_series(N=400000,
                                    alpha_r=1.0,
                                    beta_r=1.0,
                                    alpha_i=2.8,
                                    beta_i=2.9,
                                    lam=1e-4,
                                    rho=1.0)
    metrics = breathing_metrics(a, n, scale_frac=0.5)
    print(metrics)


