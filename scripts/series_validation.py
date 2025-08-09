import numpy as np
from typing import Tuple, Callable


def baseline_term(n: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    ln_n = np.log(n)
    ln_n = np.where(ln_n == 0.0, np.finfo(float).tiny, ln_n)
    return 1.0 / (np.power(n, alpha) * np.power(ln_n, beta))


def f_memory(M: np.ndarray, lam: float, rho: float) -> np.ndarray:
    return np.exp(-lam * np.power(M, rho))


def memory_weighted_series(N: int,
                           alpha: float,
                           beta: float,
                           lam: float,
                           rho: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.arange(2, N + 2, dtype=float)  # start at 2 for ln n
    b = baseline_term(n, alpha, beta)
    a = np.empty_like(b)
    M = np.empty_like(b)
    accum = 0.0
    for i in range(N):
        m_val = accum
        fm = f_memory(np.array([m_val]), lam, rho)[0]
        ai = b[i] * fm
        a[i] = ai
        accum += ai
        M[i] = accum
    return n, a, M


def windowed_growth(M: np.ndarray, window: int = 10000) -> float:
    if len(M) < 2 * window:
        window = max(10, len(M) // 10)
    start = len(M) - 2 * window
    end = len(M)
    if start < 0:
        start = 0
    delta = M[end - 1] - M[start]
    return delta / max(1, (end - start))


def tail_mean(a: np.ndarray, window: int = 10000) -> float:
    if len(a) < window:
        window = max(10, len(a) // 5)
    return float(np.mean(a[-window:]))


def classify(N: int,
             alpha: float,
             beta: float,
             lam: float,
             rho: float) -> dict:
    n, a, M = memory_weighted_series(N, alpha, beta, lam, rho)
    wg = windowed_growth(M)
    tm = tail_mean(a)
    theoretical = (
        "convergent" if (alpha > 1.0 or (abs(alpha - 1.0) < 1e-12 and beta > 1.0)) else "divergent"
    )
    result = {
        "N": N,
        "alpha": alpha,
        "beta": beta,
        "lam": lam,
        "rho": rho,
        "sum": float(M[-1]),
        "windowed_growth": float(wg),
        "tail_mean": float(tm),
        "classification": theoretical,
    }
    return result


if __name__ == "__main__":
    configs = [
        (0.9, 0.0), (1.0, 0.0), (1.0, 1.0), (1.1, 0.0), (1.0, 1.5)
    ]
    lam_vals = [0.0, 1e-4, 1e-3]
    rho_vals = [0.5, 1.0, 2.0]
    N = 800000
    for (alpha, beta) in configs:
        for lam in lam_vals:
            for rho in rho_vals:
                res = classify(N, alpha, beta, lam, rho)
                print(res)


