import numpy as np


def baseline_term(n: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    ln_n = np.log(n)
    ln_n = np.where(ln_n == 0.0, np.finfo(float).tiny, ln_n)
    return 1.0 / (np.power(n, alpha) * np.power(ln_n, beta))


def f_memory(M: np.ndarray, lam: float, rho: float) -> np.ndarray:
    return np.exp(-lam * np.power(M, rho))


def entropy_series(N: int,
                   alpha: float,
                   beta: float,
                   lam: float,
                   rho: float,
                   kB: float = 1.0) -> dict:
    n = np.arange(2, N + 2, dtype=float)
    b = baseline_term(n, alpha, beta)
    a = np.empty_like(b)
    M = np.empty_like(b)
    accum = 0.0
    for i in range(N):
        fm = f_memory(np.array([accum]), lam, rho)[0]
        ai = b[i] * fm
        a[i] = ai
        accum += ai
        M[i] = accum
    S = kB * np.cumsum(a)
    return {
        "n": n,
        "delta_S": a * kB,
        "S": S,
        "M": M,
    }


if __name__ == "__main__":
    for alpha, beta in [(0.9, 0.0), (1.0, 1.0), (1.1, 0.0)]:
        out = entropy_series(600000, alpha, beta, lam=1e-4, rho=1.0)
        print({"alpha": alpha, "beta": beta, "S_end": float(out["S"][-1])})


