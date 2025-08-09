import numpy as np


def H_series(T: int,
             alpha: float,
             beta: float,
             lam: float,
             rho: float,
             H0: float = 1.0) -> np.ndarray:
    t = np.arange(2, T + 2, dtype=float)
    ln_t = np.log(t)
    ln_t = np.where(ln_t == 0.0, np.finfo(float).tiny, ln_t)
    b = 1.0 / (np.power(t, alpha) * np.power(ln_t, beta))
    H = np.empty_like(b)
    M = 0.0
    acc = 0.0
    for i in range(T):
        fm = np.exp(-lam * (M ** rho))
        term = b[i] * fm
        acc += term
        M += term
        H[i] = H0 + acc
    return H


if __name__ == "__main__":
    H = H_series(400000, alpha=1.0, beta=1.0, lam=1e-4, rho=1.0, H0=1.0)
    print({"H_start": float(H[0]), "H_end": float(H[-1])})


