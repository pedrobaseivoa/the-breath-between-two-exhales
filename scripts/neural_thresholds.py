import numpy as np


def ltp_response(steps: int,
                 alpha: float,
                 beta: float,
                 lam: float,
                 rho: float,
                 stim: float = 1.0) -> dict:
    # Simplified mapping: potentiation increments follow threshold series with memory
    n = np.arange(2, steps + 2, dtype=float)
    ln_n = np.log(n)
    ln_n = np.where(ln_n == 0.0, np.finfo(float).tiny, ln_n)
    b = stim / (np.power(n, alpha) * np.power(ln_n, beta))
    a = np.empty_like(b)
    M = np.empty_like(b)
    accum = 0.0
    for i in range(steps):
        fm = np.exp(-lam * (accum ** rho))
        ai = b[i] * fm
        a[i] = ai
        accum += ai
        M[i] = accum
    return {"n": n, "delta_W": a, "W": M}


if __name__ == "__main__":
    out = ltp_response(200000, alpha=1.0, beta=1.0, lam=1e-3, rho=1.0, stim=1.0)
    print({"W_end": float(out["W"][-1]), "delta_W_last": float(out["delta_W"][-1])})


