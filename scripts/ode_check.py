import numpy as np
from typing import Callable


def baseline_b(n: float, alpha: float, beta: float) -> float:
    return 1.0 / (np.power(n, alpha) * np.power(np.log(max(n, 2.0)), beta))


def f_memory(M: float, lam: float, rho: float) -> float:
    return float(np.exp(-lam * (M ** rho)))


def integrate_ode(N: int,
                  alpha: float,
                  beta: float,
                  lam: float,
                  rho: float,
                  h: float = 2.0) -> float:
    # dM/dn = b(n) f(M)
    M = 0.0
    n = 2.0
    for _ in range(N):
        b = baseline_b(n, alpha, beta)
        k1 = b * f_memory(M, lam, rho)
        k2 = baseline_b(n + 0.5 * h, alpha, beta) * f_memory(M + 0.5 * h * k1, lam, rho)
        k3 = baseline_b(n + 0.5 * h, alpha, beta) * f_memory(M + 0.5 * h * k2, lam, rho)
        k4 = baseline_b(n + h, alpha, beta) * f_memory(M + h * k3, lam, rho)
        M += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        n += h
    return M


if __name__ == "__main__":
    for alpha, beta in [(0.9, 0.0), (1.0, 1.0), (1.1, 0.0)]:
        M_ode = integrate_ode(200000, alpha, beta, lam=1e-4, rho=1.0, h=2.0)
        print({"alpha": alpha, "beta": beta, "M_ode": M_ode})


