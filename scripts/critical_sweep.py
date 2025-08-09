import os
import numpy as np
import pandas as pd
from series_validation import classify


def sweep_alpha_beta(N: int,
                     alpha_center: float = 1.0,
                     beta_center: float = 1.0,
                     delta: float = 0.02,
                     steps: int = 9,
                     lam: float = 1e-4,
                     rho: float = 1.0,
                     csv_path: str | None = "figures/critical_sweep.csv"):
    alphas = np.linspace(alpha_center - delta, alpha_center + delta, steps)
    betas = np.linspace(beta_center - delta, beta_center + delta, steps)
    results = []
    for a in alphas:
        for b in betas:
            res = classify(N, a, b, lam, rho)
            results.append(res)
            print(res)
    if csv_path:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
    return results


if __name__ == "__main__":
    sweep_alpha_beta(N=800000, delta=0.02, steps=9)


