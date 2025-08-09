import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from complex_breathing import complex_series, breathing_metrics


def ensure_out(dirpath: str = "figures") -> str:
    os.makedirs(dirpath, exist_ok=True)
    return dirpath


def scan_grid(alpha_r: float = 1.0,
              beta_r: float = 1.0,
              alpha_i_min: float = 2.4,
              alpha_i_max: float = 3.2,
              beta_i_min: float = 2.4,
              beta_i_max: float = 3.2,
              steps: int = 7,
              N: int = 60000,
              lam: float = 1e-4,
              rho: float = 1.0) -> pd.DataFrame:
    a_is = np.linspace(alpha_i_min, alpha_i_max, steps)
    b_is = np.linspace(beta_i_min, beta_i_max, steps)
    rows = []
    for ai in a_is:
        for bi in b_is:
            n, a, phase, M = complex_series(N, alpha_r, beta_r, ai, bi, lam, rho)
            mets = breathing_metrics(a, n, scale_frac=0.5)
            rows.append({
                "alpha_i": ai,
                "beta_i": bi,
                "residence": mets["residence"],
                "amplitude_std": mets["amplitude_std"],
                "mean_mag": mets["mean_mag"],
            })
    return pd.DataFrame(rows)


def heatmap(df: pd.DataFrame, value: str, outpng: str):
    a_vals = np.sort(df["alpha_i"].unique())
    b_vals = np.sort(df["beta_i"].unique())
    grid = np.zeros((len(a_vals), len(b_vals)))
    for i, ai in enumerate(a_vals):
        for j, bi in enumerate(b_vals):
            grid[i, j] = df[(df.alpha_i == ai) & (df.beta_i == bi)][value].values[0]
    plt.figure(figsize=(6, 5))
    im = plt.imshow(grid, origin='lower', aspect='auto',
                    extent=[b_vals.min(), b_vals.max(), a_vals.min(), a_vals.max()])
    plt.colorbar(im, label=value)
    plt.xlabel("beta_i")
    plt.ylabel("alpha_i")
    plt.title(f"Heatmap of {value}")
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()


if __name__ == "__main__":
    figdir = ensure_out()
    df = scan_grid()
    csv_path = os.path.join(figdir, "complex_grid.csv")
    df.to_csv(csv_path, index=False)
    heatmap(df, "residence", os.path.join(figdir, "complex_grid_residence.png"))
    heatmap(df, "amplitude_std", os.path.join(figdir, "complex_grid_amplitude.png"))
    print({"saved": [csv_path,
                     "figures/complex_grid_residence.png",
                     "figures/complex_grid_amplitude.png"]})


