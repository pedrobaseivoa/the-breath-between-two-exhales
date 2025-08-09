import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_figdir(path: str = "figures") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def heatmap_from_csv(csv_path: str,
                     value_col: str,
                     out_png: str,
                     cmap: str = "viridis"):
    df = pd.read_csv(csv_path)
    # pivot to grid
    a_vals = np.sort(df["alpha"].astype(float).unique())
    b_vals = np.sort(df["beta"].astype(float).unique())
    grid = np.zeros((len(a_vals), len(b_vals)))
    for i, a in enumerate(a_vals):
        for j, b in enumerate(b_vals):
            sub = df[(df["alpha"].astype(float) == a) & (df["beta"].astype(float) == b)]
            if not sub.empty:
                grid[i, j] = float(sub.iloc[0][value_col])
            else:
                grid[i, j] = np.nan
    plt.figure(figsize=(6.5, 5))
    im = plt.imshow(grid, origin='lower', aspect='auto', cmap=cmap,
                    extent=[b_vals.min(), b_vals.max(), a_vals.min(), a_vals.max()])
    plt.colorbar(im, label=value_col)
    plt.xlabel("beta")
    plt.ylabel("alpha")
    plt.title(f"Critical Sweep: {value_col}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


if __name__ == "__main__":
    figdir = ensure_figdir()
    csv = os.path.join(figdir, "critical_sweep.csv")
    heatmap_from_csv(csv, "windowed_growth", os.path.join(figdir, "critical_heatmap_windowed_growth.png"))
    heatmap_from_csv(csv, "tail_mean", os.path.join(figdir, "critical_heatmap_tail_mean.png"))
    print({"saved": [
        "figures/critical_heatmap_windowed_growth.png",
        "figures/critical_heatmap_tail_mean.png",
    ]})


