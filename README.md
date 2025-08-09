https://pedrobaseivoa.github.io/the-breath-between-two-exhales/

# Analysis Scripts and Reproducibility

Setup:
- Python 3.x
- Install dependencies:
  - `python -m pip install -r requirements.txt`

Main scripts (run from project root):
- Classification/regimes:
  - `python scripts/series_validation.py`
  - `python scripts/critical_sweep.py` → saves `figures/critical_sweep.csv`
  - `python scripts/plot_critical_sweep.py` → heatmaps in `figures/critical_heatmap_*.png`
- Complex breathing (log‑periodic diagnostics):
  - `python scripts/complex_breathing.py`
  - `python scripts/phase_analysis.py` → `phase_vs_lnn.png`, `phase_slope.png`, `complex_series_spectrum.png`, and `phase_fit.txt`
  - `python scripts/complex_grid.py` → grid CSV/heatmaps of residence/amplitude
- ODE vs. Discrete sanity check:
  - `python scripts/ode_check.py`
- Domain illustrations:
  - `python scripts/thermo_entropy.py`
  - `python scripts/cosmology_memory.py`
  - `python scripts/neural_thresholds.py`
- Figures bundle:
  - `python scripts/plots.py` → creates core figures in `figures/`

Outputs overview (in `figures/`):
- `entropy_S.png`, `cosmology_H.png`, `awareness_accumulation.png`
- `breathing_magnitude.png`, `breathing_spectrum.png`
- `phase_vs_lnn.png`, `phase_slope.png`, `complex_series_spectrum.png`, `phase_fit.txt`
- `critical_sweep.csv`, `critical_heatmap_windowed_growth.png`, `critical_heatmap_tail_mean.png`
- `complex_grid.csv`, `complex_grid_residence.png`, `complex_grid_amplitude.png`

Notes:
- Current simulations use index‑based time for clarity. Domain‑specific time mappings N↔t are outlined in Appendix A.0 of the paper.
- Randomness is not used in the current deterministic runs; for long runs, consider pinning versions in `requirements.txt`.

# Next Cycle Plan

1) Physical anchoring (N↔t calibration per domínio)
- Cosmology: test N(t)≈(t/t0)^γ; check trends with SNe/BAO series (synthetic/public)
- Thermodynamics: choose Δt per experiment; validate S(t) shapes
- Neural: map Δt_syn vs. stimulation rates in LTP protocols

2) Log‑periodicity in real data
- Extract phase/PSD from empirical series and compare with phase diagnostics (slope ≈ α_i + β_i/ln n)

3) Sweeps and metrics
- Estimate N_c(Φ_c) vs. distance to (α,β)=(1,1)
- Wider (α_i,β_i) scans and residence/phase‑coherence maps

4) Memory: classes of f(M)
- Test families beyond exponential (e.g., (1+λM^ρ)^(-κ)) and assess robustness of convergence class and time‑to‑threshold

5) Reproducibility engineering
- Save run configs/metadata (JSON) per execution
- CLI flags for sizes (N), windows, tolerances; outputs with version stamps

