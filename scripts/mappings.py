import numpy as np


def uniform_steps(t_max: float, dt: float) -> np.ndarray:
    """Generate uniform time grid."""
    n_steps = int(np.floor(t_max / dt))
    t = np.linspace(dt, n_steps * dt, n_steps)
    return t


def power_law_events(t: np.ndarray, t0: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """Map continuous time to discrete event index N(t) ~ (t/t0)^gamma."""
    ratio = np.maximum(t / max(t0, 1e-12), 1e-12)
    N = np.floor(np.power(ratio, gamma)).astype(int)
    N[N < 2] = 2  # ensure ln N well-defined
    return N


def identity_index(n_max: int) -> np.ndarray:
    return np.arange(2, n_max + 2, dtype=float)


