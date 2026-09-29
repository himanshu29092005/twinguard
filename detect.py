"""Fault detection: two-sided CUSUM on standardized twin residuals + flatline check."""
import numpy as np

FLAT_WINDOW = 8


def calibrate(residuals_list):
    """Mean/std of residuals from healthy runs (used to standardize)."""
    r = np.concatenate(residuals_list)
    return float(r.mean()), float(r.std())


def cusum(res, mu, sigma, k=0.5, h=8.0):
    """Two-sided CUSUM. Returns index (into `res`) of the first alarm or None."""
    z = (np.asarray(res) - mu) / sigma
    gp = gn = 0.0
    for i, v in enumerate(z):
        gp = max(0.0, gp + v - k)
        gn = max(0.0, gn - v - k)
        if gp > h or gn > h:
            return i
    return None


def threshold(res, mu, sigma, n_sigma=5.0):
    """Baseline detector: alarm when |z| exceeds n_sigma once."""
    z = np.abs((np.asarray(res) - mu) / sigma)
    idx = np.where(z > n_sigma)[0]
    return int(idx[0]) if len(idx) else None


def flatline(y, window=FLAT_WINDOW):
    """Alarm if the raw measurement is exactly constant for `window` samples
    (a healthy noisy sensor never does that). Returns index into y or None."""
    y = np.asarray(y)
    same = np.diff(y) == 0.0
    run_len = 0
    for i, s in enumerate(same):
        run_len = run_len + 1 if s else 0
        if run_len >= window - 1:
            return i + 1
    return None


def detect(y, u, twin, mu, sigma, k=0.5, h=8.0):
    """Combined detector. Returns (alarm_step, source) in original step indices, or (None, None).
    Residual r[i] belongs to step i+1."""
    res = twin.residuals(y, u)
    a = cusum(res, mu, sigma, k, h)
    a_step = a + 1 if a is not None else None
    f_step = flatline(y)
    cands = [(a_step, "cusum"), (f_step, "flatline")]
    cands = [c for c in cands if c[0] is not None]
    if not cands:
        return None, None
    return min(cands, key=lambda c: c[0])
