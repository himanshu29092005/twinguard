"""Rule-based fault diagnosis from the window of data right after an alarm.

  stuck        -> raw sensor is frozen (identical readings for several samples)
  degradation  -> residual is explained by heater command u (r ~ -c*u, no offset)
  drift        -> residual is a persistent offset/trend independent of u
"""
import numpy as np
from detect import flatline, FLAT_WINDOW

WINDOW = 90  # steps after alarm used for diagnosis (15 min)


def diagnose(y, u, res, alarm_step, window=WINDOW):
    lo = max(1, alarm_step - 10)
    hi = min(len(y), alarm_step + window)
    yy = y[max(1, alarm_step - 10):hi]    # look for a frozen stretch around/after the alarm
    if flatline(yy, FLAT_WINDOW) is not None:
        return "stuck"
    r = res[lo - 1:hi - 1]            # residual r[i] belongs to step i+1
    uu = u[lo - 1:hi - 1]
    if len(r) < 8:
        return "drift"
    t = np.arange(len(r), dtype=float)
    # model A: r = a + b*t (offset/trend)   model B: r = c*u (proportional to heater power)
    XA = np.column_stack([np.ones_like(t), t])
    XB = uu.reshape(-1, 1)
    def sse(X):
        beta, *_ = np.linalg.lstsq(X, r, rcond=None)
        return float(np.sum((r - X @ beta) ** 2))
    return "degradation" if sse(XB) < sse(XA) else "drift"
