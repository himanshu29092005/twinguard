"""Hybrid digital twin = physics model + gradient-boosting residual model (one-step-ahead)."""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from plant import physics_step, T_AMB
from sim import run


def _features(y, u):
    return np.column_stack([y, u, y - T_AMB, u * (y - T_AMB)])


class HybridTwin:
    def __init__(self, use_ml=True):
        self.use_ml = use_ml
        self.model = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                               learning_rate=0.08, random_state=0)

    def fit(self, n_runs=12):
        X, r = [], []
        for s in range(n_runs):
            d = run(seed=1000 + s, prbs=True)
            y, u = d["y"], d["u"]
            X.append(_features(y[:-1], u[:-1]))
            r.append(y[1:] - physics_step(y[:-1], u[:-1]))
        self.model.fit(np.vstack(X), np.concatenate(r))
        return self

    def predict_next(self, y, u):
        """Predict y[t+1] from y[t], u[t] (arrays)."""
        y, u = np.asarray(y, float), np.asarray(u, float)
        base = physics_step(y, u)
        if not self.use_ml:
            return base
        return base + self.model.predict(_features(y, u))

    def residuals(self, y, u):
        """r[t] = y[t+1] - twin_prediction(y[t], u[t]); length len(y)-1."""
        return y[1:] - self.predict_next(y[:-1], u[:-1])
