"""Fault injection. Faults act either on the sensor reading or on the plant itself."""
from dataclasses import dataclass
import numpy as np

FAULT_TYPES = ["none", "drift", "stuck", "degradation"]


@dataclass
class Fault:
    kind: str = "none"        # one of FAULT_TYPES
    t0: int = 10**9           # step at which the fault starts
    drift_rate: float = 0.02  # deg C per step (sensor drift)
    eta_drop: float = 0.25    # fractional efficiency loss (heater degradation)

    def apply_plant(self, plant, t, base_eta):
        """Heater degradation changes the real plant."""
        if self.kind == "degradation" and t >= self.t0:
            plant.eta = base_eta * (1.0 - self.eta_drop)

    def apply_sensor(self, y, t, state):
        """Drift and stuck-sensor corrupt only the reading."""
        if self.kind == "drift" and t >= self.t0:
            return y + self.drift_rate * (t - self.t0)
        if self.kind == "stuck" and t >= self.t0:
            if "stuck_val" not in state:
                state["stuck_val"] = y
            return state["stuck_val"]
        return y
