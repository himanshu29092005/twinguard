"""Simulated 'real' water-tank plant + the first-principles physics model.

The real plant has effects the physics model does not know about:
  * heater efficiency < 1
  * slightly nonlinear heat loss
so a physics-only model is biased, and a hybrid twin (physics + ML residual)
is needed to get a clean residual signal for fault detection.
"""
from dataclasses import dataclass
import numpy as np

DT = 10.0          # seconds per step
T_AMB = 20.0       # ambient temperature (deg C)
P_MAX = 6.0        # heater power limit (kW)
C_TANK = 200.0     # thermal capacity (kJ/degC)
K_LOSS = 0.05      # heat-loss coefficient (kW/degC)
SENSOR_STD = 0.02  # measurement noise (deg C)


def physics_step(T, u):
    """Physics-only model: ideal heater, linear loss. Returns next temperature."""
    return T + DT * (u * P_MAX - K_LOSS * (T - T_AMB)) / C_TANK


@dataclass
class Plant:
    """The 'real' tank. `eta` is the heater efficiency (a fault can lower it)."""
    eta: float = 0.90
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.T = T_AMB

    def reset(self, T0=T_AMB):
        self.T = T0
        return self.T

    def step(self, u):
        u = float(np.clip(u, 0.0, 1.0))
        d = self.T - T_AMB
        loss = K_LOSS * d * (1.0 + 0.004 * d)       # nonlinear loss the physics model misses
        self.T += DT * (self.eta * u * P_MAX - loss) / C_TANK
        return self.T

    def measure(self):
        return self.T + self.rng.normal(0.0, SENSOR_STD)


class PID:
    def __init__(self, kp=0.25, ki=0.004, kd=0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i = 0.0
        self.prev = None

    def __call__(self, setpoint, y):
        e = setpoint - y
        self.i = float(np.clip(self.i + e * DT, -150.0, 150.0))
        d = 0.0 if self.prev is None else (e - self.prev) / DT
        self.prev = e
        return float(np.clip(self.kp * e + self.ki * self.i + self.kd * d, 0.0, 1.0))
