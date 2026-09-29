"""Closed-loop simulation of the tank under PID control, with optional fault."""
import numpy as np
from plant import Plant, PID, SENSOR_STD
from faults import Fault

N_STEPS = 600
# warm-up to 50 degC, then setpoint changes every 10 min (also excites the heater command)
SETPOINT_PLAN = [(0, 50.0), (150, 55.0), (210, 50.0), (270, 56.0), (330, 51.0),
                 (390, 56.0), (450, 50.0), (510, 55.0)]


def setpoint_at(t, plan=SETPOINT_PLAN):
    sp = plan[0][1]
    for start, val in plan:
        if t >= start:
            sp = val
    return sp


def run(fault=None, seed=0, n=N_STEPS, prbs=False):
    """Returns dict of arrays: y (measured), u (heater cmd), T (true temp), sp, fault_t0.
    prbs=True adds random power kicks (used only to collect twin training data)."""
    fault = fault or Fault()
    rng = np.random.default_rng(seed + 10_000)
    plant = Plant(seed=seed)
    base_eta = plant.eta
    pid = PID()
    plant.reset()
    y_hist, u_hist, T_hist, sp_hist = [], [], [], []
    state = {}
    plan = SETPOINT_PLAN
    if prbs:
        plan = [(0, 40.0)]
        for s in range(60, n, 60):
            plan.append((s, float(rng.uniform(30, 62))))
    kick = 0.0
    for t in range(n):
        fault.apply_plant(plant, t, base_eta)
        y = fault.apply_sensor(plant.measure(), t, state)
        sp = setpoint_at(t, plan)
        u = pid(sp, y)
        if prbs:
            if t % 7 == 0:
                kick = rng.uniform(-0.4, 0.4)
            u = float(np.clip(u + kick, 0.0, 1.0))
        plant.step(u)
        y_hist.append(y); u_hist.append(u); T_hist.append(plant.T); sp_hist.append(sp)
    return dict(y=np.array(y_hist), u=np.array(u_hist), T=np.array(T_hist),
                sp=np.array(sp_hist), t0=fault.t0, kind=fault.kind)
