# TwinGuard: Fault Detection and Diagnosis on a Hybrid Digital Twin

> A hybrid (physics + ML) digital twin of a heated water tank that detects and diagnoses sensor drift, stuck sensors and heater degradation in a closed-loop PID-controlled plant, with a Streamlit dashboard and a reproducible Monte-Carlo evaluation.

![Examples](results/examples.png)

## Why this project
A digital twin is only useful in operation if you can trust it to spot when the real asset misbehaves. TwinGuard compares the twin's one-step prediction with what the plant reports. The gap (the *residual*) is near zero when everything is healthy and shifts in fault-specific ways when it is not.

## How it works
1. **Hybrid twin** (`twin.py`): first-principles energy balance plus a gradient-boosting model that learns what the physics misses (heater efficiency, nonlinear heat loss). Trained on healthy runs only.
2. **Fault injector** (`faults.py`): three faults with random start time and severity.
   - *Sensor drift*: reading slowly walks away from the true temperature (either direction).
   - *Stuck sensor*: reading freezes at its last value.
   - *Heater degradation*: heater efficiency drops by 8-35%.
3. **Detection** (`detect.py`): two-sided CUSUM on the standardized residual, plus a flatline check for frozen readings.
4. **Diagnosis** (`diagnose.py`): rule-based, on the 15 minutes after the alarm.
   - frozen readings: **stuck**
   - residual proportional to heater command: **degradation**
   - residual offset independent of heater command: **drift**
5. **Evaluation** (`evaluate.py`): 50 random runs per fault type plus 50 healthy runs, fixed seeds.

## Results
Closed-loop PID plant, 10 s time step, sensor noise 0.02 deg C, fault start and severity randomized (`python evaluate.py --runs 50`).

| Fault | Recall | Mean delay (min) | Median delay (min) | Diagnosis accuracy |
|-------|--------|------------------|--------------------|--------------------|
| Sensor drift | 100% | 5.4 | 3.3 | 100% |
| Stuck sensor | 100% | 0.6 | 0.5 | 100% |
| Heater degradation | 100% | 5.6 | 3.0 | 100% |
| **All faults** | **100%** | **3.8** | | **100%** |

False alarm rate on 50 healthy runs: **0%**.

### Ablation: why the ML residual matters
Same detector, same settings, but the twin is physics-only:

| Twin | False alarm rate (healthy runs) | Recall (alarms after fault start) |
|------|---------------------------------|-----------------------------------|
| Hybrid (physics + ML) | **0%** | **100%** |
| Physics-only | 100% | 47% |

The physics model is biased against the real plant, so its residual is never zero-mean and a sensitive detector fires on a healthy system. Correcting the bias with the ML residual is what makes sensitive detection possible.

![Confusion matrix](results/confusion.png)

## Limitations (honest notes)
- Everything is simulated. The faults are idealized and the plant is the same simulator family that generated the training data, so these numbers are optimistic compared with real hardware.
- Drift and degradation are hard to tell apart when the heater command is constant. The diagnosis relies on the setpoint changes in the operating scenario to vary the heater power.
- The stuck-sensor case is caught mainly by the flatline rule, not by the twin. A real sensor with quantization would need a different rule.
- CUSUM parameters (`k=0.25`, `h=10`) were tuned on the same simulator; a real deployment would tune on plant data.

## Quick start
```bash
git clone https://github.com/himanshu29092005/twinguard.git
cd twinguard
pip install -r requirements.txt
python evaluate.py --runs 50   # reproduces the tables and figures in results/
streamlit run app.py           # interactive dashboard
```

## Project structure
| File | Purpose |
|------|---------|
| `plant.py` | Simulated tank plant, physics model, PID controller |
| `sim.py` | Closed-loop simulation with a setpoint plan |
| `faults.py` | Fault injector (drift, stuck, degradation) |
| `twin.py` | Hybrid twin: physics + gradient-boosting residual |
| `detect.py` | CUSUM, fixed-threshold baseline, flatline check |
| `diagnose.py` | Rule-based fault diagnosis |
| `evaluate.py` | Monte-Carlo evaluation, tables and plots |
| `app.py` | Streamlit dashboard |

## Roadmap
- Plug in a plant from Simulink or Amesim, or a real dataset
- Learn the diagnosis with a small classifier and compare with the rules
- Compare with a forecast-band detector using a time-series foundation model
- LLM-generated natural-language explanation of each alarm

## Tech stack
Python, NumPy, pandas, scikit-learn, Matplotlib, Streamlit
