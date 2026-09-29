"""TwinGuard dashboard: inject a fault, watch the residual, see the alarm and the diagnosis.
Run:  streamlit run app.py
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from plant import DT
from sim import run
from faults import Fault
from twin import HybridTwin
from detect import calibrate, detect
from diagnose import diagnose

K, H = 0.25, 10.0

st.set_page_config(page_title="TwinGuard", layout="wide")
st.title("TwinGuard: fault detection and diagnosis on a hybrid digital twin")
st.caption("Physics + gradient-boosting twin of a heated water tank under PID control. "
           "Faults are injected into the simulated plant; the twin's residual reveals them.")


@st.cache_resource(show_spinner="Fitting hybrid twin...")
def get_twin():
    twin = HybridTwin().fit()
    res = [twin.residuals(run(seed=s)["y"], run(seed=s)["u"]) for s in range(200, 220)]
    mu, sigma = calibrate(res)
    return twin, mu, sigma


twin, mu, sigma = get_twin()

with st.sidebar:
    st.header("Fault injector")
    kind = st.selectbox("Fault type", ["none", "drift", "stuck", "degradation"], index=1)
    t0 = st.slider("Fault starts at step (10 s each)", 150, 400, 250, 10)
    drift_rate = st.slider("Sensor drift (deg C per step)", -0.03, 0.03, 0.02, 0.005)
    eta_drop = st.slider("Heater efficiency loss", 0.05, 0.40, 0.25, 0.05)
    seed = st.number_input("Random seed", 0, 9999, 11)

fault = Fault(kind=kind, t0=t0, drift_rate=drift_rate, eta_drop=eta_drop)
d = run(fault=fault, seed=int(seed))
res = twin.residuals(d["y"], d["u"])
alarm, source = detect(d["y"], d["u"], twin, mu, sigma, K, H)

c1, c2, c3 = st.columns(3)
if alarm is None:
    c1.metric("Alarm", "none")
    c2.metric("Detection delay", "-")
    c3.metric("Diagnosis", "-")
else:
    diag = diagnose(d["y"], d["u"], res, alarm)
    delay = (alarm - t0) * DT / 60 if kind != "none" else None
    c1.metric("Alarm raised at", f"{alarm * DT / 60:.1f} min", help=f"detector: {source}")
    c2.metric("Detection delay", "false alarm" if delay is None else f"{delay:.1f} min")
    c3.metric("Diagnosis", diag, delta="correct" if diag == kind else ("wrong" if kind != "none" else None))

t = np.arange(len(d["y"]))
fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
ax[0].plot(t, d["sp"], "k--", lw=1, label="setpoint")
ax[0].plot(t, d["y"], lw=1, label="sensor reading")
ax[0].plot(t, d["T"], lw=1, alpha=.7, label="true temperature")
ax[0].set_ylabel("temp (deg C)"); ax[0].legend(fontsize=8)
ax[1].plot(t[1:], (res - mu) / sigma, lw=.8)
ax[1].set_ylabel("residual (sigma)")
ax[2].plot(t, d["u"], lw=1); ax[2].set_ylabel("heater command"); ax[2].set_xlabel("step (10 s)")
for a in ax:
    if kind != "none":
        a.axvline(t0, color="r", ls=":", lw=1.2)
    if alarm is not None:
        a.axvline(alarm, color="g", lw=1.2)
fig.tight_layout()
st.pyplot(fig)
st.caption("Red dotted: fault start. Green: alarm. "
           "Residual = measured temperature minus the twin's one-step prediction, in healthy-noise sigmas.")

with st.expander("How it works"):
    st.markdown(
        "- **Twin**: physics energy balance + gradient-boosting model of what the physics misses.\n"
        "- **Detection**: two-sided CUSUM on the residual, plus a flatline check for frozen sensors.\n"
        "- **Diagnosis**: frozen readings mean *stuck*; residual proportional to heater power means "
        "*degradation*; residual offset independent of heater power means *drift*.")
