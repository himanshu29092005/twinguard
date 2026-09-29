"""Monte-Carlo evaluation of TwinGuard. Usage: python evaluate.py [--runs 40]

For each fault type, inject the fault at a random time with random severity, run the
closed-loop plant, detect, diagnose. Also measures false alarms on healthy runs and
compares the hybrid twin against a physics-only twin.
Writes results/results.csv, results/summary.md and results/*.png
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plant import DT
from sim import run
from faults import Fault
from twin import HybridTwin
from detect import calibrate, detect
from diagnose import diagnose

K, H = 0.25, 10.0          # CUSUM parameters (chosen so healthy runs never alarm)
FAULTS = ["drift", "stuck", "degradation"]


def random_fault(kind, rng):
    return Fault(kind=kind, t0=int(rng.integers(150, 350)),
                 drift_rate=float(rng.choice([-1, 1]) * rng.uniform(0.005, 0.03)),
                 eta_drop=float(rng.uniform(0.08, 0.35)))


def calibrate_twin(twin):
    res = []
    for s in range(200, 220):
        d = run(seed=s)
        res.append(twin.residuals(d["y"], d["u"]))
    return calibrate(res)


def evaluate_twin(twin, n_runs, seed=0):
    mu, sigma = calibrate_twin(twin)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_runs):                          # healthy runs -> false alarms
        d = run(seed=5000 + i)
        a, _ = detect(d["y"], d["u"], twin, mu, sigma, K, H)
        rows.append(dict(kind="none", detected=a is not None, delay_min=np.nan, diagnosis=None, correct=None))
    for kind in FAULTS:
        for i in range(n_runs):
            f = random_fault(kind, rng)
            d = run(fault=f, seed=7000 + 1000 * FAULTS.index(kind) + i)   # fixed seeds -> reproducible
            a, src = detect(d["y"], d["u"], twin, mu, sigma, K, H)
            if a is not None and a < f.t0:            # alarm before fault = false alarm, not a detection
                rows.append(dict(kind=kind, detected=False, delay_min=np.nan, diagnosis=None, correct=False, early=True))
                continue
            if a is None:
                rows.append(dict(kind=kind, detected=False, delay_min=np.nan, diagnosis=None, correct=False))
                continue
            dg = diagnose(d["y"], d["u"], twin.residuals(d["y"], d["u"]), a)
            rows.append(dict(kind=kind, detected=True, delay_min=(a - f.t0) * DT / 60, source=src,
                             diagnosis=dg, correct=(dg == kind)))
    return pd.DataFrame(rows), (mu, sigma)


def summarize(df):
    out = {}
    healthy = df[df.kind == "none"]
    out["false_alarm_rate"] = float(healthy.detected.mean())
    for kind in FAULTS:
        s = df[df.kind == kind]
        out[kind] = dict(recall=float(s.detected.mean()),
                         mean_delay_min=float(s.delay_min.mean()),
                         median_delay_min=float(s.delay_min.median()),
                         diag_acc=float(s.correct.astype(float).mean()))
    faulty = df[df.kind != "none"]
    out["overall_recall"] = float(faulty.detected.mean())
    out["overall_delay_min"] = float(faulty.delay_min.mean())
    out["overall_diag_acc"] = float(faulty.correct.astype(float).mean())
    return out


def plot_examples(twin, mu, sigma, path):
    fig, ax = plt.subplots(3, 3, figsize=(13, 8), sharex=True)
    for j, kind in enumerate(FAULTS):
        f = Fault(kind=kind, t0=250, drift_rate=0.02, eta_drop=0.25)
        d = run(fault=f, seed=11)
        a, src = detect(d["y"], d["u"], twin, mu, sigma, K, H)
        res = twin.residuals(d["y"], d["u"])
        t = np.arange(len(d["y"]))
        ax[0, j].plot(t, d["sp"], "k--", lw=1, label="setpoint")
        ax[0, j].plot(t, d["y"], lw=1, label="sensor reading")
        ax[0, j].plot(t, d["T"], lw=1, alpha=.7, label="true temp")
        ax[0, j].set_title(f"fault: {kind}"); ax[0, j].legend(fontsize=7)
        ax[1, j].plot(t[1:], (res - mu) / sigma, lw=.8); ax[1, j].set_ylabel("residual (sigma)")
        ax[2, j].plot(t, d["u"], lw=1); ax[2, j].set_ylabel("heater cmd"); ax[2, j].set_xlabel("step (10 s)")
        for r in range(3):
            ax[r, j].axvline(f.t0, color="r", ls=":", lw=1.2)
            if a is not None:
                ax[r, j].axvline(a, color="g", ls="-", lw=1.2)
    ax[0, 0].set_ylabel("temp (deg C)")
    fig.suptitle("TwinGuard examples: red dotted = fault start, green = alarm", y=0.995)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_confusion(df, path):
    labels = FAULTS
    m = np.zeros((3, 3))
    for i, k in enumerate(labels):
        s = df[(df.kind == k) & df.detected]
        for j, l in enumerate(labels):
            m[i, j] = (s.diagnosis == l).sum()
    m = m / np.maximum(m.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(m, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, rotation=20); ax.set_yticks(range(3)); ax.set_yticklabels(labels)
    ax.set_xlabel("diagnosed as"); ax.set_ylabel("true fault")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m[i, j]:.0%}", ha="center", va="center", color="w" if m[i, j] > .5 else "k")
    ax.set_title("Diagnosis confusion matrix"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40, help="runs per fault type (and healthy runs)")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    print("Fitting hybrid twin (physics + gradient boosting)...")
    hybrid = HybridTwin().fit()
    physics = HybridTwin(use_ml=False)

    df_h, (mu, sigma) = evaluate_twin(hybrid, args.runs)
    df_p, _ = evaluate_twin(physics, args.runs)
    df_h.to_csv("results/results.csv", index=False)
    sh, sp = summarize(df_h), summarize(df_p)

    lines = [f"# TwinGuard results ({args.runs} runs per fault type, {args.runs} healthy runs)\n",
             "## Hybrid twin (physics + ML residual) + CUSUM + flatline check\n",
             "| Fault | Recall | Mean delay (min) | Median delay (min) | Diagnosis accuracy |",
             "|-------|--------|------------------|--------------------|--------------------|"]
    for k in FAULTS:
        r = sh[k]
        lines.append(f"| {k} | {r['recall']:.0%} | {r['mean_delay_min']:.1f} | {r['median_delay_min']:.1f} | {r['diag_acc']:.0%} |")
    lines += [f"| **all faults** | **{sh['overall_recall']:.0%}** | **{sh['overall_delay_min']:.1f}** | | **{sh['overall_diag_acc']:.0%}** |",
              f"\nFalse alarm rate on healthy runs: **{sh['false_alarm_rate']:.0%}**\n",
              "## Ablation: same detector on a physics-only twin (no ML residual)\n",
              f"- False alarm rate on healthy runs: **{sp['false_alarm_rate']:.0%}**",
              f"- Recall on faulty runs (counting only alarms after fault start): {sp['overall_recall']:.0%}",
              "\nThe physics-only model is biased, so its residual is never zero-mean and the detector fires on healthy plants."]
    open("results/summary.md", "w").write("\n".join(lines))
    print("\n".join(lines))

    plot_examples(hybrid, mu, sigma, "results/examples.png")
    plot_confusion(df_h, "results/confusion.png")
    print("\nSaved results/ (results.csv, summary.md, examples.png, confusion.png)")


if __name__ == "__main__":
    main()
