#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effect of changing body weight distribution to 70–140 kg on optimal QD/QW doses
-------------------------------------------------------------------------------
- Loads trained LSTM PD(t) model (no retraining)
- Rebuilds/loads time & PD scalers from Phase-1 data
- Monte Carlo with BW ~ Uniform(70, 140) kg, COMED ~ Bernoulli(p_phase1)
- QD grid: auto-extends (0.5 mg steps) until it crosses 90% success
- QW grid: 5 mg steps (fixed)
- Success = max_t PD(t) ≤ 3.3 ng/mL over full interval at steady-state
- Outputs: minimal dose achieving ≥90% success (mean & 95% CI) for QD & QW
"""

import os
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 140

from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras import models

# ----------------------------- Defaults -----------------------------
STRICT_THRESHOLD  = 3.3
TAU_QD            = 24.0
TAU_QW            = 168.0

QD_MAX_DOSE       = 40.0
QD_STEP           = 0.5
QW_MAX_DOSE       = 400.0
QW_STEP           = 5.0

K_EXPAND          = 10
R_MC              = 200
BW_RANGE          = (70.0, 140.0)
SEED              = 42

# ----------------------------- Utils -----------------------------
def maybe_enable_gpu_memory_growth():
    try:
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

def save_scaler(scaler: StandardScaler, path: str):
    import joblib
    joblib.dump(scaler, path)

def load_scaler(path: str) -> StandardScaler:
    import joblib
    return joblib.load(path)

def load_trained_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing '{path}'. Train your LSTM first (produces an .h5 or SavedModel dir)."
        )
    try:
        return models.load_model(
            path,
            custom_objects={"mse": tf.keras.losses.MeanSquaredError()},
            compile=False,
        )
    except TypeError:
        return models.load_model(path, compile=False)

def load_or_build_scalers(time_scaler_path: str, y_scaler_path: str, data_csv: str):
    have_both = os.path.exists(time_scaler_path) and os.path.exists(y_scaler_path)
    if have_both:
        print("Found saved scalers. Loading …")
        return load_scaler(time_scaler_path), load_scaler(y_scaler_path)

    print("Preparing scalers from Phase-1 data …")
    df = pd.read_csv(data_csv)
    for c in ["ID","BW","COMED","DOSE","TIME","DV","EVID","MDV","DVID","AMT"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ID","TIME","EVID","MDV","DVID"])
    df = df[df["TIME"] >= 0].sort_values(["ID","TIME"])

    obs = df[(df["EVID"]==0) & (df["MDV"]==0)]
    pd_df = obs[obs["DVID"]==2].copy()

    def infer_tau(s):
        d = s[s["EVID"]==1]
        if len(d):
            t = np.sort(d["TIME"].dropna().unique())
            return float(np.median(np.diff(t))) if len(t)>=2 else TAU_QD
        return TAU_QD

    def last_window(s, tau, min_points=3, max_expand=8.0):
        if s.empty: return s.iloc[0:0].copy()
        tmax = float(s["TIME"].max()); best, bestn = s.iloc[0:0].copy(), 0; f=1.0
        while f <= max_expand:
            w = s[(s["TIME"]>=tmax - tau*f) & (s["TIME"]<=tmax)]
            if len(w)>bestn: best, bestn = w.copy(), len(w)
            if len(w) >= min_points: return w.sort_values("TIME")
            f *= 1.5
        return best.sort_values("TIME")

    def resample_to_grid(time, value, grid):
        if len(time)==0: return np.full_like(grid, np.nan, float)
        ix = np.argsort(time)
        t, v = np.asarray(time)[ix], np.asarray(value)[ix]
        out = np.interp(grid, t - t.min(), v, left=np.nan, right=np.nan)
        if np.isnan(out[0]):
            first = np.nanmin(np.where(~np.isnan(out))[0])
            out[:first] = out[first]
        m = np.isnan(out)
        if m.any():
            out[m] = np.interp(np.where(m)[0], np.where(~m)[0], out[~m])
        return out

    # Build 24-point PD sequences to fit scalers
    T = 24
    grid_24h = np.linspace(0, TAU_QD, T)
    y_all, t_all = [], []
    for sid, s in pd_df.groupby("ID"):
        tau = infer_tau(df[df["ID"]==sid])
        w = last_window(s.sort_values("TIME"), tau, 3, 8.0)
        if w.empty: continue
        vg = resample_to_grid(w["TIME"].values.astype(float),
                              w["DV"].values.astype(float), grid_24h)
        y_all.append(vg)
        t_all.append(grid_24h / TAU_QD)

    if not y_all:
        raise RuntimeError("Could not build PD sequences to fit scalers from EstData.csv.")

    y_all = np.asarray(y_all, float)  # (N,24)
    t_all = np.asarray(t_all, float)  # (N,24)

    y_scaler = StandardScaler().fit(y_all.reshape(-1,1))
    time_scaler = StandardScaler().fit(t_all.reshape(-1,1))

    save_scaler(time_scaler, time_scaler_path)
    save_scaler(y_scaler, y_scaler_path)
    print("✓ Scalers fitted & saved.")
    return time_scaler, y_scaler

def phase1_comed_prob(data_csv: str) -> float:
    df = pd.read_csv(data_csv)
    for c in ["ID","BW","COMED","DOSE","TIME","DV","EVID","MDV","DVID","AMT"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    ids = df["ID"].dropna().unique()
    COM_obs = []
    for sid in ids:
        s = df[df["ID"]==sid]
        if s["COMED"].notna().any():
            COM_obs.append(int(s["COMED"].dropna().iloc[0]))
    COM_obs = np.array([c for c in COM_obs if c in [0,1]])
    return 0.5 if COM_obs.size == 0 else float(np.mean(COM_obs))

def make_batch(time_scaler, TAU, Tpoints, BW, COMED, dose_mg, tau_norm):
    N = len(BW)
    tgrid = np.linspace(0, TAU, Tpoints)
    dose_mgkg = dose_mg / np.clip(BW, 1e-6, None)
    Xb = np.stack([
        (tgrid/TAU) * np.ones((N, Tpoints)),
        np.tile(dose_mgkg[:, None], (1, Tpoints)),
        np.tile(COMED[:, None], (1, Tpoints)),
        np.full((N, Tpoints), tau_norm, float)
    ], axis=-1).astype(float)
    Xb[..., 0] = time_scaler.transform(Xb[..., 0].reshape(-1,1)).reshape(N, Tpoints)
    return Xb, tgrid

def predict_qd_pd_ngml(model, y_scaler, time_scaler, BW, COMED, dose_mg, Tpoints=24):
    Xb, tgrid = make_batch(time_scaler, TAU_QD, Tpoints, BW, COMED, dose_mg, tau_norm=1.0)
    yhat_z = model.predict(Xb, verbose=0).squeeze(-1)
    yhat   = y_scaler.inverse_transform(yhat_z.reshape(-1,1)).reshape(yhat_z.shape)
    return yhat, tgrid

def predict_qw_pd_ngml(model, y_scaler, time_scaler, BW, COMED, dose_weekly_mg, Tpoints=168):
    Xb, tgrid = make_batch(time_scaler, TAU_QW, Tpoints, BW, COMED, dose_weekly_mg, tau_norm=TAU_QW/TAU_QD)
    yhat_z = model.predict(Xb, verbose=0).squeeze(-1)
    yhat   = y_scaler.inverse_transform(yhat_z.reshape(-1,1)).reshape(yhat_z.shape)
    return yhat, tgrid

def mc_success_curve_qd(rng, model, y_scaler, time_scaler, dose_grid, N_base, p_comed, R=R_MC):
    succ_mat = np.zeros((R, len(dose_grid)))
    dose90   = np.full(R, np.nan)
    for r in tqdm(range(R), desc="QD MC"):
        BW = rng.uniform(BW_RANGE[0], BW_RANGE[1], size=N_base*K_EXPAND)
        COM = rng.binomial(1, p_comed, size=N_base*K_EXPAND)
        s = []
        for d in dose_grid:
            yhat, _ = predict_qd_pd_ngml(model, y_scaler, time_scaler, BW, COM, d, Tpoints=24)
            ok = (np.nanmax(yhat, axis=1) <= STRICT_THRESHOLD)
            s.append(ok.mean())
        s = np.array(s)
        succ_mat[r,:] = s
        i90 = np.where(s >= 0.90)[0]
        if i90.size:
            dose90[r] = float(dose_grid[i90[0]])
    return succ_mat, dose90

def mc_success_curve_qw(rng, model, y_scaler, time_scaler, dose_grid, N_base, p_comed, R=R_MC):
    succ_mat = np.zeros((R, len(dose_grid)))
    dose90   = np.full(R, np.nan)
    for r in tqdm(range(R), desc="QW MC"):
        BW = rng.uniform(BW_RANGE[0], BW_RANGE[1], size=N_base*K_EXPAND)
        COM = rng.binomial(1, p_comed, size=N_base*K_EXPAND)
        s = []
        for d in dose_grid:
            yhat, _ = predict_qw_pd_ngml(model, y_scaler, time_scaler, BW, COM, d, Tpoints=168)
            ok = (np.nanmax(yhat, axis=1) <= STRICT_THRESHOLD)
            s.append(ok.mean())
        s = np.array(s)
        succ_mat[r,:] = s
        i90 = np.where(s >= 0.90)[0]
        if i90.size:
            dose90[r] = float(dose_grid[i90[0]])
    return succ_mat, dose90

def summarize_curve(succ_mat, dose_grid, dose90_samples):
    succ_mean = succ_mat.mean(axis=0)
    succ_lo   = np.percentile(succ_mat, 2.5, axis=0)
    succ_hi   = np.percentile(succ_mat, 97.5, axis=0)
    if np.isfinite(dose90_samples).any():
        d90_mean = float(np.nanmean(dose90_samples))
        d90_ci   = (float(np.nanpercentile(dose90_samples, 2.5)),
                    float(np.nanpercentile(dose90_samples, 97.5)))
    else:
        d90_mean, d90_ci = np.nan, (np.nan, np.nan)
    idx = np.where(succ_mean >= 0.90)[0]
    d90_mean_curve = float(dose_grid[idx[0]]) if idx.size else np.nan
    return succ_mean, succ_lo, succ_hi, d90_mean, d90_ci, d90_mean_curve

def run_qd_until_cross(rng, model, y_scaler, time_scaler, N_base,
                       start=0.5, step=0.5, initial_max=40.0,
                       extend_by=20.0, max_extends=10, hard_cap=240.0, p_comed=0.5):
    grid = np.arange(start, initial_max + step, step)
    succ_qd, d90_qd = mc_success_curve_qd(rng, model, y_scaler, time_scaler, grid, N_base, p_comed, R=R_MC)
    succ_mean = succ_qd.mean(axis=0)
    if float(np.nanmax(succ_mean)) >= 0.90:
        return grid, succ_qd, d90_qd
    extends = 0
    while extends < max_extends and grid[-1] < hard_cap:
        last = grid[-1]
        new_top = min(last + extend_by, hard_cap)
        grid = np.arange(step, new_top + step, step)
        succ_qd, d90_qd = mc_success_curve_qd(rng, model, y_scaler, time_scaler, grid, N_base, p_comed, R=R_MC)
        succ_mean = succ_qd.mean(axis=0)
        if float(np.nanmax(succ_mean)) >= 0.90:
            return grid, succ_qd, d90_qd
        extends += 1
    return grid, succ_qd, d90_qd

def fmt_ci(ci):
    return f"[{ci[0]:.1f}, {ci[1]:.1f}] mg" if np.isfinite(ci[0]) and np.isfinite(ci[1]) else "N/A"

# ----------------------------- Main -----------------------------
def main():
    parser = argparse.ArgumentParser(description="BW shift (70–140 kg) QD/QW dose scan using a trained LSTM PD(t) model")
    parser.add_argument("--data", type=str, default="EstData.csv", help="Path to EstData.csv")
    parser.add_argument("--model", type=str, default="lstm_best.h5", help="Path to trained LSTM model (.h5 or SavedModel)")
    parser.add_argument("--out", type=str, default="results_bw70140", help="Output directory")
    parser.add_argument("--qd-step", type=float, default=QD_STEP, help="QD dose step (mg)")
    parser.add_argument("--qd-initial-max", type=float, default=QD_MAX_DOSE, help="QD initial grid max (mg)")
    parser.add_argument("--qw-step", type=float, default=QW_STEP, help="QW dose step (mg)")
    parser.add_argument("--qw-max", type=float, default=QW_MAX_DOSE, help="QW grid max (mg)")
    parser.add_argument("--mc", type=int, default=R_MC, help="MC replicates")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Paths for scalers
    time_scaler_path = outdir / "time_scaler_qd.pkl"
    y_scaler_path    = outdir / "y_scaler_pd.pkl"

    # Reproducibility
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)
    maybe_enable_gpu_memory_growth()

    print("Loading trained LSTM model …")
    model = load_trained_model(args.model)
    print("✓ Model loaded.")

    time_scaler, y_scaler = load_or_build_scalers(time_scaler_path.as_posix(),
                                                  y_scaler_path.as_posix(),
                                                  args.data)

    # Phase-1 COMED probability
    p_comed = phase1_comed_prob(args.data)

    # Base N from Phase-1 (subjects with any PD record)
    df_phase1 = pd.read_csv(args.data)
    for c in ["ID","BW","COMED","DOSE","TIME","DV","EVID","MDV","DVID","AMT"]:
        if c in df_phase1.columns: df_phase1[c] = pd.to_numeric(df_phase1[c], errors="coerce")
    obs = df_phase1[(df_phase1["EVID"]==0) & (df_phase1["MDV"]==0)]
    N_base = int(obs[obs["DVID"]==2]["ID"].nunique())
    if N_base < 1:
        N_base = 48

    print(f"Phase-1 PD subjects: {N_base}")
    print(f"Target BW distribution for simulation: Uniform[{BW_RANGE[0]}, {BW_RANGE[1]}] kg")
    print("Running QD and QW Monte-Carlo …")

    # QD: auto-extend until 90% crossing
    qd_grid, succ_qd, d90_qd = run_qd_until_cross(
        rng, model, y_scaler, time_scaler, N_base,
        start=args.qd_step, step=args.qd_step,
        initial_max=args.qd_initial_max, extend_by=20.0, max_extends=10,
        hard_cap=240.0, p_comed=p_comed
    )

    # QW: fixed grid
    qw_grid = np.arange(args.qw_step, args.qw_max + args.qw_step, args.qw_step)
    succ_qw, d90_qw = mc_success_curve_qw(rng, model, y_scaler, time_scaler, qw_grid, N_base, p_comed, R=args.mc)

    qd_mean, qd_lo, qd_hi, qd_d90, qd_d90_ci, qd_d90_curve = summarize_curve(succ_qd, qd_grid, d90_qd)
    qw_mean, qw_lo, qw_hi, qw_d90, qw_d90_ci, qw_d90_curve = summarize_curve(succ_qw, qw_grid, d90_qw)

    # Plots
    plt.figure(figsize=(8,5))
    plt.plot(qd_grid, qd_mean, label="QD mean success")
    plt.fill_between(qd_grid, qd_lo, qd_hi, alpha=0.2, label="QD 95% MC band")
    plt.axhline(0.90, color="tab:red", ls="--", lw=1.2, label="90% target")
    plt.ylim(0,1); plt.xlabel("Once-daily dose (mg)")
    plt.ylabel(f"Success fraction (PD(t) ≤ {STRICT_THRESHOLD} ng/mL over 24 h)")
    plt.grid(True, alpha=0.35); plt.legend(); plt.tight_layout()
    qd_png = outdir / "qd_70_140kg_dose_success.png"
    plt.savefig(qd_png.as_posix(), dpi=180); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(qw_grid, qw_mean, label="QW mean success")
    plt.fill_between(qw_grid, qw_lo, qw_hi, alpha=0.2, label="QW 95% MC band")
    plt.axhline(0.90, color="tab:red", ls="--", lw=1.2, label="90% target")
    plt.ylim(0,1); plt.xlabel("Once-weekly dose (mg)")
    plt.ylabel(f"Success fraction (PD(t) ≤ {STRICT_THRESHOLD} ng/mL over 168 h)")
    plt.grid(True, alpha=0.35); plt.legend(); plt.tight_layout()
    qw_png = outdir / "qw_70_140kg_dose_success.png"
    plt.savefig(qw_png.as_posix(), dpi=180); plt.close()

    # CSVs
    qd_csv = outdir / "qd_70_140kg_dose_success_summary.csv"
    pd.DataFrame({
        "qd_dose_mg": qd_grid,
        "succ_mean": qd_mean, "succ_lo2p5": qd_lo, "succ_hi97p5": qd_hi
    }).to_csv(qd_csv, index=False)

    qw_csv = outdir / "qw_70_140kg_dose_success_summary.csv"
    pd.DataFrame({
        "qw_dose_mg": qw_grid,
        "succ_mean": qw_mean, "succ_lo2p5": qw_lo, "succ_hi97p5": qw_hi
    }).to_csv(qw_csv, index=False)

    # Answers
    def _fmt_ci(ci):
        return f"[{ci[0]:.1f}, {ci[1]:.1f}] mg" if np.isfinite(ci[0]) and np.isfinite(ci[1]) else "N/A"

    print("\n=== Results for BW ~ Uniform(70, 140) kg ===")
    print(f"QD minimal dose for ≥90% success (MC mean): {qd_d90:.1f} mg  | 95% CI: {_fmt_ci(qd_d90_ci)}")
    print(f"QD minimal dose (on mean curve):            {qd_d90_curve:.1f} mg")
    print(f"QW minimal dose for ≥90% success (MC mean): {qw_d90:.0f} mg  | 95% CI: {_fmt_ci(qw_d90_ci)}")
    print(f"QW minimal dose (on mean curve):            {qw_d90_curve:.0f} mg")

    print("\nDebug:")
    print(f"  QD grid top = {qd_grid[-1]:.1f} mg; QD max success at grid top = {float(qd_mean[-1]):.3f}")

    print("\nSaved:")
    print(f"  {qd_png}")
    print(f"  {qw_png}")
    print(f"  {qd_csv}")
    print(f"  {qw_csv}")

if __name__ == "__main__":
    main()
