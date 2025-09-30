#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
No-concomitant medication scenario (COMED=0):
Optimal once-daily & once-weekly doses for ≥90% success (PD(t) ≤ 3.3 ng/mL)
-------------------------------------------------------------------------------
- Loads trained LSTM (no retrain)
- Loads/rebuilds time & PD scalers from Phase-1 data
- Monte Carlo over Phase-1 BW distribution (with resampling); COMED forced to 0
- QD grid auto-extends (0.5 mg steps) until it crosses 90% success
- Outputs dose→success curves + minimal 90% dose (mean & 95% CI) for QD and QW
"""
import os, warnings, argparse
import numpy as np, pandas as pd, matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 140

from pathlib import Path
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras import models

# ----------------------------- Defaults -----------------------------
MODEL_PATH_DEFAULT = "lstm_best.h5"
DATA_CSV_DEFAULT   = "EstData.csv"
OUTDIR_DEFAULT     = "results_no_comed"

STRICT_THRESHOLD  = 3.3
TAU_QD            = 24.0
TAU_QW            = 168.0

QD_MAX_DOSE       = 40.0
QD_STEP           = 0.5
QW_MAX_DOSE       = 400.0
QW_STEP           = 5.0

K_EXPAND          = 10
R_MC_DEFAULT      = 200
BW_JITTER_GCV     = 0.10
SEED_DEFAULT      = 42

TIME_SCALER_NAME  = "time_scaler_qd.pkl"
Y_SCALER_NAME     = "y_scaler_pd.pkl"

COLOR_LINE        = "#0000CD"  # MediumBlue
COLOR_FILL        = "#0000CD"  # same hue with alpha

# ----------------------------- Utils -----------------------------
def maybe_enable_gpu_memory_growth():
    try:
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

def save_scaler(scaler: StandardScaler, path: str):
    import joblib; joblib.dump(scaler, path)

def load_scaler(path: str) -> StandardScaler:
    import joblib; return joblib.load(path)

def load_trained_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing '{path}'. Train an LSTM first.")
    try:
        return models.load_model(path, custom_objects={"mse": tf.keras.losses.MeanSquaredError()}, compile=False)
    except TypeError:
        return models.load_model(path, compile=False)

def load_or_build_scalers(outdir: Path, data_csv: str):
    tsp = outdir / TIME_SCALER_NAME
    ysp = outdir / Y_SCALER_NAME
    if tsp.exists() and ysp.exists():
        print("Found saved scalers. Loading …")
        return load_scaler(tsp.as_posix()), load_scaler(ysp.as_posix())

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
            first = np.nanmin(np.where(~np.isnan(out))[0]); out[:first] = out[first]
        m = np.isnan(out)
        if m.any(): out[m] = np.interp(np.where(m)[0], np.where(~m)[0], out[~m])
        return out

    T = 24
    grid_24h = np.linspace(0, TAU_QD, T)
    y_all, t_all = [], []
    for sid, s in pd_df.groupby("ID"):
        tau = infer_tau(df[df["ID"]==sid])
        w = last_window(s.sort_values("TIME"), tau, 3, 8.0)
        if w.empty: continue
        vg = resample_to_grid(w["TIME"].values.astype(float), w["DV"].values.astype(float), grid_24h)
        y_all.append(vg); t_all.append(grid_24h / TAU_QD)

    if not y_all: raise RuntimeError("Could not build PD sequences to fit scalers from EstData.csv.")

    y_scaler = StandardScaler().fit(np.asarray(y_all).reshape(-1,1))
    time_scaler = StandardScaler().fit(np.asarray(t_all).reshape(-1,1))
    save_scaler(time_scaler, tsp.as_posix()); save_scaler(y_scaler, ysp.as_posix())
    print("✓ Scalers fitted & saved.")
    return time_scaler, y_scaler

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

# ------------------------- MC helpers -----------------------------
def draw_population_no_comed(rng, n_base: int, BW_per_id: np.ndarray):
    idx = rng.integers(0, BW_per_id.size, size=n_base*K_EXPAND)
    BW  = BW_per_id[idx] * np.exp(rng.normal(0.0, BW_JITTER_GCV, size=n_base*K_EXPAND))
    BW  = np.clip(BW, 30.0, 200.0)
    COM = np.zeros_like(BW, dtype=int)  # restriction: COMED=0
    return BW, COM

def mc_success_curve_qd(rng, model, y_scaler, time_scaler, dose_grid, N_base, BW_per_id, R):
    succ_mat = np.zeros((R, len(dose_grid))); dose90 = np.full(R, np.nan)
    for r in tqdm(range(R), desc="QD MC (COMED=0)"):
        BW, COM = draw_population_no_comed(rng, N_base, BW_per_id)
        s = []
        for d in dose_grid:
            yhat, _ = predict_qd_pd_ngml(model, y_scaler, time_scaler, BW, COM, d, Tpoints=24)
            ok = (np.nanmax(yhat, axis=1) <= STRICT_THRESHOLD)
            s.append(ok.mean())
        s = np.array(s); succ_mat[r,:] = s
        i90 = np.where(s >= 0.90)[0]
        if i90.size: dose90[r] = float(dose_grid[i90[0]])
    return succ_mat, dose90

def mc_success_curve_qw(rng, model, y_scaler, time_scaler, dose_grid, N_base, BW_per_id, R):
    succ_mat = np.zeros((R, len(dose_grid))); dose90 = np.full(R, np.nan)
    for r in tqdm(range(R), desc="QW MC (COMED=0)"):
        BW, COM = draw_population_no_comed(rng, N_base, BW_per_id)
        s = []
        for d in dose_grid:
            yhat, _ = predict_qw_pd_ngml(model, y_scaler, time_scaler, BW, COM, d, Tpoints=168)
            ok = (np.nanmax(yhat, axis=1) <= STRICT_THRESHOLD)
            s.append(ok.mean())
        s = np.array(s); succ_mat[r,:] = s
        i90 = np.where(s >= 0.90)[0]
        if i90.size: dose90[r] = float(dose_grid[i90[0]])
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

def run_qd_until_cross(rng, model, y_scaler, time_scaler, N_base, BW_per_id,
                       start=0.5, step=0.5, initial_max=40.0,
                       extend_by=20.0, max_extends=10, hard_cap=240.0,
                       R=R_MC_DEFAULT):
    grid = np.arange(start, initial_max + step, step)
    succ_qd, d90_qd = mc_success_curve_qd(rng, model, y_scaler, time_scaler, grid, N_base, BW_per_id, R)
    succ_mean = succ_qd.mean(axis=0)
    if float(np.nanmax(succ_mean)) >= 0.90:
        return grid, succ_qd, d90_qd
    extends = 0
    while extends < max_extends and grid[-1] < hard_cap:
        new_top = min(grid[-1] + extend_by, hard_cap)
        grid = np.arange(step, new_top + step, step)
        succ_qd, d90_qd = mc_success_curve_qd(rng, model, y_scaler, time_scaler, grid, N_base, BW_per_id, R)
        succ_mean = succ_qd.mean(axis=0)
        if float(np.nanmax(succ_mean)) >= 0.90:
            return grid, succ_qd, d90_qd
        extends += 1
    return grid, succ_qd, d90_qd

def fmt_ci(ci):
    return f"[{ci[0]:.1f}, {ci[1]:.1f}] mg" if np.isfinite(ci[0]) and np.isfinite(ci[1]) else "N/A"

# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="No-concomitant (COMED=0) dose scan using a trained LSTM PD(t) model")
    ap.add_argument("--data",  type=str, default=DATA_CSV_DEFAULT, help="Path to EstData.csv")
    ap.add_argument("--model", type=str, default=MODEL_PATH_DEFAULT, help="Path to trained LSTM model")
    ap.add_argument("--out",   type=str, default=OUTDIR_DEFAULT, help="Output directory")
    ap.add_argument("--mc",    type=int, default=R_MC_DEFAULT, help="MC replicates")
    ap.add_argument("--seed",  type=int, default=SEED_DEFAULT, help="Random seed")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed); rng = np.random.default_rng(args.seed); tf.random.set_seed(args.seed)
    maybe_enable_gpu_memory_growth()

    print("Loading trained LSTM model …")
    model = load_trained_model(args.model)
    print("✓ Model loaded.")

    time_scaler, y_scaler = load_or_build_scalers(outdir, args.data)

    df_phase1 = pd.read_csv(args.data)
    for c in ["ID","BW","COMED","DOSE","TIME","DV","EVID","MDV","DVID","AMT"]:
        if c in df_phase1.columns: df_phase1[c] = pd.to_numeric(df_phase1[c], errors="coerce")

    obs = df_phase1[(df_phase1["EVID"]==0) & (df_phase1["MDV"]==0)]
    N_base = int(obs[obs["DVID"]==2]["ID"].nunique())
    if N_base < 1: N_base = 48

    BW_per_id = []
    for sid, s in df_phase1.groupby("ID"):
        if s["BW"].notna().any():
            BW_per_id.append(float(s["BW"].dropna().iloc[0]))
    BW_per_id = np.array([b for b in BW_per_id if np.isfinite(b) and b>0]) or np.array([70.0])

    print(f"Phase-1 PD subjects: {N_base}")
    print("Restriction applied: COMED=0 for all simulated subjects")
    print("Running QD and QW Monte-Carlo …")

    qd_grid, succ_qd, d90_qd = run_qd_until_cross(
        rng, model, y_scaler, time_scaler, N_base, BW_per_id,
        start=QD_STEP, step=QD_STEP, initial_max=QD_MAX_DOSE,
        extend_by=20.0, max_extends=10, hard_cap=240.0, R=args.mc
    )

    qw_grid = np.arange(QW_STEP, QW_MAX_DOSE + QW_STEP, QW_STEP)
    succ_qw, d90_qw = mc_success_curve_qw(rng, model, y_scaler, time_scaler, qw_grid, N_base, BW_per_id, R=args.mc)

    qd_mean, qd_lo, qd_hi, qd_d90, qd_d90_ci, qd_d90_curve = summarize_curve(succ_qd, qd_grid, d90_qd)
    qw_mean, qw_lo, qw_hi, qw_d90, qw_d90_ci, qw_d90_curve = summarize_curve(succ_qw, qw_grid, d90_qw)

    # -------- Plots & CSV --------
    plt.figure(figsize=(8,5))
    plt.plot(qd_grid, qd_mean, label="QD mean success (COMED=0)", color=COLOR_LINE)
    plt.fill_between(qd_grid, qd_lo, qd_hi, alpha=0.20, label="95% MC band", color=COLOR_FILL)
    plt.axhline(0.90, color="tab:red", ls="--", lw=1.2, label="90% target")
    plt.ylim(0,1); plt.xlabel("Once-daily dose (mg)")
    plt.ylabel(f"Success fraction (PD(t) ≤ {STRICT_THRESHOLD} ng/mL over 24 h)")
    plt.grid(True, alpha=0.35); plt.legend(); plt.tight_layout()
    qd_png = outdir / "qd_no_comed_dose_success.png"
    plt.savefig(qd_png.as_posix(), dpi=180); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(qw_grid, qw_mean, label="QW mean success (COMED=0)", color=COLOR_LINE)
    plt.fill_between(qw_grid, qw_lo, qw_hi, alpha=0.20, label="95% MC band", color=COLOR_FILL)
    plt.axhline(0.90, color="tab:red", ls="--", lw=1.2, label="90% target")
    plt.ylim(0,1); plt.xlabel("Once-weekly dose (mg)")
    plt.ylabel(f"Success fraction (PD(t) ≤ {STRICT_THRESHOLD} ng/mL over 168 h)")
    plt.grid(True, alpha=0.35); plt.legend(); plt.tight_layout()
    qw_png = outdir / "qw_no_comed_dose_success.png"
    plt.savefig(qw_png.as_posix(), dpi=180); plt.close()

    pd.DataFrame({"qd_dose_mg": qd_grid, "succ_mean": qd_mean, "succ_lo2p5": qd_lo, "succ_hi97p5": qd_hi})\
      .to_csv(outdir / "qd_no_comed_dose_success_summary.csv", index=False)
    pd.DataFrame({"qw_dose_mg": qw_grid, "succ_mean": qw_mean, "succ_lo2p5": qw_lo, "succ_hi97p5": qw_hi})\
      .to_csv(outdir / "qw_no_comed_dose_success_summary.csv", index=False)

    # -------- Console summary --------
    def fmt_ci(ci):
        return f"[{ci[0]:.1f}, {ci[1]:.1f}] mg" if np.isfinite(ci[0]) and np.isfinite(ci[1]) else "N/A"

    print("\n=== No-concomitant medication scenario (COMED=0) ===")
    print(f"QD minimal dose for ≥90% success (MC mean): {qd_d90:.1f} mg  | 95% CI: {fmt_ci(qd_d90_ci)}")
    print(f"QD minimal dose (on mean curve):            {qd_d90_curve:.1f} mg")
    print(f"QW minimal dose for ≥90% success (MC mean): {qw_d90:.0f} mg  | 95% CI: {fmt_ci(qw_d90_ci)}")
    print(f"QW minimal dose (on mean curve):            {qw_d90_curve:.0f} mg")

    print("\nSaved:")
    print(f"  {qd_png}")
    print(f"  {qw_png}")
    print(f"  {outdir/'qd_no_comed_dose_success_summary.csv'}")
    print(f"  {outdir/'qw_no_comed_dose_success_summary.csv'}")

if __name__ == "__main__":
    main()
