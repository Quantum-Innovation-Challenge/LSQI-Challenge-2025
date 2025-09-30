#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weekly dose scan with an LSTM PD(t) model trained on 24 h sequences.
Finds minimal once-weekly dose (5 mg steps by default) to reach ≥90% population success
(PD(t) ≤ 3.3 ng/mL for all t in [0,168] at steady state).

Usage:
    python weekly_dose_scan.py --data EstData.csv --out results_dir
"""

import os
import math
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 140

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

# TensorFlow / Keras
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras import layers, callbacks, models


# ----------------------------- Config (defaults) -----------------------------
STRICT_THRESHOLD = 3.3     # ng/mL success threshold
TAU_QD = 24.0              # qd interval seen in training
TAU_QW = 168.0             # target weekly interval
GRID_MAX_WEEKLY = 400.0    # max weekly dose to scan (mg)
WEEKLY_STEP = 5.0          # dose step in mg
VAL_SPLIT = 0.2
SEED = 42

# Monte Carlo
K_EXPAND = 10              # expand population ~10x
R_MC = 200                 # MC replicates (increase for tighter bands)
SIGMA_BW = 0.10            # 10% log-normal jitter on BW
BW_CLIP = (35.0, 120.0)

# Training
EPOCHS = 400               # upper bound on epochs (EarlyStopping will cut earlier)


# ----------------------------- Utilities -----------------------------
def set_seeds(seed: int = SEED):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return rng


def maybe_enable_gpu_memory_growth():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass


def load_raw(path_csv: str) -> pd.DataFrame:
    if not os.path.exists(path_csv):
        raise FileNotFoundError(
            f"Data file not found: {path_csv}. Please provide EstData.csv via --data."
        )
    df = pd.read_csv(path_csv)
    for c in ["ID", "BW", "COMED", "DOSE", "TIME", "DV", "EVID", "MDV", "DVID", "AMT"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ID", "TIME", "EVID", "MDV", "DVID"])
    df = df[df["TIME"] >= 0].sort_values(["ID", "TIME"]).reset_index(drop=True)
    return df


def infer_schedule(s: pd.DataFrame):
    d = s[s["EVID"] == 1]
    if len(d):
        dose = float(d["AMT"].median()) if d["AMT"].notna().any() else float(
            s["DOSE"].replace(0, np.nan).median()
        )
        t = np.sort(d["TIME"].dropna().unique())
        tau = float(np.median(np.diff(t))) if len(t) >= 2 else TAU_QD
    else:
        dose = float(s["DOSE"].replace(0, np.nan).median())
        tau = TAU_QD
    if not np.isfinite(dose):
        dose = 0.0
    return dose, tau


def last_window(s: pd.DataFrame, tau: float, min_points=3, max_expand=8.0):
    if s.empty:
        return s.iloc[0:0].copy()
    tmax = float(s["TIME"].max())
    best, bestn = s.iloc[0:0].copy(), 0
    f = 1.0
    while f <= max_expand:
        w = s[(s["TIME"] >= tmax - tau * f) & (s["TIME"] <= tmax)]
        if len(w) > bestn:
            best, bestn = w.copy(), len(w)
        if len(w) >= min_points:
            return w.sort_values("TIME")
        f *= 1.5
    return best.sort_values("TIME")


def resample_to_grid(time, value, grid):
    if len(time) == 0:
        return np.full_like(grid, np.nan, float)
    ix = np.argsort(time)
    t, v = np.asarray(time)[ix], np.asarray(value)[ix]
    return np.interp(grid, t - t.min(), v, left=np.nan, right=np.nan)


def build_sequences(pd_df: pd.DataFrame, dose_tbl: pd.DataFrame, T=24):
    grid_24h = np.linspace(0, TAU_QD, T)
    seq_X, seq_y, meta = [], [], []
    for sid, s in tqdm(pd_df.groupby("ID"), total=pd_df["ID"].nunique(), leave=False, desc="sequences"):
        drow = dose_tbl.loc[dose_tbl["ID"] == sid]
        if drow.empty:
            continue
        tau = float(drow["INTERVAL_H"].iloc[0]) if np.isfinite(drow["INTERVAL_H"].iloc[0]) else TAU_QD
        w = last_window(s.sort_values("TIME"), tau, min_points=3, max_expand=8.0)
        if w.empty:
            continue

        t = w["TIME"].values.astype(float)
        v = w["DV"].values.astype(float)  # PD in ng/mL
        vg = resample_to_grid(t, v, grid_24h)
        if np.isnan(vg).all():
            continue
        # forward-fill first NaNs, then linear-fill
        if np.isnan(vg[0]):
            first = np.nanmin(np.where(~np.isnan(vg))[0])
            vg[:first] = vg[first]
        nmask = np.isnan(vg)
        if nmask.any():
            vg[nmask] = np.interp(np.where(nmask)[0], np.where(~nmask)[0], vg[~nmask])

        bw = float(drow["BW"].iloc[0]) if np.isfinite(drow["BW"].iloc[0]) else np.nan
        comed = int(drow["COMED"].iloc[0]) if np.isfinite(drow["COMED"].iloc[0]) else 0
        dose = float(drow["DOSE_MG"].iloc[0]) if np.isfinite(drow["DOSE_MG"].iloc[0]) else 0.0
        dose_mgkg = dose / bw if (np.isfinite(bw) and bw > 0) else 0.0

        # Features per step: [time_norm, dose_mgkg, comed, tau_norm]
        X_i = np.column_stack([
            grid_24h / TAU_QD,
            np.full(T, dose_mgkg, float),
            np.full(T, comed, int),
            np.full(T, 1.0, float)  # tau_norm = 24h/24h = 1 during training
        ])
        seq_X.append(X_i)
        seq_y.append(vg)
        meta.append((bw, comed))

    X = np.asarray(seq_X, float)   # (N, T, 4)
    y = np.asarray(seq_y, float)   # (N, T)
    meta = np.asarray(meta, float) # (N, 2) -> [BW, COMED]
    return X, y, meta, grid_24h


def build_and_train_model(X_tr_s, y_tr_z, X_va_s, y_va_z, epochs: int, ckpt_path: str):
    tf.keras.backend.clear_session()
    model = models.Sequential([
        layers.Input(shape=(X_tr_s.shape[1], X_tr_s.shape[2])),
        layers.Masking(mask_value=0.0),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.TimeDistributed(layers.Dense(32, activation="relu")),
        layers.TimeDistributed(layers.Dense(1))  # PD in z-space
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0), loss="mse")

    callbacks_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=60, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=0),
    ]

    hist = model.fit(
        X_tr_s, y_tr_z[..., None],
        validation_data=(X_va_s, y_va_z[..., None]),
        epochs=epochs, batch_size=16, shuffle=True,
        callbacks=callbacks_list, verbose=1
    )
    return model, hist


def make_weekly_batch(BW, COMED, dose_weekly_mg, time_scaler):
    N = len(BW)
    T168 = 168
    tgrid = np.linspace(0, TAU_QW, T168)
    dose_mgkg = dose_weekly_mg / np.clip(BW, 1e-6, None)

    Xb = np.stack([
        tgrid/TAU_QW * np.ones((N, T168)),      # time_norm (weekly)
        np.tile(dose_mgkg[:, None], (1, T168)), # mg/kg
        np.tile(COMED[:, None], (1, T168)),     # comed
        np.full((N, T168), TAU_QW/TAU_QD, float)# tau_norm = 7
    ], axis=-1).astype(float)

    # Scale time channel with the SAME scaler (fit on 24 h)
    Xb[..., 0] = time_scaler.transform(Xb[..., 0].reshape(-1, 1)).reshape(Xb[..., 0].shape)
    return Xb, tgrid


def predict_weekly_pd_ngml(model, y_scaler, time_scaler, BW, COMED, dose_weekly_mg):
    Xb, tgrid = make_weekly_batch(BW, COMED, dose_weekly_mg, time_scaler)
    yhat_z = model.predict(Xb, verbose=0).squeeze(-1)  # (N, 168) z-space
    yhat = y_scaler.inverse_transform(yhat_z.reshape(-1,1)).reshape(yhat_z.shape)  # ng/mL
    return yhat, tgrid


# ----------------------------- Main -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Weekly dose scan with 24h-trained LSTM PD model")
    parser.add_argument("--data", type=str, default="EstData.csv", help="Path to EstData.csv")
    parser.add_argument("--out", type=str, default="results", help="Output directory")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Max training epochs")
    parser.add_argument("--weekly-step", type=float, default=WEEKLY_STEP, help="Weekly dose step (mg)")
    parser.add_argument("--grid-max-weekly", type=float, default=GRID_MAX_WEEKLY, help="Weekly dose grid max (mg)")
    parser.add_argument("--mc", type=int, default=R_MC, help="Number of MC replicates")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Reproducibility + GPU memory
    rng = set_seeds(args.seed)
    maybe_enable_gpu_memory_growth()

    # Step 0: Load & light-clean
    print("Step 0/7: Loading data …")
    df = load_raw(args.data)
    obs = df[(df["EVID"] == 0) & (df["MDV"] == 0)].copy()
    # pk = obs[obs["DVID"] == 1].copy()  # not used further
    pd_df = obs[obs["DVID"] == 2].copy()

    # Step 1: Per-subject dosing
    print("Step 1/7: Inferring per-subject dosing …")
    rows = []
    for sid, s in tqdm(df.groupby("ID"), total=df["ID"].nunique(), leave=False, desc="dosing"):
        bw = float(s["BW"].dropna().iloc[0]) if s["BW"].notna().any() else np.nan
        comed = int(s["COMED"].dropna().iloc[0]) if s["COMED"].notna().any() else 0
        dose, tau = infer_schedule(s)
        rows.append({"ID": int(sid), "BW": bw, "COMED": comed, "DOSE_MG": dose, "INTERVAL_H": tau})
    dose_tbl = pd.DataFrame(rows)

    # Step 2: Build PD sequences (24 h)
    print("Step 2/7: Building 24 h sequences …")
    X, y, meta, grid_24h = build_sequences(pd_df, dose_tbl, T=24)
    print(f"Built sequences: N={len(X)}  T={X.shape[1]}  F={X.shape[2]}")

    if len(X) < 4:
        raise RuntimeError("Too few sequences to train. Check EstData.csv contents.")

    # Step 3: Train/val split + scaling
    print("Step 3/7: Train/validation split + scaling …")
    X, y, meta = shuffle(X, y, meta, random_state=args.seed)
    X_tr, X_va, y_tr, y_va, meta_tr, meta_va = train_test_split(
        X, y, meta, test_size=VAL_SPLIT, random_state=args.seed
    )

    # Standardize PD target on TRAIN only
    y_scaler = StandardScaler()
    y_tr_z = y_scaler.fit_transform(y_tr.reshape(-1, 1)).reshape(y_tr.shape)
    y_va_z = y_scaler.transform(y_va.reshape(-1, 1)).reshape(y_va.shape)

    # Standardize time feature only; keep others as-is
    time_scaler = StandardScaler()
    X_tr_s = X_tr.copy(); X_va_s = X_va.copy()
    t_tr_vec = X_tr[..., 0].reshape(-1, 1)  # (N*T, 1)
    time_scaler.fit(t_tr_vec)
    X_tr_s[..., 0] = time_scaler.transform(X_tr[..., 0].reshape(-1, 1)).reshape(X_tr[..., 0].shape)
    X_va_s[..., 0] = time_scaler.transform(X_va[..., 0].reshape(-1, 1)).reshape(X_va[..., 0].shape)

    # Step 4: Build & train LSTM
    print("Step 4/7: Training LSTM …")
    ckpt_path = str(outdir / "lstm_best.h5")
    model, hist = build_and_train_model(X_tr_s, y_tr_z, X_va_s, y_va_z, args.epochs, ckpt_path)

    # Plot loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("MSE loss"); plt.legend(); plt.tight_layout()
    loss_png = outdir / "lstm_qw_loss_curve.png"
    plt.savefig(loss_png.as_posix(), dpi=160); plt.close()
    print(f"Saved: {loss_png.name}")

    # Step 5: Weekly simulator helpers are already defined

    # Step 6: MC weekly dose scan
    print("Step 6/7: Monte-Carlo weekly dose scan …")
    dose_grid = np.arange(args.weekly_step, args.grid_max_weekly + args.weekly_step, args.weekly_step)
    succ_mat = np.zeros((args.mc, len(dose_grid)))
    dose90 = np.full(args.mc, np.nan)

    # Base population from data:
    BW_obs = meta[:, 0]
    COM_obs = meta[:, 1]
    BW_obs = BW_obs[np.isfinite(BW_obs)]
    if BW_obs.size == 0:
        BW_obs = np.array([70.0])
    p_comed = 0.5 if not np.isfinite(COM_obs).any() else float(np.nanmean(COM_obs))

    for r in tqdm(range(args.mc), desc="weekly dose grid"):
        # draw population ~10x base
        N_base = len(BW_obs)
        base_idx = rng.integers(0, N_base, size=N_base * K_EXPAND)
        BW = BW_obs[base_idx] * np.exp(rng.normal(0.0, SIGMA_BW, size=N_base * K_EXPAND))
        BW = np.clip(BW, *BW_CLIP)
        COM = rng.binomial(1, p_comed, size=N_base * K_EXPAND)

        s = []
        for d in dose_grid:
            yhat, _ = predict_weekly_pd_ngml(model, y_scaler, time_scaler, BW, COM, d)  # (N, 168) ng/mL
            ok = (np.nanmax(yhat, axis=1) <= STRICT_THRESHOLD)  # success if PD ≤ 3.3 for all 168 h
            s.append(ok.mean())
        s = np.array(s)
        succ_mat[r, :] = s
        idx90 = np.where(s >= 0.90)[0]
        if idx90.size:
            dose90[r] = float(dose_grid[idx90[0]])

    succ_mean = succ_mat.mean(axis=0)
    succ_lo   = np.percentile(succ_mat, 2.5, axis=0)
    succ_hi   = np.percentile(succ_mat, 97.5, axis=0)

    dose90_mean = float(np.nanmean(dose90)) if np.isfinite(dose90).any() else np.nan
    dose90_ci = (
        (float(np.nanpercentile(dose90, 2.5)), float(np.nanpercentile(dose90, 97.5)))
        if np.isfinite(dose90).any()
        else (np.nan, np.nan)
    )

    # Step 7: Plot & report
    print("Step 7/7: Plotting …")
    plt.figure(figsize=(8, 5))
    plt.plot(dose_grid, succ_mean, label="Weekly success fraction (LSTM)")
    plt.fill_between(dose_grid, succ_lo, succ_hi, alpha=0.20, label="95% MC band")
    plt.axhline(0.90, color="tab:red", ls="--", lw=1.2, label="90% target")
    plt.ylim(0, 1); plt.xlabel("Weekly dose (mg)")
    plt.ylabel(f"Success fraction (PD(t) ≤ {STRICT_THRESHOLD} ng/mL over 168 h)")
    plt.grid(True, alpha=0.35); plt.legend(); plt.tight_layout()
    dose_png = outdir / "lstm_weekly_dose_success.png"
    plt.savefig(dose_png.as_posix(), dpi=180); plt.close()

    summary_csv = outdir / "lstm_weekly_dose_success_summary.csv"
    pd.DataFrame({
        "weekly_dose_mg": dose_grid,
        "succ_mean": succ_mean,
        "succ_lo2p5": succ_lo,
        "succ_hi97p5": succ_hi
    }).to_csv(summary_csv, index=False)

    if np.isfinite(dose90_mean):
        print(f"Minimal weekly dose for 90% success (mean): {dose90_mean:.0f} mg"
              f" | 95% CI: [{dose90_ci[0]:.0f}, {dose90_ci[1]:.0f}] mg")
    else:
        print("No weekly dose on the grid achieved ≥90% success. "
              "Increase --grid-max-weekly or review model fit.")

    print("Saved files:")
    print(f"  {loss_png}")
    print(f"  {dose_png}")
    print(f"  {summary_csv}")


if __name__ == "__main__":
    main()
