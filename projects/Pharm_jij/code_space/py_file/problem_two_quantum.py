#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effect of changing body weight distribution to 70–140 kg on optimal QD/QW doses (QLSTM version)
-----------------------------------------------------------------------------------------------
- QLSTM (PennyLane + PyTorch) surrogate for PD(t)
- Loads a trained checkpoint if available, otherwise trains then saves one
- Rebuilds/loads time & PD scalers from Phase-1 data
- Monte Carlo with BW ~ Uniform(70, 140) kg, COMED ~ Bernoulli(p_phase1)
- QD grid auto-extends (0.5 mg steps) until crossing 90% success
- QW grid fixed at 5 mg steps
- Success = max_t PD(t) ≤ 3.3 ng/mL over full interval at steady state
- Outputs minimal dose achieving ≥90% success (mean & 95% CI) for QD & QW

Requirements:
  pip install pennylane pennylane-qchem torch numpy pandas scikit-learn matplotlib tqdm joblib
"""

import os
import math
import warnings
import argparse
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

import joblib

import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

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

VAL_SPLIT         = 0.2
T_POINTS          = 24          # hourly grid over 24 h for training
N_QUBITS          = 4
N_QLAYERS         = 1
HIDDEN_SIZE       = 64
BATCH_SIZE        = 8
MAX_EPOCHS        = 300
PATIENCE          = 120
MIN_DELTA         = 1e-3
LR                = 1e-3
WD                = 5e-4

# ----------------------------- QLSTM -----------------------------
class QLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits=4, n_qlayers=1,
                 batch_first=True, backend="default.qubit", input_scale_pi=True):
        super().__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.batch_first = batch_first
        self.backend = backend
        self.input_scale_pi = input_scale_pi

        self.wires = [f"w{i}" for i in range(n_qubits)]
        self.dev = qml.device(backend, wires=self.wires, shots=None)

        def _circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=self.wires)
            qml.BasicEntanglerLayers(weights, wires=self.wires)
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]

        self.qnode = qml.QNode(_circuit, self.dev, interface="torch", diff_method="backprop")
        self.weight_shapes = {"weights": (n_qlayers, n_qubits)}
        self.vqc = qml.qnn.TorchLayer(self.qnode, self.weight_shapes)

        # Classical projections around the VQC
        self.cl_in  = nn.Linear(self.concat_size, n_qubits)
        self.act_in = nn.Tanh()
        self.cl_out = nn.Linear(n_qubits, hidden_size)

        # Gate-specific projections from the shared quantum embedding
        self.proj_forget = nn.Linear(hidden_size, hidden_size)
        self.proj_input  = nn.Linear(hidden_size, hidden_size)
        self.proj_update = nn.Linear(hidden_size, hidden_size)
        self.proj_output = nn.Linear(hidden_size, hidden_size)

        for m in [self.cl_in, self.cl_out, self.proj_forget, self.proj_input, self.proj_update, self.proj_output]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

        print(f"[QLSTM] n_qubits={n_qubits}, n_qlayers={n_qlayers}")

    def _angles(self, vt):
        y = self.cl_in(vt)
        y = self.act_in(y)
        if self.input_scale_pi:
            y = math.pi * y
        return y

    def forward(self, x):
        if self.batch_first:
            B, T, F = x.size()
        else:
            T, B, F = x.size()
            x = x.transpose(0, 1)

        h_t = torch.zeros(B, self.hidden_size, device=x.device)
        c_t = torch.zeros(B, self.hidden_size, device=x.device)
        out_seq = []

        for t in range(T):
            xt = x[:, t, :]
            vt = torch.cat([h_t, xt], dim=1)
            ang = self._angles(vt)
            z   = self.vqc(ang)
            qh  = self.cl_out(z)

            f_t = torch.sigmoid(self.proj_forget(qh))
            i_t = torch.sigmoid(self.proj_input(qh))
            g_t = torch.tanh   (self.proj_update(qh))
            o_t = torch.sigmoid(self.proj_output(qh))

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            out_seq.append(h_t.unsqueeze(1))

        out_seq = torch.cat(out_seq, dim=1)  # (B,T,H)
        return out_seq

class QLSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, n_qubits=4, n_qlayers=1, backend="default.qubit"):
        super().__init__()
        self.qlstm = QLSTM(input_size, hidden_size, n_qubits, n_qlayers, backend=backend)
        self.head  = nn.Linear(hidden_size, 1)
        print(f"[QLSTMRegressor] Qubits in use: {self.qlstm.n_qubits}")

    def forward(self, x):
        h = self.qlstm(x)      # (B,T,H)
        y = self.head(h)       # (B,T,1)
        return y

# ----------------------------- Utils -----------------------------
def save_scaler(scaler: StandardScaler, path: str):
    joblib.dump(scaler, path)

def load_scaler(path: str) -> StandardScaler:
    return joblib.load(path)

def load_raw(data_csv: str):
    df = pd.read_csv(data_csv)
    for c in ["ID","BW","COMED","DOSE","TIME","DV","EVID","MDV","DVID","AMT"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ID","TIME","EVID","MDV","DVID"])
    df = df[df["TIME"] >= 0].sort_values(["ID","TIME"]).reset_index(drop=True)
    return df

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

def build_phase1_seq(df: pd.DataFrame, tau_default=TAU_QD, T=T_POINTS):
    obs = df[(df["EVID"]==0) & (df["MDV"]==0)]
    pd_obs = obs[obs["DVID"]==2].copy()
    times = np.linspace(0, tau_default, T)
    rowsX, rowsY, meta = [], [], []

    # simple per-ID summaries
    dose_tbl = []
    for sid, s in df.groupby("ID"):
        d = s[s["EVID"]==1]
        if len(d):
            dose = float(d["AMT"].median()) if d["AMT"].notna().any() else float(s["DOSE"].replace(0,np.nan).median())
            tt = np.sort(d["TIME"].dropna().unique())
            tau = float(np.median(np.diff(tt))) if len(tt) >= 2 else TAU_QD
        else:
            dose = float(s["DOSE"].replace(0,np.nan).median()); tau = TAU_QD
        bw = float(s["BW"].dropna().iloc[0]) if s["BW"].notna().any() else np.nan
        com = int(s["COMED"].dropna().iloc[0]) if s["COMED"].notna().any() else 0
        dose_tbl.append({"ID":int(sid),"BW":bw,"COMED":com,"DOSE_MG":dose,"INTERVAL_H":tau})
    dose_tbl = pd.DataFrame(dose_tbl)

    for sid, s in pd_obs.groupby("ID"):
        row = dose_tbl[dose_tbl["ID"]==sid]
        if row.empty: continue
        tau = float(row["INTERVAL_H"].iloc[0]) if np.isfinite(row["INTERVAL_H"].iloc[0]) else TAU_QD
        w = last_window(s.sort_values("TIME"), tau, 3, 8.0)
        if w.empty: continue
        vg = resample_to_grid(w["TIME"].values.astype(float), w["DV"].values.astype(float), times)
        if np.isnan(vg).all(): continue

        bw = float(row["BW"].iloc[0]) if np.isfinite(row["BW"].iloc[0]) else np.nan
        com = int(row["COMED"].iloc[0]) if np.isfinite(row["COMED"].iloc[0]) else 0
        dose = float(row["DOSE_MG"].iloc[0]) if np.isfinite(row["DOSE_MG"].iloc[0]) else 0.0
        dose_mgkg = dose / bw if (np.isfinite(bw) and bw>0) else 0.0

        X_i = np.column_stack([
            times / TAU_QD,                # time in [0,1]
            np.full(T, dose_mgkg, float),  # mg/kg
            np.full(T, com, int)           # COMED flag
        ])
        rowsX.append(X_i); rowsY.append(vg); meta.append((bw, com))

    X = np.asarray(rowsX, float)  # (N,T,3)
    y = np.asarray(rowsY, float)  # (N,T)
    meta = np.asarray(meta, float)
    return X, y, meta, times

def fit_or_load_scalers(outdir: Path, df: pd.DataFrame):
    time_p = outdir / "time_scaler_qd.pkl"
    y_p    = outdir / "y_scaler_pd.pkl"
    if time_p.exists() and y_p.exists():
        print("Found saved scalers. Loading …")
        return load_scaler(time_p.as_posix()), load_scaler(y_p.as_posix())
    print("Fitting scalers from Phase-1 data …")
    X, y, _, _ = build_phase1_seq(df)
    time_scaler = StandardScaler().fit(X[...,0].reshape(-1,1))
    y_scaler    = StandardScaler().fit(y.reshape(-1,1))
    save_scaler(time_scaler, time_p.as_posix())
    save_scaler(y_scaler,    y_p.as_posix())
    print("✓ Scalers fitted & saved.")
    return time_scaler, y_scaler

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y[...,None], dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

def train_or_load_model(ckpt_path: Path, df: pd.DataFrame, time_scaler: StandardScaler, y_scaler: StandardScaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QLSTMRegressor(input_size=3, hidden_size=HIDDEN_SIZE, n_qubits=N_QUBITS,
                           n_qlayers=N_QLAYERS, backend="default.qubit").to(device)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path.as_posix(), map_location=device))
        print(f"✓ Loaded checkpoint: {ckpt_path}")
        return model, device

    print("No checkpoint found — training QLSTM on Phase-1 PD sequences …")
    X, y, _, _ = build_phase1_seq(df)
    # scale: time channel only; leave mg/kg & COMED as-is
    Xs = X.copy()
    Xs[...,0] = time_scaler.transform(Xs[...,0])
    # PD z-space target
    yz = y_scaler.transform(y.reshape(-1,1)).reshape(y.shape)

    Xs, yz = shuffle(Xs, yz, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(Xs, yz, test_size=VAL_SPLIT, random_state=SEED)

    train_ds = SeqDataset(X_tr, y_tr); val_ds = SeqDataset(X_va, y_va)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=12, cooldown=8, min_lr=1e-5)
    crit = nn.MSELoss()

    best_val = float("inf"); best_state = None; wait=0
    for ep in range(1, MAX_EPOCHS+1):
        model.train(); tl=0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            yhat = model(xb)
            loss = crit(yhat, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()*xb.size(0)
        tl /= len(train_ds)

        model.eval(); vl=0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                yhat = model(xb)
                loss = crit(yhat, yb)
                vl += loss.item()*xb.size(0)
        vl /= len(val_ds)
        sch.step(vl)
        print(f"Epoch {ep:03d} | train MSE {tl:.5f} | val MSE {vl:.5f}")

        if vl < best_val - MIN_DELTA:
            best_val = vl; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping."); break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), ckpt_path.as_posix())
    print(f"✓ Saved checkpoint: {ckpt_path}")
    return model, device

def phase1_comed_prob(df: pd.DataFrame) -> float:
    ids = df["ID"].dropna().unique()
    vals = []
    for sid in ids:
        s = df[df["ID"]==sid]
        if s["COMED"].notna().any():
            v = int(s["COMED"].dropna().iloc[0])
            if v in (0,1): vals.append(v)
    return 0.5 if len(vals)==0 else float(np.mean(vals))

def make_batch(time_scaler: StandardScaler, TAU, Tpoints, BW, COMED, dose_mg, tau_norm):
    N = len(BW)
    tgrid = np.linspace(0, TAU, Tpoints)
    dose_mgkg = dose_mg / np.clip(BW, 1e-6, None)
    Xb = np.stack([
        (tgrid/TAU) * np.ones((N, Tpoints)),      # time_norm
        np.tile(dose_mgkg[:, None], (1, Tpoints)),
        np.tile(COMED[:, None], (1, Tpoints)),
    ], axis=-1).astype(float)
    Xb[..., 0] = time_scaler.transform(Xb[..., 0])
    return Xb, tgrid

@torch.no_grad()
def predict_qd_pd_ngml(model, device, y_scaler, time_scaler, BW, COMED, dose_mg, Tpoints=24):
    Xb, tgrid = make_batch(time_scaler, TAU_QD, Tpoints, BW, COMED, dose_mg, tau_norm=1.0)
    xb = torch.tensor(Xb, dtype=torch.float32, device=device)
    yhat_z = model(xb).squeeze(-1).cpu().numpy()
    yhat   = y_scaler.inverse_transform(yhat_z.reshape(-1,1)).reshape(yhat_z.shape)
    return yhat, tgrid

@torch.no_grad()
def predict_qw_pd_ngml(model, device, y_scaler, time_scaler, BW, COMED, dose_weekly_mg, Tpoints=168):
    Xb, tgrid = make_batch(time_scaler, TAU_QW, Tpoints, BW, COMED, dose_weekly_mg, tau_norm=TAU_QW/TAU_QD)
    xb = torch.tensor(Xb, dtype=torch.float32, device=device)
    yhat_z = model(xb).squeeze(-1).cpu().numpy()
    yhat   = y_scaler.inverse_transform(yhat_z.reshape(-1,1)).reshape(yhat_z.shape)
    return yhat, tgrid

def mc_success_curve_qd(rng, model, device, y_scaler, time_scaler, dose_grid, N_base, p_comed, R=R_MC):
    succ_mat = np.zeros((R, len(dose_grid)))
    dose90   = np.full(R, np.nan)
    for r in tqdm(range(R), desc="QD MC"):
        BW = rng.uniform(BW_RANGE[0], BW_RANGE[1], size=N_base*K_EXPAND)
        COM = rng.binomial(1, p_comed, size=N_base*K_EXPAND)
        s = []
        for d in dose_grid:
            yhat, _ = predict_qd_pd_ngml(model, device, y_scaler, time_scaler, BW, COM, d, Tpoints=24)
            ok = (np.nanmax(yhat, axis=1) <= STRICT_THRESHOLD)
            s.append(ok.mean())
        s = np.array(s)
        succ_mat[r,:] = s
        i90 = np.where(s >= 0.90)[0]
        if i90.size:
            dose90[r] = float(dose_grid[i90[0]])
    return succ_mat, dose90

def mc_success_curve_qw(rng, model, device, y_scaler, time_scaler, dose_grid, N_base, p_comed, R=R_MC):
    succ_mat = np.zeros((R, len(dose_grid)))
    dose90   = np.full(R, np.nan)
    for r in tqdm(range(R), desc="QW MC"):
        BW = rng.uniform(BW_RANGE[0], BW_RANGE[1], size=N_base*K_EXPAND)
        COM = rng.binomial(1, p_comed, size=N_base*K_EXPAND)
        s = []
        for d in dose_grid:
            yhat, _ = predict_qw_pd_ngml(model, device, y_scaler, time_scaler, BW, COM, d, Tpoints=168)
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

def run_qd_until_cross(rng, model, device, y_scaler, time_scaler, N_base,
                       start=0.5, step=0.5, initial_max=40.0,
                       extend_by=20.0, max_extends=10, hard_cap=240.0, p_comed=0.5):
    grid = np.arange(start, initial_max + step, step)
    succ_qd, d90_qd = mc_success_curve_qd(rng, model, device, y_scaler, time_scaler, grid, N_base, p_comed, R=R_MC)
    succ_mean = succ_qd.mean(axis=0)
    if float(np.nanmax(succ_mean)) >= 0.90:
        return grid, succ_qd, d90_qd
    extends = 0
    while extends < max_extends and grid[-1] < hard_cap:
        last = grid[-1]
        new_top = min(last + extend_by, hard_cap)
        grid = np.arange(step, new_top + step, step)
        succ_qd, d90_qd = mc_success_curve_qd(rng, model, device, y_scaler, time_scaler, grid, N_base, p_comed, R=R_MC)
        succ_mean = succ_qd.mean(axis=0)
        if float(np.nanmax(succ_mean)) >= 0.90:
            return grid, succ_qd, d90_qd
        extends += 1
    return grid, succ_qd, d90_qd

def fmt_ci(ci):
    return f"[{ci[0]:.1f}, {ci[1]:.1f}] mg" if np.isfinite(ci[0]) and np.isfinite(ci[1]) else "N/A"

# ----------------------------- Main -----------------------------
def main():
    parser = argparse.ArgumentParser(description="BW shift (70–140 kg) QD/QW dose scan using a QLSTM PD(t) model")
    parser.add_argument("--data", type=str, default="EstData.csv", help="Path to EstData.csv")
    parser.add_argument("--ckpt", type=str, default="qlstm_ckpt.pt", help="Path to QLSTM checkpoint (.pt)")
    parser.add_argument("--out",  type=str, default="results_bw70140_qlstm", help="Output directory")
    parser.add_argument("--qd-step", type=float, default=QD_STEP, help="QD dose step (mg)")
    parser.add_argument("--qd-initial-max", type=float, default=QD_MAX_DOSE, help="QD initial grid max (mg)")
    parser.add_argument("--qw-step", type=float, default=QW_STEP, help="QW dose step (mg)")
    parser.add_argument("--qw-max", type=float, default=QW_MAX_DOSE, help="QW grid max (mg)")
    parser.add_argument("--mc", type=int, default=R_MC, help="MC replicates")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--qubits", type=int, default=N_QUBITS, help="QLSTM qubits")
    parser.add_argument("--qlayers", type=int, default=N_QLAYERS, help="QLSTM layers")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    df = load_raw(args.data)

    # scalers
    time_scaler, y_scaler = fit_or_load_scalers(outdir, df)

    # (re)build model with requested qubits/layers & train/load
    global N_QUBITS, N_QLAYERS
    N_QUBITS  = int(args.qubits)
    N_QLAYERS = int(args.qlayers)

    ckpt_path = Path(args.ckpt)
    model, device = train_or_load_model(ckpt_path, df, time_scaler, y_scaler)

    # Phase-1 COMED probability & base N
    p_comed = phase1_comed_prob(df)
    obs = df[(df["EVID"]==0) & (df["MDV"]==0)]
    N_base = int(obs[obs["DVID"]==2]["ID"].nunique())
    if N_base < 1: N_base = 48

    print(f"Phase-1 PD subjects: {N_base}")
    print(f"Target BW distribution for simulation: Uniform[{BW_RANGE[0]}, {BW_RANGE[1]}] kg")
    print("Running QD and QW Monte-Carlo …")

    # QD: auto-extend until 90% crossing
    qd_grid, succ_qd, d90_qd = run_qd_until_cross(
        rng, model, device, y_scaler, time_scaler, N_base,
        start=args.qd_step, step=args.qd_step,
        initial_max=args.qd_initial_max, extend_by=20.0, max_extends=10,
        hard_cap=240.0, p_comed=p_comed
    )

    # QW: fixed grid
    qw_grid = np.arange(args.qw_step, args.qw_max + args.qw_step, args.qw_step)
    succ_qw, d90_qw = mc_success_curve_qw(rng, model, device, y_scaler, time_scaler, qw_grid, N_base, p_comed, R=args.mc)

    qd_mean, qd_lo, qd_hi, qd_d90, qd_d90_ci, qd_d90_curve = summarize_curve(succ_qd, qd_grid, d90_qd)
    qw_mean, qw_lo, qw_hi, qw_d90, qw_d90_ci, qw_d90_curve = summarize_curve(succ_qw, qw_grid, d90_qw)

    # Plots
    plt.figure(figsize=(8,5))
    plt.plot(qd_grid, qd_mean, label="QD mean success (QLSTM)")
    plt.fill_between(qd_grid, qd_lo, qd_hi, alpha=0.2, label="QD 95% MC band")
    plt.axhline(0.90, color="tab:red", ls="--", lw=1.2, label="90% target")
    plt.ylim(0,1); plt.xlabel("Once-daily dose (mg)")
    plt.ylabel(f"Success fraction (PD(t) ≤ {STRICT_THRESHOLD} ng/mL over 24 h)")
    plt.grid(True, alpha=0.35); plt.legend(); plt.tight_layout()
    qd_png = outdir / "qd_70_140kg_dose_success_qlstm.png"
    plt.savefig(qd_png.as_posix(), dpi=180); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(qw_grid, qw_mean, label="QW mean success (QLSTM)")
    plt.fill_between(qw_grid, qw_lo, qw_hi, alpha=0.2, label="QW 95% MC band")
    plt.axhline(0.90, color="tab:red", ls="--", lw=1.2, label="90% target")
    plt.ylim(0,1); plt.xlabel("Once-weekly dose (mg)")
    plt.ylabel(f"Success fraction (PD(t) ≤ {STRICT_THRESHOLD} ng/mL over 168 h)")
    plt.grid(True, alpha=0.35); plt.legend(); plt.tight_layout()
    qw_png = outdir / "qw_70_140kg_dose_success_qlstm.png"
    plt.savefig(qw_png.as_posix(), dpi=180); plt.close()

    # CSVs
    qd_csv = outdir / "qd_70_140kg_dose_success_summary_qlstm.csv"
    pd.DataFrame({
        "qd_dose_mg": qd_grid,
        "succ_mean": qd_mean, "succ_lo2p5": qd_lo, "succ_hi97p5": qd_hi
    }).to_csv(qd_csv, index=False)

    qw_csv = outdir / "qw_70_140kg_dose_success_summary_qlstm.csv"
    pd.DataFrame({
        "qw_dose_mg": qw_grid,
        "succ_mean": qw_mean, "succ_lo2p5": qw_lo, "succ_hi97p5": qw_hi
    }).to_csv(qw_csv, index=False)

    # Answers
    def _fmt_ci(ci):
        return f"[{ci[0]:.1f}, {ci[1]:.1f}] mg" if np.isfinite(ci[0]) and np.isfinite(ci[1]) else "N/A"

    print("\n=== QLSTM Results for BW ~ Uniform(70, 140) kg ===")
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
    print(f"  {Path(args.ckpt).resolve()}")

if __name__ == "__main__":
    main()
