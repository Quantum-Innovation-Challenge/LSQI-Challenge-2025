#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLSTM PK→PD (data-driven) + Monte-Carlo dose scan with proper units & inverse-scaling
- Replaces the classical Keras LSTM with a PennyLane-powered Quantum LSTM (QLSTM) in PyTorch.
- Prints the number of qubits used.

Requirements (tested with recent versions):
  pip install pennylane pennylane-qchem torch numpy pandas scikit-learn matplotlib tqdm

Notes:
- Backend defaults to 'default.qubit'. You can switch to other PennyLane devices if available.
- Quantum training is slower; start with small n_qubits and n_qlayers.
"""
import os, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 140

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

import pennylane as qml

# ---------------- Config ----------------
STRICT_THRESHOLD = 3.3          # ng/mL success threshold (applied AFTER inverse-scaling)
TAU_H            = 24.0
GRID_MAX         = 40.0
DOSE_STEP        = 0.5
VAL_SPLIT        = 0.2
SEED             = 42

K_EXPAND         = 10           # MC expansion
R_MC             = 200          # MC replicates for error bars (raise later)
SIGMA_BW         = 0.10         # log-normal jitter on BW
BW_CLIP          = (35.0, 120.0)

rng = np.random.default_rng(SEED)
np.random.seed(SEED)
_ = torch.manual_seed(SEED)

# ============ QLSTM module (PennyLane + Torch) ============
class QLSTM(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 n_qubits=4,
                 n_qlayers=1,
                 batch_first=True,
                 return_sequences=False, 
                 return_state=False,
                 backend="default.qubit",
                 input_scale_pi=True):
        super(QLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend
        self.input_scale_pi = input_scale_pi

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        # Named wires per gate type
        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input  = [f"wire_input_{i}"  for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        # Use analytic, backprop-compatible simulator for stable gradients
        self.dev_forget = qml.device(self.backend, wires=self.wires_forget, shots=None)
        self.dev_input  = qml.device(self.backend, wires=self.wires_input,  shots=None)
        self.dev_update = qml.device(self.backend, wires=self.wires_update, shots=None)
        self.dev_output = qml.device(self.backend, wires=self.wires_output, shots=None)

        def _circuit_forget(inputs, weights):
            qml.AngleEmbedding(inputs, wires=self.wires_forget)
            qml.BasicEntanglerLayers(weights, wires=self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]
        self.qlayer_forget = qml.QNode(_circuit_forget, self.dev_forget, interface="torch", diff_method="backprop")

        def _circuit_input(inputs, weights):
            qml.AngleEmbedding(inputs, wires=self.wires_input)
            qml.BasicEntanglerLayers(weights, wires=self.wires_input)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_input]
        self.qlayer_input = qml.QNode(_circuit_input, self.dev_input, interface="torch", diff_method="backprop")

        def _circuit_update(inputs, weights):
            qml.AngleEmbedding(inputs, wires=self.wires_update)
            qml.BasicEntanglerLayers(weights, wires=self.wires_update)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]
        self.qlayer_update = qml.QNode(_circuit_update, self.dev_update, interface="torch", diff_method="backprop")

        def _circuit_output(inputs, weights):
            qml.AngleEmbedding(inputs, wires=self.wires_output)
            qml.BasicEntanglerLayers(weights, wires=self.wires_output)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]
        self.qlayer_output = qml.QNode(_circuit_output, self.dev_output, interface="torch", diff_method="backprop")

        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")
        print(f"[QLSTM] Using n_qubits = {n_qubits}")

        # Input projection -> bound to [-pi, pi] for stable angle embedding
        self.clayer_in = nn.Linear(self.concat_size, n_qubits)
        self.in_act = nn.Tanh()

        self.VQC = {
            'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            'input':  qml.qnn.TorchLayer(self.qlayer_input,  weight_shapes),
            'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
        }
        self.clayer_out = nn.Linear(self.n_qubits, self.hidden_size)

        # Small init on linear layers helps avoid saturating quantum angles
        for m in [self.clayer_in, self.clayer_out]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def _prep_angles(self, v_t):
        y = self.clayer_in(v_t)
        y = self.in_act(y)
        if self.input_scale_pi:
            y = math.pi * y
        return y

    def forward(self, x, init_states=None):
        if self.batch_first:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            x_t = x[:, t, :]
            v_t = torch.cat((h_t, x_t), dim=1)
            y_t = self._prep_angles(v_t)

            f_t = torch.sigmoid(self.clayer_out(self.VQC['forget'](y_t)))
            i_t = torch.sigmoid(self.clayer_out(self.VQC['input'](y_t)))
            g_t = torch.tanh   (self.clayer_out(self.VQC['update'](y_t)))
            o_t = torch.sigmoid(self.clayer_out(self.VQC['output'](y_t)))

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

# A small wrapper that maps the QLSTM hidden states to a scalar PD per timestep
class QLSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, n_qubits=4, n_qlayers=1, backend="default.qubit"):
        super().__init__()
        self.lstm = QLSTM(input_size=input_size, hidden_size=hidden_size, n_qubits=n_qubits,
                          n_qlayers=n_qlayers, batch_first=True, backend=backend)
        self.fc = nn.Linear(hidden_size, 1)
        print(f"[QLSTMRegressor] Qubits in use: {self.lstm.n_qubits}")

    def forward(self, x):
        out_seq, _ = self.lstm(x)
        y = self.fc(out_seq)
        return y

# ------------- Load raw & light-clean -------------
def load_raw():
    df = pd.read_csv("EstData.csv")
    for c in ["ID","BW","COMED","DOSE","TIME","DV","EVID","MDV","DVID","AMT"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ID","TIME","EVID","MDV","DVID"])
    df = df[df["TIME"] >= 0].sort_values(["ID","TIME"]).reset_index(drop=True)
    return df

df = load_raw()
obs = df[(df["EVID"]==0) & (df["MDV"]==0)].copy()
pk  = obs[obs["DVID"]==1].copy()
pd_df = obs[obs["DVID"]==2].copy()

# -------- per-subject dosing summary --------
def infer_schedule(s):
    d = s[s["EVID"]==1]
    if len(d):
        dose = float(d["AMT"].median()) if d["AMT"].notna().any() else float(s["DOSE"].replace(0,np.nan).median())
        t = np.sort(d["TIME"].dropna().unique())
        tau = float(np.median(np.diff(t))) if len(t) >= 2 else TAU_H
    else:
        dose = float(s["DOSE"].replace(0,np.nan).median()); tau = TAU_H
    if not np.isfinite(dose): dose = 0.0
    return dose, tau

rows = []
print("Step 1/7: Dosing summary …")
for sid, s in tqdm(df.groupby("ID"), total=df["ID"].nunique(), leave=False, desc="dosing"):
    bw = float(s["BW"].dropna().iloc[0]) if s["BW"].notna().any() else np.nan
    comed = int(s["COMED"].dropna().iloc[0]) if s["COMED"].notna().any() else 0
    dose, tau = infer_schedule(s)
    rows.append({"ID":int(sid), "BW":bw, "COMED":comed, "DOSE_MG":dose, "INTERVAL_H":tau})
dose_tbl = pd.DataFrame(rows)

# -------- build 24h PD sequences (supervised PD prediction) --------
def last_window(s, tau, min_points=3, max_expand=8.0):
    if s.empty: return s.iloc[0:0].copy()
    tmax = float(s["TIME"].max()); best, bestn = s.iloc[0:0].copy(), 0; f=1.0
    while f <= max_expand:
        w = s[(s["TIME"]>=tmax-tau*f) & (s["TIME"]<=tmax)]
        if len(w)>bestn: best, bestn = w.copy(), len(w)
        if len(w) >= min_points: return w.sort_values("TIME")
        f *= 1.5
    return best.sort_values("TIME")

T = 24  # 24 time steps across the interval (hourly target grid)
times_grid = np.linspace(0, TAU_H, T)

def resample_to_grid(time, value, grid):
    if len(time)==0: return np.full_like(grid, np.nan, float)
    ix = np.argsort(time)
    t, v = np.asarray(time)[ix], np.asarray(value)[ix]
    return np.interp(grid, t - t.min(), v, left=np.nan, right=np.nan)

print("Step 2/7: Building PD sequences …")
seq_X, seq_y, seq_ids, meta = [], [], [], []
for sid, s in tqdm(pd_df.groupby("ID"), total=pd_df["ID"].nunique(), leave=False, desc="sequences"):
    drow = dose_tbl.loc[dose_tbl["ID"]==sid]
    if drow.empty: continue
    tau = float(drow["INTERVAL_H"].iloc[0]) if np.isfinite(drow["INTERVAL_H"].iloc[0]) else TAU_H
    w = last_window(s.sort_values("TIME"), tau, min_points=3, max_expand=8.0)
    if w.empty: continue

    t = w["TIME"].values.astype(float)
    v = w["DV"].values.astype(float)  # ng/mL
    vg = resample_to_grid(t, v, times_grid)
    if np.isnan(vg).all(): continue
    if np.isnan(vg[0]): 
        first = np.nanmin(np.where(~np.isnan(vg))[0], initial=None)
        if first is None: continue
        vg[:first] = vg[first]
    nmask = np.isnan(vg)
    if nmask.any():
        vg[nmask] = np.interp(np.where(nmask)[0], np.where(~nmask)[0], vg[~nmask])

    bw = float(drow["BW"].iloc[0]) if np.isfinite(drow["BW"].iloc[0]) else np.nan
    comed = int(drow["COMED"].iloc[0]) if np.isfinite(drow["COMED"].iloc[0]) else 0
    dose = float(drow["DOSE_MG"].iloc[0]) if np.isfinite(drow["DOSE_MG"].iloc[0]) else 0.0
    dose_mgkg = dose / bw if (np.isfinite(bw) and bw>0) else 0.0

    X_i = np.column_stack([
        times_grid / TAU_H,                       # time in [0,1]
        np.full(T, dose_mgkg, float),             # mg/kg (constant over t for qd)
        np.full(T, comed, int)                    # comed flag
    ])
    seq_X.append(X_i)
    seq_y.append(vg)   # PD in ng/mL (we'll standardize later)
    seq_ids.append(int(sid))
    meta.append((bw, comed))

X = np.asarray(seq_X, float)           # (N, T, F=3)
y = np.asarray(seq_y, float)           # (N, T)
ids = np.asarray(seq_ids, int)
meta = np.asarray(meta, float)         # columns: BW, COMED

print(f"Built sequences: N={len(X)}  T={T}  F={X.shape[2]}")

# -------- Train/val split (robust) & scaling --------
X, y, ids, meta = shuffle(X, y, ids, meta, random_state=SEED)

X_tr, X_va, y_tr, y_va, meta_tr, meta_va = train_test_split(
    X, y, meta, test_size=VAL_SPLIT, random_state=SEED
)

y_scaler = StandardScaler(with_mean=True, with_std=True)
y_tr_z = y_scaler.fit_transform(y_tr.reshape(-1, 1)).reshape(y_tr.shape)
y_va_z = y_scaler.transform(y_va.reshape(-1, 1)).reshape(y_va.shape)

# Optional: scale only the time channel (index 0); leave mg/kg and comed as-is
time_scaler = StandardScaler()
X_tr_scaled = X_tr.copy(); X_va_scaled = X_va.copy()
X_tr_scaled[...,0] = time_scaler.fit_transform(X_tr[...,0])
X_va_scaled[...,0] = time_scaler.transform(X_va[...,0])

# -------- Torch datasets --------
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y[..., None], dtype=torch.float32)  # (N,T,1)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

bs = 8  # small batch for quantum layer stability
train_ds = SeqDataset(X_tr_scaled, y_tr_z)
val_ds   = SeqDataset(X_va_scaled, y_va_z)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)
val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=bs, shuffle=False, drop_last=False)

# -------- QLSTM model --------
input_size = X.shape[2]
model = QLSTMRegressor(input_size=X.shape[2], hidden_size=64, n_qubits=4, n_qlayers=1,
                       backend="default.qubit")
print(f"[Info] QLSTM reports {model.lstm.n_qubits} qubits in use.")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

# LR scheduler: reduce LR when validation loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=12, cooldown=8, min_lr=1e-5)

# Gradient clipping value similar to Keras default practices
clip_value = 1.0

patience = 120  # allow long plateau before stopping
min_delta = 1e-3  # minimal improvement required
best_val = float('inf')
best_state = None
wait = 0
max_epochs = 300  # extended training per request

train_losses, val_losses = [], []
print("Training QLSTM …")
for epoch in range(1, max_epochs+1):
    model.train()
    tl = 0.0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        yhat = model(xb)
        loss = criterion(yhat, yb)
        loss.backward()
        # clip gradients to combat exploding/vanishing through quantum layers
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        tl += loss.item() * xb.size(0)
    tl /= len(train_ds)

    model.eval()
    vl = 0.0
    with torch.no_grad():
        for xb, yb in val_dl:
            yhat = model(xb)
            loss = criterion(yhat, yb)
            vl += loss.item() * xb.size(0)
    vl /= len(val_ds)

    train_losses.append(tl)
    val_losses.append(vl)

    # Step LR scheduler and log when LR changes
    prev_lr = optimizer.param_groups[0]['lr']
    scheduler.step(vl)
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr < prev_lr:
        print(f"LR reduced: {prev_lr:.2e} → {new_lr:.2e}")

    print(f"Epoch {epoch:03d} | train MSE {tl:.5f} | val MSE {vl:.5f}")

    if vl < best_val - min_delta:
        best_val = vl
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# -------- Loss curve --------
plt.figure(figsize=(6,4))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("Epoch"); plt.ylabel("MSE loss"); plt.legend(); plt.tight_layout()
plt.savefig("qlstm_loss_curve.png", dpi=160); plt.show()
print("Saved: qlstm_loss_curve.png")

# -------- Utilities: predict PD(t) in ng/mL for a population at a given dose --------
def make_batch_for_dose(BW, COMED, dose_mg):
    N = len(BW)
    dose_mgkg = dose_mg / np.clip(BW, 1e-6, None)
    time_feat = np.tile(times_grid / TAU_H, (N,1))
    Xb = np.stack([
        time_feat,
        np.tile(dose_mgkg[:,None], (1,T)),
        np.tile(COMED[:,None], (1,T))
    ], axis=-1).astype(float)
    Xb[...,0] = time_scaler.transform(Xb[...,0])
    return Xb

@torch.no_grad()
def predict_pd_ngml(BW, COMED, dose_mg):
    Xb = make_batch_for_dose(BW, COMED, dose_mg)
    xb = torch.tensor(Xb, dtype=torch.float32)
    yhat_z = model(xb).squeeze(-1).cpu().numpy()  # (N,T)
    yhat = y_scaler.inverse_transform(yhat_z.reshape(-1,1)).reshape(yhat_z.shape)
    return yhat

# -------- Base population (from TRAIN+VAL subjects) --------
BW_obs = meta[:,0]
COM_obs = meta[:,1]
BW_obs = BW_obs[np.isfinite(BW_obs)]
if BW_obs.size == 0: BW_obs = np.array([70.0])
p_comed = 0.5 if not np.isfinite(COM_obs).any() else float(np.nanmean(COM_obs))

# -------- Monte Carlo dose scan --------
dose_grid = np.arange(DOSE_STEP, GRID_MAX+DOSE_STEP, DOSE_STEP)
succ_mat = np.zeros((R_MC, len(dose_grid)))
dose90 = np.full(R_MC, np.nan)

print("Step 7/7: Scanning doses (0.5 mg grid) …")
for r in tqdm(range(R_MC), desc="dose grid"):
    N_base = len(BW_obs)
    base_idx = rng.integers(0, N_base, size=N_base*K_EXPAND)
    BW = BW_obs[base_idx] * np.exp(rng.normal(0.0, SIGMA_BW, size=N_base*K_EXPAND))
    BW = np.clip(BW, *BW_CLIP)
    COM = rng.binomial(1, p_comed, size=N_base*K_EXPAND)

    s = []
    for d in dose_grid:
        yhat = predict_pd_ngml(BW, COM, d)  # (N,T)
        ok = (np.nanmax(yhat, axis=1) <= STRICT_THRESHOLD)
        s.append(ok.mean())
    s = np.array(s)
    succ_mat[r,:] = s
    idx90 = np.where(s >= 0.90)[0]
    if idx90.size:
        dose90[r] = float(dose_grid[idx90[0]])

succ_mean = succ_mat.mean(axis=0)
succ_lo   = np.percentile(succ_mat, 2.5, axis=0)
succ_hi   = np.percentile(succ_mat,97.5, axis=0)

dose90_mean = float(np.nanmean(dose90)) if np.isfinite(dose90).any() else np.nan
dose90_ci   = (float(np.nanpercentile(dose90, 2.5)), float(np.nanpercentile(dose90,97.5))) if np.isfinite(dose90).any() else (np.nan, np.nan)

# -------- Plot dose → success --------
plt.figure(figsize=(8,5))
plt.plot(dose_grid, succ_mean, label="QLSTM success fraction")
plt.fill_between(dose_grid, succ_lo, succ_hi, alpha=0.20, label="95% MC band")
plt.axhline(0.90, color="tab:red", ls="--", lw=1.2, label="90% target")
plt.ylim(0,1); plt.xlabel("Daily dose (mg)")
plt.ylabel(f"Success fraction (max PD over 24 h ≤ {STRICT_THRESHOLD} ng/mL)")
plt.grid(True, alpha=0.35); plt.legend(); plt.tight_layout()
plt.savefig("qlstm_dose_success.png", dpi=180); plt.show()

pd.DataFrame({"dose_mg":dose_grid, "succ_mean":succ_mean, "succ_lo2p5":succ_lo, "succ_hi97p5":succ_hi})\
  .to_csv("qlstm_dose_success_summary.csv", index=False)

if np.isfinite(dose90_mean):
    print(f"Minimal dose for 90% success (mean): {dose90_mean:.1f} mg  |  95% CI: [{dose90_ci[0]:.1f}, {dose90_ci[1]:.1f}] mg")
else:
    print("No dose on the grid achieved ≥90% success.")
print("Saved: qlstm_loss_curve.png, qlstm_dose_success.png and qlstm_dose_success_summary.csv")
