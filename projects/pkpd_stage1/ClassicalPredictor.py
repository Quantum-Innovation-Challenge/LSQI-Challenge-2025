import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# =========================
# 1) Normalizers / target transforms
# =========================
class ChannelStandardizer:
    """
    Standardize per-channel across (N,T) after optional log1p.
    For input shaped [N, T, C].
    """
    def __init__(self, log1p: bool = True, eps: float = 1e-8):
        self.log1p = log1p
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        # X: [N,T,C]
        Z = np.log1p(X) if self.log1p else X
        mean = Z.mean(axis=(0,1), keepdims=True)
        std = Z.std(axis=(0,1), keepdims=True) + self.eps
        self.mean = mean
        self.std = std

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = np.log1p(X) if self.log1p else X
        return (Z - self.mean) / self.std

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        X = Z * self.std + self.mean
        return np.expm1(X) if self.log1p else X

class MaskedLogParamTransform:
    """
    For strictly-positive params: target = (log(param+eps) - mean)/std,
    mean/std computed only over mask==1 entries.
    """
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, Y: np.ndarray, M: np.ndarray):
        # Y: [N,D], M: [N,D]
        D = Y.shape[1]
        mean = np.zeros((D,), dtype=float)
        std = np.ones((D,), dtype=float)
        for j in range(D):
            vals = Y[M[:,j] > 0.5, j]
            vals = np.log(vals + self.eps)
            if len(vals) > 10:
                mean[j] = vals.mean()
                std[j] = vals.std() + 1e-8
        self.mean = mean
        self.std = std

    def transform(self, Y: np.ndarray) -> np.ndarray:
        Z = np.log(Y + self.eps)
        Z = (Z - self.mean[None,:]) / self.std[None,:]
        return Z

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        Y = Z * self.std[None,:] + self.mean[None,:]
        return np.exp(Y) - self.eps

# =========================
# 2) Dataset
# =========================
class MultiTaskSeqDataset(Dataset):
    def __init__(self, X_2xT, y_cls, y_reg, y_mask, times_1d,
                 x_norm: ChannelStandardizer,
                 y_norm: MaskedLogParamTransform):
        """
        X_2xT: [N,2,T] as you requested storage format.
        Internally we feed transformer [N,T,2].
        """
        X = np.transpose(X_2xT, (0,2,1))  # [N,T,2]
        X = x_norm.transform(X).astype(np.float32)

        Y = y_norm.transform(y_reg).astype(np.float32)
        self.X = torch.from_numpy(X)
        self.y_cls = torch.from_numpy(y_cls).long()
        self.y_reg = torch.from_numpy(Y)
        self.y_mask = torch.from_numpy(y_mask.astype(np.float32))
        self.times = torch.from_numpy(times_1d.astype(np.float32))  # [T]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y_cls[idx], self.y_reg[idx], self.y_mask[idx], self.times

# =========================
# 3) Time-aware positional encoding (irregular times)
# =========================
class ContinuousSinusoidalPE(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        inv_freq = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, times: torch.Tensor):
        """
        x: [B,T,d_model]
        times: [B,T] or [T] (float)
        """
        if times.dim() == 1:
            times = times[None, :].repeat(x.size(0), 1)  # [B,T]
        # normalize times to roughly [0,1]
        t = times / (times.max(dim=1, keepdim=True).values + 1e-8)
        sinus_inp = t[..., None] * self.inv_freq[None, None, :]  # [B,T,d_model/2]
        pe = torch.zeros_like(x)
        pe[..., 0::2] = torch.sin(sinus_inp)
        pe[..., 1::2] = torch.cos(sinus_inp)
        return self.dropout(x + pe)

# =========================
# 4) Multitask Transformer with [CLS] token pooling
# =========================
class MultiTaskTransformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 reg_dim: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.posenc = ContinuousSinusoidalPE(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, reg_dim)
        )

    def forward(self, x: torch.Tensor, times: torch.Tensor):
        """
        x: [B,T,input_dim]
        times: [B,T] or [T]
        """
        h = self.input_proj(x)  # [B,T,d]
        B = h.size(0)
        cls = self.cls_token.expand(B, -1, -1)  # [B,1,d]
        h = torch.cat([cls, h], dim=1)  # [B,T+1,d]

        # prepend a dummy time for CLS (0)
        if times.dim() == 1:
            times_b = times[None, :].repeat(B, 1)
        else:
            times_b = times
        cls_t = torch.zeros((B,1), device=times_b.device, dtype=times_b.dtype)
        times_ext = torch.cat([cls_t, times_b], dim=1)  # [B,T+1]

        h = self.posenc(h, times_ext)
        h = self.encoder(h)
        h0 = self.norm(h[:,0,:])  # CLS pooled

        logits = self.cls_head(h0)
        reg = self.reg_head(h0)
        return logits, reg

# =========================
# 5) Loss: masked regression (only relevant params)
# =========================
def masked_huber(pred, target, mask, delta=1.0):
    # pred/target/mask: [B,D]
    err = pred - target
    abs_err = err.abs()
    quad = torch.minimum(abs_err, torch.tensor(delta, device=pred.device))
    lin = abs_err - quad
    loss = 0.5 * quad**2 + delta * lin
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom

# =========================
# 6) Train / eval loop (AdamW + Cosine + AMP + grad clip)
# =========================
def run_train(model, train_loader, val_loader, device,
              epochs=50, lr=3e-4, weight_decay=1e-2,
              lambda_reg=1.0, max_grad_norm=1.0):
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        tr_n = 0

        for x, y_cls, y_reg, y_mask, times in train_loader:
            x = x.to(device)
            y_cls = y_cls.to(device)
            y_reg = y_reg.to(device)
            y_mask = y_mask.to(device)
            times = times.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, reg = model(x, times)
                loss_cls = ce(logits, y_cls)
                loss_reg = masked_huber(reg, y_reg, y_mask, delta=1.0)
                loss = loss_cls + lambda_reg * loss_reg

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()

            tr_loss += loss.item() * x.size(0)
            tr_n += x.size(0)

        sched.step()

        # validation
        model.eval()
        va_loss = 0.0
        va_n = 0
        correct = 0
        with torch.no_grad():
            for x, y_cls, y_reg, y_mask, times in val_loader:
                x = x.to(device)
                y_cls = y_cls.to(device)
                y_reg = y_reg.to(device)
                y_mask = y_mask.to(device)
                times = times.to(device)

                logits, reg = model(x, times)
                loss_cls = ce(logits, y_cls)
                loss_reg = masked_huber(reg, y_reg, y_mask, delta=1.0)
                loss = loss_cls + lambda_reg * loss_reg

                va_loss += loss.item() * x.size(0)
                va_n += x.size(0)
                pred = logits.argmax(dim=-1)
                correct += (pred == y_cls).sum().item()

        tr_loss /= max(tr_n, 1)
        va_loss /= max(va_n, 1)
        acc = correct / max(va_n, 1)
        print(f"Epoch {ep:03d} | train {tr_loss:.4f} | val {va_loss:.4f} | val acc {acc:.3f} | lr {sched.get_last_lr()[0]:.2e}")

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

@torch.no_grad()
def inference(model, x_seq_T2, times_1d, device,
              x_norm: ChannelStandardizer,
              y_norm: MaskedLogParamTransform):
    """
    x_seq_T2: numpy [T,2] (dose+conc for PK or conc+biom for PD)
    returns: class_probs, reg_params_in_original_space
    """
    model.eval()
    X = x_seq_T2[None, :, :]  # [1,T,2]
    X = x_norm.transform(X).astype(np.float32)
    x = torch.from_numpy(X).to(device)
    times = torch.from_numpy(times_1d.astype(np.float32)).to(device)

    logits, reg = model(x, times)
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    reg_z = reg[0].cpu().numpy()[None, :]
    reg_params = y_norm.inverse_transform(reg_z)[0]
    return probs, reg_params

# =========================
# 7) Example: generate data and train PK and PD models
# =========================
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    data = generate_pk_pd_datasets(N=5000, seed=123)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- PK ----
    Xpk = data["X_pk_2xT"]
    ypk_cls = data["y_pk_cls"]
    ypk_reg = data["y_pk_reg"]
    ypk_mask = data["y_pk_mask"]
    pk_times = data["pk_times"]

    # build normalizers using train split only
    idx = np.arange(len(Xpk))
    idx_tr, idx_va = train_test_split(idx, test_size=0.1, random_state=0, stratify=ypk_cls)

    x_norm_pk = ChannelStandardizer(log1p=True)
    # fit on [N,T,2]
    x_norm_pk.fit(np.transpose(Xpk[idx_tr], (0,2,1)))

    y_norm_pk = MaskedLogParamTransform()
    y_norm_pk.fit(ypk_reg[idx_tr], ypk_mask[idx_tr])

    ds_pk = MultiTaskSeqDataset(Xpk, ypk_cls, ypk_reg, ypk_mask, pk_times, x_norm_pk, y_norm_pk)
    ds_pk_tr = torch.utils.data.Subset(ds_pk, idx_tr.tolist())
    ds_pk_va = torch.utils.data.Subset(ds_pk, idx_va.tolist())

    dl_pk_tr = DataLoader(ds_pk_tr, batch_size=64, shuffle=True, drop_last=False)
    dl_pk_va = DataLoader(ds_pk_va, batch_size=256, shuffle=False, drop_last=False)

    pk_model = MultiTaskTransformer(input_dim=2, num_classes=10, reg_dim=PK_PARAM_DIM,
                                   d_model=64, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1)
    pk_model = run_train(pk_model, dl_pk_tr, dl_pk_va, device, epochs=40, lr=3e-4, lambda_reg=1.0)

    # ---- PD ----
    Xpd = data["X_pd_2xT"]
    ypd_cls = data["y_pd_cls"]
    ypd_reg = data["y_pd_reg"]
    ypd_mask = data["y_pd_mask"]
    pd_times = data["pd_times"]

    idx = np.arange(len(Xpd))
    idx_tr, idx_va = train_test_split(idx, test_size=0.1, random_state=0, stratify=ypd_cls)

    x_norm_pd = ChannelStandardizer(log1p=True)
    x_norm_pd.fit(np.transpose(Xpd[idx_tr], (0,2,1)))

    y_norm_pd = MaskedLogParamTransform()
    y_norm_pd.fit(ypd_reg[idx_tr], ypd_mask[idx_tr])

    ds_pd = MultiTaskSeqDataset(Xpd, ypd_cls, ypd_reg, ypd_mask, pd_times, x_norm_pd, y_norm_pd)
    ds_pd_tr = torch.utils.data.Subset(ds_pd, idx_tr.tolist())
    ds_pd_va = torch.utils.data.Subset(ds_pd, idx_va.tolist())

    dl_pd_tr = DataLoader(ds_pd_tr, batch_size=64, shuffle=True, drop_last=False)
    dl_pd_va = DataLoader(ds_pd_va, batch_size=256, shuffle=False, drop_last=False)

    pd_model = MultiTaskTransformer(input_dim=2, num_classes=10, reg_dim=PD_PARAM_DIM,
                                   d_model=64, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1)
    pd_model = run_train(pd_model, dl_pd_tr, dl_pd_va, device, epochs=40, lr=3e-4, lambda_reg=1.0)

    # ---- Inference demo: one PK trajectory ----
    sample_pk = Xpk[0]               # [2,39]
    x_seq = sample_pk.T              # [39,2] for inference helper
    probs, params = inference(pk_model, x_seq, pk_times, device, x_norm_pk, y_norm_pk)
    print("PK top-3 probs:", probs.argsort()[-3:][::-1], probs[probs.argsort()[-3:][::-1]])
    print("PK param vec:", params)

    # ---- Inference demo: one PD trajectory ----
    sample_pd = Xpd[0]               # [2,25]
    x_seq = sample_pd.T              # [25,2]
    probs, params = inference(pd_model, x_seq, pd_times, device, x_norm_pd, y_norm_pd)
    print("PD top-3 probs:", probs.argsort()[-3:][::-1], probs[probs.argsort()[-3:][::-1]])
    print("PD param vec:", params)
