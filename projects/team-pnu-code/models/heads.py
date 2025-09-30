"""
Heads Models
"""

import math
from typing import Any, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Utils
# =========================

def _apply_mask(
    y: torch.Tensor, pred: torch.Tensor, m: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if m is None:
        return y.view(-1), pred.view(-1), y.numel()
    w = (m > 0).float()
    if y.dim() != w.dim():
        if y.dim() == 1 and w.dim() == 2:
            y = y.unsqueeze(1).expand_as(w)
            pred = pred.unsqueeze(1).expand_as(w)
        elif y.dim() == 2 and w.dim() == 1:
            w = w.unsqueeze(1).expand_as(y)
    y = y * w
    pred = pred * w
    denom = float(w.sum().item()) if w.sum().item() > 0 else float(y.numel())
    return y.view(-1), pred.view(-1), denom


def _reg_metrics(
    pred: torch.Tensor, y: torch.Tensor, m: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    if m is None:
        # Simple regression metrics
        diff = pred.view(-1) - y.view(-1)
        mse = torch.mean(diff * diff)
        mae = torch.mean(torch.abs(diff))
        rmse = torch.sqrt(mse)
        
        # R2 calculation
        # y_mean = torch.mean(y.view(-1))
        y_mean = torch.mean(y.float())

        ss_res = torch.sum((y.view(-1) - pred.view(-1)) ** 2)
        ss_tot = torch.sum((y.view(-1) - y_mean) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
        
        return {
            "mse": float(mse.item()),
            "rmse": float(rmse.item()),
            "mae": float(mae.item()),
            "r2": float(r2.item()),
        }
    
    yv, pv, denom = _apply_mask(y, pred, m)
    diff = pv - yv
    mse = torch.sum(diff * diff) / max(denom, 1e-8)
    mae = torch.sum(torch.abs(diff)) / max(denom, 1e-8)
    ymean = (yv.sum() / max(denom, 1e-8))
    ss_res = torch.sum((yv - pv) ** 2)
    ss_tot = torch.sum((yv - ymean) ** 2) + 1e-12
    r2 = 1.0 - (ss_res / ss_tot)
    return {
        "mse": float(mse.item()),
        "rmse": float(torch.sqrt(mse).item()),
        "mae": float(mae.item()),
        "r2": float(r2.item()),
    }

# =========================
# Base Head
# =========================
class BaseHead(nn.Module):
    def forward(self, z: torch.Tensor, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def loss(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

# =========================
# Binary Classification Head
# =========================
class BinaryClassificationHead(BaseHead):
    def __init__(self, in_dim: int, dropout: float = 0.0):
        super().__init__()
        self.mean = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, 2)
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.mean(z)
        return {"pred": logits}
    
    def loss(self, outputs, batch):
        y = batch["y"].long().view(-1)    # [B]
        logits = outputs["pred"]          # [B, 2]
        loss = F.cross_entropy(logits, y)
        return loss, {"loss": loss.item()}

    def metrics(self, outputs, batch):
        y = batch["y"].long().view(-1)    # [B]
        logits = outputs["pred"]          # [B, 2]
        acc = (logits.argmax(dim=-1) == y).float().mean().item()
        return {"accuracy": acc}



# =========================
# MSE Head
# =========================

class MSEHead(BaseHead):
    def __init__(self, in_dim: int, dropout: float = 0.0):
        super().__init__()
        self.mean = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu = self.mean(z).squeeze(-1)
        return {"pred": mu}

    def loss(self, outputs, batch):
        y = batch["y"].float().view(-1)
        pred = outputs["pred"].view(-1)
        m = batch.get("m", None)
        if m is None:
            loss = F.mse_loss(pred, y)
        else:
            yv, pv, denom = _apply_mask(y, pred, m)
            loss = torch.sum((pv - yv) ** 2) / max(denom, 1e-8)
        with torch.no_grad():
            metrics = _reg_metrics(pred, y, m)
        return loss, metrics

# =========================
# Gaussian NLL Head
# =========================

class GaussianNLLHead(BaseHead):
    def __init__(self, in_dim: int, min_logvar: float = -10.0, max_logvar: float = 4.0, tied_variance: bool = False):
        super().__init__()
        self.mu = nn.Linear(in_dim, 1)
        if tied_variance:
            self.logvar = nn.Parameter(torch.tensor(0.0))
            self.tied = True
        else:
            self.logvar_head = nn.Linear(in_dim, 1)
            self.tied = False
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.criterion = torch.nn.GaussianNLLLoss(full=True, reduction="mean")

    def forward(self, z: torch.Tensor, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        mu = self.mu(z).squeeze(-1)
        if self.tied:
            logvar = self.logvar.expand_as(mu)
        else:
            logvar = self.logvar_head(z).squeeze(-1)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        var = torch.exp(logvar)
        return {"pred": mu, "var": var, "logvar": logvar}

    def loss(self, outputs, batch):
        y = batch["y"].float().view(-1)
        mu = outputs["pred"].view(-1)
        var = outputs["var"].view(-1).clamp_min(1e-8)
        m = batch.get("m", None)
        if m is None:
            loss = self.criterion(mu, y, var)
        else:
            yv, pv, _ = _apply_mask(y, mu, m)
            vv, _, _ = _apply_mask(y, var, m)
            loss = self.criterion(pv, yv, vv)
        with torch.no_grad():
            metrics = {
                "nll": float(loss.item()),
                **_reg_metrics(mu, y, m),
                "avg_sigma": float(torch.sqrt(var).mean().item()),
            }
        return loss, metrics

# =========================
# Poisson Head
# =========================

class PoissonHead(BaseHead):
    def __init__(self, in_dim: int, eps: float = 1e-6):
        super().__init__()
        self.log_lambda = nn.Linear(in_dim, 1)
        self.eps = eps
        self.criterion = torch.nn.PoissonNLLLoss(log_input=True, full=True, reduction="mean")

    def forward(self, z: torch.Tensor, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        loglam = self.log_lambda(z).squeeze(-1)
        loglam = torch.clamp(loglam, min=math.log(self.eps))
        lam = torch.exp(loglam)
        return {"pred": lam, "log_lambda": loglam}

    def loss(self, outputs, batch):
        y = batch["y"].float().clamp_min(0.0)
        loglam = outputs["log_lambda"]
        m = batch.get("m", None)
        loss = self.criterion(loglam, y)
        with torch.no_grad():
            lam = torch.exp(loglam)
            metrics = {"nll": float(loss.item()), **_reg_metrics(lam, y, m)}
        return loss, metrics

# =========================
# Emax Head
# =========================

class EmaxHead(BaseHead):
    def __init__(self, in_dim: int, conditional: bool = True,
                 use_proj_C: bool = True, c_key: Any = ("C","PK_PRED","pk_pred"),
                 min_n: float = 0.5, max_n: float = 6.0):
        super().__init__()
        self.c_key = c_key if isinstance(c_key, (list,tuple)) else (c_key,)
        self.use_proj_C = use_proj_C
        if use_proj_C:
            self.c_proj = nn.Sequential(nn.Linear(in_dim, 1), nn.Softplus())

        if conditional:
            self.param_head = nn.Sequential(
                nn.Linear(in_dim, max(64, in_dim//2)), nn.ReLU(),
                nn.Linear(max(64, in_dim//2), 4)
            )
            self.conditional = True
            self.register_buffer("min_n_buf", torch.tensor(min_n))
            self.register_buffer("max_n_buf", torch.tensor(max_n))
        else:
            self.conditional = False
            self.E0 = nn.Parameter(torch.tensor(0.0))
            self.Emax = nn.Parameter(torch.tensor(1.0))
            self.logEC50 = nn.Parameter(torch.tensor(0.0))
            self.n_param = nn.Parameter(torch.tensor(1.0))
            self.min_n = min_n
            self.max_n = max_n

    def _get_params(self, z: torch.Tensor):
        if self.conditional:
            p = self.param_head(z)
            E0      = p[..., 0]
            Emax    = F.softplus(p[..., 1])
            logEC50 = p[..., 2]
            n_norm  = torch.sigmoid(p[..., 3])
            n       = self.min_n_buf + (self.max_n_buf - self.min_n_buf) * n_norm
            EC50    = torch.exp(logEC50)
        else:
            E0   = self.E0.expand(z.size(0))
            Emax = F.softplus(self.Emax).expand(z.size(0))
            EC50 = torch.exp(self.logEC50).expand(z.size(0))
            n    = torch.clamp(self.n_param, self.min_n, self.max_n).expand(z.size(0))
        return E0, Emax, EC50, n

    def _pick_C(self, z, batch):
        for name in self.c_key:
            if isinstance(batch, dict) and name in batch and torch.is_tensor(batch[name]):
                C = batch[name].float()
                if C.dim()==3: C = C.mean(dim=1)
                if C.dim()==2 and C.size(-1)==1: C = C.squeeze(-1)
                return C
        if self.use_proj_C:
            return self.c_proj(z).squeeze(-1).clamp_min(1e-8)
        raise KeyError(f"EmaxHead needs one of {self.c_key} or use_proj_C=True.")

    def forward(self, z, batch):
        C = self._pick_C(z, batch)
        E0, Emax, EC50, n = self._get_params(z)
        Cn = torch.pow(torch.clamp(C, min=1e-8), n)
        ECn = torch.pow(torch.clamp(EC50, min=1e-8), n)
        pred = E0 + Emax * (Cn / (ECn + Cn))
        return {"pred": pred, "E0": E0, "Emax": Emax, "EC50": EC50, "n": n, "C": C}

    def loss(self, outputs, batch):
        y = batch["y"].float()
        pred = outputs["pred"]
        m = batch.get("m", None)
        if m is None:
            loss = F.mse_loss(pred, y)
        else:
            yv, pv, denom = _apply_mask(y, pred, m)
            loss = torch.sum((pv - yv) ** 2) / max(denom, 1e-8)
        with torch.no_grad():
            metrics = {**_reg_metrics(pred, y, m),
                       "E0_mean": float(outputs["E0"].mean().item()),
                       "Emax_mean": float(outputs["Emax"].mean().item()),
                       "EC50_mean": float(outputs["EC50"].mean().item()),
                       "n_mean": float(outputs["n"].mean().item())}
        return loss, metrics


class EmaxGaussianHead(BaseHead):
    def __init__(self, in_dim, min_logvar=-10, max_logvar=4):
        super().__init__()
        # Emax parameters head
        self.param_head = nn.Sequential(
            nn.Linear(in_dim, max(64, in_dim // 2)), nn.ReLU(),
            nn.Linear(max(64, in_dim // 2), 4)  # E0, Emax, logEC50, n
        )
        # variance head
        self.logvar_head = nn.Linear(in_dim, 1)
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.criterion = torch.nn.GaussianNLLLoss(full=True, reduction="mean")

    def forward(self, z, batch):
        p = self.param_head(z)
        E0, Emax, logEC50, n_norm = p[...,0], F.softplus(p[...,1]), p[...,2], torch.sigmoid(p[...,3])
        n = 0.5 + (6.0 - 0.5) * n_norm
        EC50 = torch.exp(logEC50)
        C = batch["C"].float().clamp_min(1e-8)
        pred = E0 + Emax * (C**n) / (EC50**n + C**n)

        logvar = self.logvar_head(z).squeeze(-1).clamp(self.min_logvar, self.max_logvar)
        var = torch.exp(logvar)

        return {"pred": pred, "var": var, "E0": E0, "Emax": Emax, "EC50": EC50, "n": n}

    def loss(self, outputs, batch):
        y = batch["y"].float().view(-1)
        mu = outputs["pred"].view(-1)
        var = outputs["var"].view(-1).clamp_min(1e-8)
        loss = self.criterion(mu, y, var)  # Gaussian NLL 기반
        with torch.no_grad():
            metrics = {
                "nll": float(loss.item()),
                "mse": F.mse_loss(mu, y).item(),
                "avg_sigma": torch.sqrt(var).mean().item(),
            }
        return loss, metrics
