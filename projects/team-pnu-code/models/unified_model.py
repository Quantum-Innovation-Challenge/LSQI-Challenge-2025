"""
Unified PK/PD Model
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any
from utils.factory import build_encoder, build_head
from .encoders import BaseEncoder
from .heads import BaseHead


class UnifiedPKPDModel(nn.Module):
    def __init__(
        self,
        config,
        pk_features: List[str],
        pd_features: List[str],
        pk_input_dim: int,
        pd_input_dim: int
    ):
        super().__init__()
        self.config = config
        self.mode = config.mode
        self.pk_features = pk_features
        self.pd_features = pd_features

        if self.mode in ["independent", "separate"]:
            self._build_independent_models(pk_input_dim, pd_input_dim)
        elif self.mode in ["cascade", "dual_stage"]:
            self._build_cascade_model(pk_input_dim, pd_input_dim)
        elif self.mode in ["multi-task", "shared"]:
            self._build_multitask_model(pk_input_dim, pd_input_dim)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Classification head
        if self.mode in ["multi-task", "shared"]:
            self.head_clf = self._create_head("binary_classification", self.encoder.out_dim)
        else:
            self.head_clf = self._create_head("binary_classification", self.pd_encoder.out_dim)

    # ==============================
    # Build methods
    # ==============================
    def _build_independent_models(self, pk_input_dim: int, pd_input_dim: int):
        pk_encoder_type = self.config.encoder_pk or self.config.encoder
        self.pk_encoder = self._create_encoder(pk_encoder_type, pk_input_dim)
        self.pk_head = self._create_head(self.config.head_pk, self.pk_encoder.out_dim)

        pd_encoder_type = self.config.encoder_pd or self.config.encoder
        self.pd_encoder = self._create_encoder(pd_encoder_type, pd_input_dim + 1)  # +1 for PK pred
        self.pd_head = self._create_head(self.config.head_pd, self.pd_encoder.out_dim)

        self.pk_encoder_type, self.pd_encoder_type = pk_encoder_type, pd_encoder_type

    def _build_cascade_model(self, pk_input_dim: int, pd_input_dim: int):
        pk_encoder_type = self.config.encoder_pk or self.config.encoder
        self.pk_encoder = self._create_encoder(pk_encoder_type, pk_input_dim)
        self.pk_head = self._create_head(self.config.head_pk, self.pk_encoder.out_dim)

        pd_encoder_type = self.config.encoder_pd or self.config.encoder
        if self.mode in ["cascade", "dual_stage"]:
            pd_input_dim_with_pk = pd_input_dim + 1

        self.pd_encoder = self._create_encoder(pd_encoder_type, pd_input_dim_with_pk)
        self.pd_head = self._create_head(self.config.head_pd, self.pd_encoder.out_dim)

        self.pk_encoder_type, self.pd_encoder_type = pk_encoder_type, pd_encoder_type


    def _build_multitask_model(self, pk_input_dim: int, pd_input_dim: int):
        shared_input_dim = max(pk_input_dim, pd_input_dim)
        self.encoder = self._create_encoder(self.config.encoder, shared_input_dim)
        self.pk_head = self._create_head(self.config.head_pk, self.encoder.out_dim)
        self.pd_head = self._create_head(self.config.head_pd, self.encoder.out_dim)

        if pk_input_dim != shared_input_dim:
            self.pk_proj = nn.Linear(pk_input_dim, shared_input_dim)
        if pd_input_dim != shared_input_dim:
            self.pd_proj = nn.Linear(pd_input_dim, shared_input_dim)


    def _create_encoder(self, encoder_type: str, input_dim: int) -> BaseEncoder:
        return build_encoder(encoder_type, input_dim, self.config)

    def _create_head(self, head_type: str, input_dim: int) -> BaseHead:
        return build_head(head_type, input_dim, self.config)

    # ==============================
    # Forward
    # ==============================
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode in ["independent", "separate"]:
            return self._forward_independent(batch)
        elif self.mode in ["cascade", "dual_stage"]:
            return self._forward_cascade(batch)
        elif self.mode in ["multi-task", "shared"]:
            return self._forward_multitask(batch)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ==============================
    # Forward helpers
    # ==============================
    def _unwrap(self, output):
        return output[0] if isinstance(output, tuple) else output

    def _prepare_pd_input(self, x_pd, pk_tensor, expected_dim=1):
        """Handle PK→PD connection (pad/truncate, unsqueeze if needed)"""
        if pk_tensor is None:
            return torch.cat([x_pd, torch.zeros(x_pd.size(0), expected_dim, device=x_pd.device)], dim=-1)

        if pk_tensor.dim() == 1:
            pk_tensor = pk_tensor.unsqueeze(-1)

        if pk_tensor.size(0) != x_pd.size(0):
            if pk_tensor.size(0) < x_pd.size(0):
                padding = torch.zeros(x_pd.size(0) - pk_tensor.size(0), pk_tensor.size(1), device=x_pd.device)
                pk_tensor = torch.cat([pk_tensor, padding], dim=0)
            else:
                pk_tensor = pk_tensor[:x_pd.size(0)]

        return torch.cat([x_pd, pk_tensor], dim=-1)

    # ==============================
    # Mode-specific forward
    # ==============================
    def _forward_independent(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        if 'pk' in batch:
            z_pk = self._unwrap(self.pk_encoder(batch['pk']['x']))
            pk_outs = self.pk_head(z_pk)
            results['pk'] = {'pred': pk_outs['pred'], 'z': z_pk, 'outs': pk_outs}

        if 'pd' in batch:
            pk_pred = results['pk']['pred'] if 'pk' in results else None
            x_pd_with_pk = self._prepare_pd_input(batch['pd']['x'], pk_pred)
            z_pd = self._unwrap(self.pd_encoder(x_pd_with_pk))
            pd_outs = self.pd_head(z_pd)
            results['pd'] = {'pred': pd_outs['pred'], 'z': z_pd, 'outs': pd_outs}

        return results

    def _forward_cascade(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        if 'pk' in batch:
            z_pk = self._unwrap(self.pk_encoder(batch['pk']['x']))
            pk_outs = self.pk_head(z_pk)
            results['pk'] = {'pred': pk_outs['pred'], 'z': z_pk, 'outs': pk_outs}

        if 'pd' in batch:
            pk_pred = results['pk']['pred'].detach() if 'pk' in results else None
            x_pd_with_pk = self._prepare_pd_input(batch['pd']['x'], pk_pred)
            z_pd = self._unwrap(self.pd_encoder(x_pd_with_pk))
            pd_outs = self.pd_head(z_pd)
            results['pd'] = {'pred': pd_outs['pred'], 'z': z_pd, 'outs': pd_outs}

        return results

    def _forward_multitask(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        if 'pk' in batch:
            x_pk = self.pk_proj(batch['pk']['x']) if hasattr(self, 'pk_proj') else batch['pk']['x']
            z = self._unwrap(self.encoder(x_pk))
            pk_outs = self.pk_head(z)
            results['pk'] = {'pred': pk_outs['pred'], 'z': z, 'outs': pk_outs}

        if 'pd' in batch:
            x_pd = self.pd_proj(batch['pd']['x']) if hasattr(self, 'pd_proj') else batch['pd']['x']
            z = self._unwrap(self.encoder(x_pd))
            pd_outs = self.pd_head(z)
            results['pd'] = {'pred': pd_outs['pred'], 'z': z, 'outs': pd_outs}

        return results


    # ==============================
    # Extra methods
    # ==============================
    def get_model_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        info = {
            'mode': self.mode,
            'total_parameters': total_params,
            'pk_features': len(self.pk_features),
            'pd_features': len(self.pd_features)
        }
        if hasattr(self, 'pk_encoder_type') and hasattr(self, 'pd_encoder_type'):
            info['pk_encoder'] = self.pk_encoder_type
            info['pd_encoder'] = self.pd_encoder_type
        else:
            info['encoder'] = self.config.encoder
        return info

    def predict_with_uncertainty(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if not getattr(self.config, "use_mc_dropout", False):
            raise ValueError("Set use_mc_dropout=True in config.")
        n_samples = getattr(self.config, "mc_samples", 50)

        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        results = {}
        with torch.no_grad():
            for branch in ['pk', 'pd']:
                if branch not in batch:
                    continue
                preds = []
                for _ in range(n_samples):
                    out = self.forward(batch if branch == 'pd' else {'pk': batch['pk']})
                    preds.append(out[branch]['pred'])
                preds = torch.stack(preds, dim=0)
                results[branch] = {
                    'pred': preds.mean(dim=0),
                    'std': preds.std(dim=0),
                    'confidence_interval': 1.96 * preds.std(dim=0),
                    'all_predictions': preds
                }
        return results

    def _forward_clf(self, z: torch.Tensor) -> Dict[str, Any]:
        clf_outs = self.head_clf(z)
        return {
            'pred_clf': clf_outs['pred'],
            'z': z,
            'outs': clf_outs
        }