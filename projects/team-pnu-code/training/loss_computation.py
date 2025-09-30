"""
Loss computation methods for PK/PD training
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any


class LossComputation:
    """Handle all loss computation logic"""
    
    def __init__(self, config, data_augmentation, contrastive_learning):
        self.config = config
        self.data_augmentation = data_augmentation
        self.contrastive_learning = contrastive_learning
        
    def _regression_loss(self, pred, target, loss_type: str = "mse"):
        if loss_type == "mse":
            return F.mse_loss(pred, target)
        elif loss_type == "mae":
            return F.l1_loss(pred, target)
        elif loss_type == "asymmetric":
            diff = pred - target
            loss = torch.where(
	            diff > 0,
	            self.config.over_weight * diff**2,
	            self.config.under_weight * diff**2
	        )
            return loss.mean()
        elif loss_type == "quantile":
            q = getattr(self.config, "quantile_q", 0.3)
            diff = target - pred
            return torch.max(q * diff, (q - 1) * diff).mean()
        elif loss_type == "one_sided":
            diff = pred - target
            return torch.mean(torch.clamp(diff, min=0)**2)
        elif loss_type == "hybrid":
            # Hybrid = λ * MSE + (1-λ) * Quantile
            q = getattr(self.config, "quantile_q", 0.3)
            λ = getattr(self.config, "hybrid_lambda", 0.5)
            mse_loss = F.mse_loss(pred, target)
            diff = target - pred
            quantile_loss = torch.max(q * diff, (q - 1) * diff).mean()
            return λ * mse_loss + (1 - λ) * quantile_loss
        else:
            raise ValueError(f"Unknown regression loss type: {loss_type}")
    
    def compute_loss(self, model, batch_pk, batch_pd, mode: str, is_training: bool = True):
        if mode in ["independent", "separate"]:
            return self._compute_independent_loss(model, batch_pk, batch_pd, is_training)
        elif mode in ["cascade", "dual_stage"]:
            return self._compute_cascade_loss(model, batch_pk, batch_pd, is_training)
        elif mode in ["multi-task", "shared"]:
            return self._compute_multitask_loss(model, batch_pk, batch_pd, is_training)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute_independent_loss(self, model, batch_pk, batch_pd, is_training=True):
        losses, preds, targets = {'pk': torch.tensor(0.0), 'pd': torch.tensor(0.0), 'pd_clf': torch.tensor(0.0)}, {}, {}
        pk_preds, pd_preds, pd_pred_clfs = [], [], []
        pk_targets, pd_targets, pd_targets_clfs = [], [], []

        # PK loss
        if batch_pk is not None:
            batch_pk_dict = self._apply_augmentation(
                self._prepare_batch(batch_pk), is_training=is_training
            )
            pk_results = model({'pk': {'x': batch_pk_dict['x'], 'y': batch_pk_dict['y']}})
            pk_pred = pk_results['pk']['pred']
            pk_target = batch_pk_dict['y'].squeeze(-1)
            pk_loss = self._regression_loss(pk_pred, pk_target, getattr(self.config, "loss_type_pk", "mse"))
            pk_preds.append(pk_pred)
            pk_targets.append(pk_target)

            if self.config.use_aug_supervised and 'x_aug' in batch_pk_dict:
                pk_results_aug = model({'pk': {'x': batch_pk_dict['x_aug'], 'y': batch_pk_dict['y_aug']}})
                pk_pred_aug = pk_results_aug['pk']['pred']
                pk_target_aug = batch_pk_dict['y_aug'].squeeze(-1)
                pk_loss_aug = self._regression_loss(pk_pred_aug, pk_target_aug, getattr(self.config, "loss_type_pk", "mse"))
                pk_loss = pk_loss + getattr(self.config, 'aug_lambda', 0.5) * pk_loss_aug
            losses['pk'] = pk_loss

        # PD loss
        if batch_pd is not None:
            batch_pd_dict = self._apply_augmentation(self._prepare_batch(batch_pd), is_training=is_training)
            pd_results = model({'pd': {'x': batch_pd_dict['x'], 'y': batch_pd_dict['y']}})
            pd_pred = pd_results['pd']['pred']
            pd_target = batch_pd_dict['y'].squeeze(-1)

            pd_loss = self._regression_loss(pd_pred, pd_target, getattr(self.config, "loss_type_pd", "hybrid"))
            pd_preds.append(pd_pred)
            pd_targets.append(pd_target)

            if self.config.use_aug_supervised and 'x_aug' in batch_pd_dict:
                pd_results_aug = model({'pd': {'x': batch_pd_dict['x_aug'], 'y': batch_pd_dict['y_aug']}})
                pd_pred_aug = pd_results_aug['pd']['pred']
                pd_target_aug = batch_pd_dict['y_aug'].squeeze(-1)
                pd_loss_aug = self._regression_loss(pd_pred_aug, pd_target_aug, getattr(self.config, "loss_type_pd", "hybrid"))
                pd_loss = pd_loss + getattr(self.config, 'aug_lambda', 0.5) * pd_loss_aug
            losses['pd'] = pd_loss

            if getattr(self.config, 'use_clf', False):
                pd_loss_clf, pd_pred_clf, pd_target_clf = self._compute_clf_loss(
                    model, pd_results['pd']['z'], batch_pd_dict
                )
                pd_pred_clfs.append(pd_pred_clf)
                pd_targets_clfs.append(pd_target_clf)
                losses['pd_clf'] = pd_loss_clf

        total_loss = losses['pk'] + losses['pd'] + 0.5 * losses['pd_clf']
        losses['total'] = total_loss

        preds['pk'], preds['pd'], preds['pd_clf'] = pk_preds, pd_preds, pd_pred_clfs
        targets['pk'], targets['pd'], targets['pd_clf'] = pk_targets, pd_targets, pd_targets_clfs
        return losses, preds, targets
        
    def _compute_cascade_loss(self, model, batch_pk, batch_pd, is_training=True):
        losses, preds, targets = {}, {}, {}
        pk_preds, pd_preds, pd_pred_clfs = [], [], []
        pk_targets, pd_targets, pd_targets_clfs = [], [], []
        batch_dict, total_loss = {}, 0.0

        if batch_pk is not None:
            batch_dict['pk'] = self._apply_augmentation(self._prepare_batch(batch_pk), is_training=is_training)
        if batch_pd is not None:
            batch_dict['pd'] = self._apply_augmentation(self._prepare_batch(batch_pd), is_training=is_training)

        # PK loss
        if 'pk' in batch_dict:
            results = model({'pk': {'x': batch_dict['pk']['x'], 'y': batch_dict['pk']['y']}})
            pk_pred = results['pk']['pred']
            pk_target = batch_dict['pk']['y'].squeeze(-1)
            pk_loss = self._regression_loss(pk_pred, pk_target, getattr(self.config, "loss_type_pk", "mse"))
            pk_preds.append(pk_pred)
            pk_targets.append(pk_target)
            
            if self.config.use_aug_supervised:
                pk_results_aug = model({'pk': {'x': batch_dict['pk']['x_aug'], 'y': batch_dict['pk']['y_aug']}})
                pk_pred_aug = pk_results_aug['pk']['pred']
                pk_target_aug = batch_dict['pk']['y_aug'].squeeze(-1)
                pk_loss_aug = self._regression_loss(pk_pred_aug, pk_target_aug, getattr(self.config, "loss_type_pk", "mse"))
                pk_loss = pk_loss + getattr(self.config, 'aug_lambda', 0.5) * pk_loss_aug
            losses['pk'] = pk_loss
            total_loss = total_loss + pk_loss


        # PD loss
        if 'pd' in batch_dict:
            pd_results = model({'pd': {'x': batch_dict['pd']['x'], 'y': batch_dict['pd']['y']}})
            pd_pred = pd_results['pd']['pred']
            pd_target = batch_dict['pd']['y'].squeeze(-1)

            pd_loss = self._regression_loss(pd_pred, pd_target, getattr(self.config, "loss_type_pd", "hybrid"))
            pd_preds.append(pd_pred)
            pd_targets.append(pd_target)

            if self.config.use_aug_supervised:
                pd_results_aug = model({'pd': {'x': batch_dict['pd']['x_aug'], 'y': batch_dict['pd']['y_aug']}})
                pd_pred_aug = pd_results_aug['pd']['pred']
                pd_target_aug = batch_dict['pd']['y_aug'].squeeze(-1)
                pd_loss_aug = self._regression_loss(pd_pred_aug, pd_target_aug, getattr(self.config, "loss_type_pd", "hybrid"))
                pd_loss = pd_loss + getattr(self.config, 'aug_lambda', 0.5) * pd_loss_aug
            losses['pd'] = pd_loss
            total_loss = total_loss + pd_loss

            if getattr(self.config, 'use_clf', False):
                pd_loss_clf, pd_pred_clf, pd_target_clf = self._compute_clf_loss(model, pd_results['pd']['z'], batch_dict['pd'])
                losses['pd_clf'] = pd_loss_clf
                total_loss = total_loss + pd_loss_clf
                pd_pred_clfs.append(pd_pred_clf)
                pd_targets_clfs.append(pd_target_clf)

        losses['total'] = total_loss
        preds['pk'], preds['pd'], preds['pd_clf'] = pk_preds, pd_preds, pd_pred_clfs
        targets['pk'], targets['pd'], targets['pd_clf'] = pk_targets, pd_targets, pd_targets_clfs
        return losses, preds, targets

    # -------------------------------
    # Multitask loss
    # -------------------------------
    def _compute_multitask_loss(self, model, batch_pk: Dict[str, Any], batch_pd: Dict[str, Any], is_training: bool = True):
        losses, preds, targets = {}, {}, {}
        pk_preds, pd_preds, pd_pred_clfs = [], [], []
        pk_targets, pd_targets, pd_targets_clfs = [], [], []
        total_loss = 0.0

        # PK
        if batch_pk is not None:
            batch_pk_dict = self._apply_augmentation(self._prepare_batch(batch_pk), is_training=is_training)
            pk_results = model({'pk': batch_pk_dict})
            pk_pred = pk_results['pk']['pred']
            pk_target = batch_pk_dict['y'].squeeze(-1)
            pk_loss = self._regression_loss(
                pk_pred, pk_target, getattr(self.config, "loss_type_pk", "mse")
            )
            if self.config.use_aug_supervised:
                pk_results_aug = model({'pk': {'x': batch_pk_dict['x_aug'], 'y': batch_pk_dict['y_aug']}})
                pk_pred_aug = pk_results_aug['pk']['pred']
                pk_target_aug = batch_pk_dict['y_aug'].squeeze(-1)
                pk_loss_aug = self._regression_loss(
                    pk_pred_aug, pk_target_aug, getattr(self.config, "loss_type_pk", "mse")
                )
                pk_loss = pk_loss + getattr(self.config, 'aug_lambda', 0.5) * pk_loss_aug
            losses['pk'], total_loss = pk_loss, total_loss + pk_loss
            pk_preds.append(pk_pred)
            pk_targets.append(pk_target)

        # PD
        if batch_pd is not None:
            batch_pd_dict = self._apply_augmentation(self._prepare_batch(batch_pd), is_training=is_training)
            pd_results = model({'pd': batch_pd_dict})
            pd_pred = pd_results['pd']['pred']
            pd_target = batch_pd_dict['y'].squeeze(-1)
            pd_loss = self._regression_loss(
                pd_pred, pd_target, getattr(self.config, "loss_type_pd", "hybrid")
            )
            if self.config.use_aug_supervised:
                pd_results_aug = model({'pd': {'x': batch_pd_dict['x_aug'], 'y': batch_pd_dict['y_aug']}})
                pd_pred_aug = pd_results_aug['pd']['pred']
                pd_target_aug = batch_pd_dict['y_aug'].squeeze(-1)
                pd_loss_aug = self._regression_loss(
                    pd_pred_aug, pd_target_aug, getattr(self.config, "loss_type_pd", "hybrid")
                )
                pd_loss = pd_loss + getattr(self.config, 'aug_lambda', 0.5) * pd_loss_aug
            losses['pd'], total_loss = pd_loss, total_loss + pd_loss
            pd_preds.append(pd_pred)
            pd_targets.append(pd_target)

            if getattr(self.config, 'use_clf', False):
                pd_loss_clf, pd_pred_clf, pd_target_clf = self._compute_clf_loss(model, pd_results['pd']['z'], batch_pd_dict)
                losses['pd_clf'] = pd_loss_clf
                pd_pred_clfs.append(pd_pred_clf)
                pd_targets_clfs.append(pd_target_clf)

        losses['total'] = total_loss
        preds['pk'], preds['pd'], preds['pd_clf'] = pk_preds, pd_preds, pd_pred_clfs
        targets['pk'], targets['pd'], targets['pd_clf'] = pk_targets, pd_targets, pd_targets_clfs
        return losses, preds, targets

    # -------------------------------
    # Classification loss
    # -------------------------------
    def _compute_clf_loss(self, model, z: torch.Tensor, batch_dict: Dict[str, Any]):
        pd_target_clf = batch_dict['y_clf']
        if pd_target_clf is None:
            raise ValueError("y_clf is None but classification loss was requested")

        results = model._forward_clf(z)
        pd_pred_clf = results['pred_clf']

        if pd_target_clf.dim() == 1 or (pd_target_clf.dim() == 2 and pd_target_clf.size(-1) == 1):
            pd_target_clf = pd_target_clf.view(-1).long()
            loss = F.cross_entropy(pd_pred_clf, pd_target_clf)
        elif pd_target_clf.dim() == 2 and pd_target_clf.size(-1) == pd_pred_clf.size(-1):
            pd_target_clf = pd_target_clf.float()
            loss = F.cross_entropy(pd_pred_clf, pd_target_clf)
        else:
            raise ValueError(f"Unexpected y_clf shape {pd_target_clf.shape}, expected [B] or [B, C={pd_pred_clf.size(-1)}]")

        return loss, pd_pred_clf, pd_target_clf

    # -------------------------------
    # Helpers
    # -------------------------------
    def _prepare_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                x, y, y_clf = batch
            elif len(batch) == 2:
                x, y = batch
                y_clf = None
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
            return {'x': x, 'y': y, 'y_clf': y_clf}
        elif isinstance(batch, dict):
            return batch
        else:
            raise ValueError("Unsupported batch format")

    def _apply_augmentation(self, batch_dict, is_training=True):
        use_aug_supervised = getattr(self.config, 'use_aug_supervised', False)
        if not self.data_augmentation.aug_method or not use_aug_supervised or not is_training:
            return batch_dict

        x_orig, y_orig = batch_dict['x'], batch_dict['y']
        x_aug, y_aug = self.data_augmentation.apply_augmentation(x_orig, y_orig)

        if torch.isnan(x_aug).any() or torch.isinf(x_aug).any():
            x_aug = torch.nan_to_num(x_aug, nan=0.0, posinf=1e3, neginf=-1e3)
        if torch.isnan(y_aug).any() or torch.isinf(y_aug).any():
            y_aug = torch.nan_to_num(y_aug, nan=0.0, posinf=1e3, neginf=0.0)

        y_aug = torch.clamp(y_aug, min=0.0)
        x_aug = torch.clamp(x_aug, min=-1e3, max=1e3)

        updated_batch = batch_dict.copy()
        updated_batch['x_aug'] = x_aug
        updated_batch['y_aug'] = y_aug
        return updated_batch
