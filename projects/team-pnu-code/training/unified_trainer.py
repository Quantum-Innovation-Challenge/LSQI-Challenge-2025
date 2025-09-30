"""
Unified PK/PD Trainer
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import numpy as np
from collections import defaultdict
import copy

from utils.logging import get_logger
from utils.helpers import get_device
from models.heads import _reg_metrics
from utils.contrastive_learning import create_pkpd_contrastive_learning
from utils.data_augmentation import create_data_augmentation
from .loss_computation import LossComputation
from .pretraining import ContrastivePretraining
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class UnifiedPKPDTrainer:
    def __init__(self, model, config, data_loaders, device=None):
        self.model = model
        self.config = config
        self.data_loaders = data_loaders
        self.device = device if device is not None else get_device()
        self.logger = get_logger(__name__)
        self.mode = config.mode

        # Move model to device
        self.model.to(self.device)

        # Setup components
        self._setup_optimizer()
        self._setup_model_save_directory()
        self._setup_components()

        # Training state
        self.best_val_loss = float('inf')
        self.best_pk_rmse = float('inf')
        self.best_pd_rmse = float('inf')
        self.patience_counter = 0
        self.epoch = 0
        self.training_history = defaultdict(list)

        self.logger.info(f"Unified Trainer initialized - Mode: {self.mode}, Device: {self.device}")
        self.logger.info(f"Number of model parameters: {self._count_parameters()}")

    # =========================
    # Setup
    # =========================
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=self.config.patience//2, 
            factor=0.5, 
            verbose=True
        )


    def _setup_model_save_directory(self):
        """Setup model save directory"""
        if self.config.encoder_pk or self.config.encoder_pd:
            pk_encoder = self.config.encoder_pk or self.config.encoder
            pd_encoder = self.config.encoder_pd or self.config.encoder
            encoder_name = f"{pk_encoder}-{pd_encoder}"
        else:
            encoder_name = self.config.encoder

        if hasattr(self.config, 'run_name') and self.config.run_name:
            self.model_save_directory = (
                Path(self.config.output_dir)
                / self.config.run_name / self.config.mode
                / encoder_name / f"s{self.config.random_state}"
            )
        else:
            self.model_save_directory = (
                Path(self.config.output_dir)
                / "models" / self.config.mode
                / encoder_name / f"s{self.config.random_state}"
            )
        self.model_save_directory.mkdir(parents=True, exist_ok=True)

    def _setup_components(self):
        """Setup training components"""
        # Data augmentation
        self.data_augmentation = create_data_augmentation(self.config)
        if self.data_augmentation.aug_method:
            self.logger.info(f"Data augmentation enabled: {self.data_augmentation.aug_method}")

        # Contrastive learning
        if self.config.use_pt_contrast:
            self.contrastive_learning = create_pkpd_contrastive_learning(self.config)
            self.logger.info(f"PK/PD Contrastive Learning enabled - Augmentation: {self.config.aug_method}")
        else:
            self.contrastive_learning = None

        # Loss computation
        self.loss_computation = LossComputation(self.config, self.data_augmentation, self.contrastive_learning)

        # Pretraining
        self.pretraining = ContrastivePretraining(
            self.model, self.config, self.data_loaders, self.device, self.logger
        )
        self.pretraining.set_model_save_directory(self.model_save_directory)
        if self.contrastive_learning:
            self.pretraining.set_contrastive_learning(self.contrastive_learning)

    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    # =========================
    # Main training entry
    # =========================
    def train(self) -> Dict[str, Any]:
        self.logger.info(f"Training start - Mode: {self.mode}, Epochs: {self.config.epochs}")

        pretraining_results = None
        if getattr(self.config, 'use_pt_contrast', False):
            self.logger.info("Starting Contrastive Pretraining Phase...")
            pretraining_results = self.pretraining.contrastive_pretraining(
                epochs=getattr(self.config, 'pretraining_epochs', 50),
                patience=getattr(self.config, 'pretraining_patience', 20)
            )
            self.pretraining.load_pretrained_model()
            self._pretraining_completed = True
            self.logger.info("Pretrained model loaded, starting supervised training...")

        if getattr(self.config, 'use_aug_supervised', False) and self.data_augmentation.aug_method:
            self.logger.info(f"Supervised training augmentation enabled: {self.data_augmentation.aug_method}")

        if self.mode in ["independent", "separate"]:
            return self._train_mode_independent(pretraining_results)
        else:
            return self._train_modes(pretraining_results)

    # =========================
    # Standard training loop
    # =========================
    def _train_modes(self, pretraining_results=None) -> Dict[str, Any]:
        start_time = time.time()

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            train_metrics = self._train_epoch_standard()
            val_metrics = self._validate_epoch_standard(self._get_val_loaders())

            self._log_metrics(epoch, train_metrics, val_metrics)
            self.scheduler.step(val_metrics['total_loss'])

            if self._check_early_stopping(val_metrics['total_loss']):
                self.logger.info(f"Early stopping - Epoch {epoch}")
                break

            if self._should_save_model(val_metrics):
                self._save_best_model()

        self.logger.info(f"Training completed - Time: {time.time() - start_time:.2f} seconds")
        return self._get_final_results()

    # =========================
    # Independent mode training
    # =========================
    def _train_mode_independent(self, pretraining_results=None) -> Dict[str, Any]:
        start_time = time.time()

        self.logger.info("=== PHASE 1: Training PK Model ===")
        pk_results = self._train_task("pk", "train_pk", "val_pk")

        self.logger.info("=== PHASE 2: Training PD Model ===")
        pd_results = self._train_task("pd", "train_pd", "val_pd")

        test_metrics = self._validate_epoch_standard(self._get_test_loaders())
        return {
            'pk': pk_results,
            'pd': pd_results,
            'mode': 'independent',
            'training_time': time.time() - start_time,
            'test_metrics': test_metrics
        }

    def _train_task(self, task: str, train_key: str, val_key: str) -> Dict[str, Any]:
        best_rmse = float('inf')
        no_improve = 0

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            train_metrics = self._train_single_epoch(task, train_key, is_training=True)
            val_metrics = self._train_single_epoch(task, val_key, is_training=False)

            self.logger.info(
                f"{task.upper()} Epoch {epoch:3d} | Train | "
                f"Loss: {train_metrics[f'{task}_loss']:.6f} | RMSE: {train_metrics[f'{task}_rmse']:.6f} | R²: {train_metrics[f'{task}_r2']:.4f}"
            )
            self.logger.info(
                f"{task.upper()} Epoch {epoch:3d} | Valid | "
                f"Loss: {val_metrics[f'{task}_loss']:.6f} | RMSE: {val_metrics[f'{task}_rmse']:.6f} | R²: {val_metrics[f'{task}_r2']:.4f}"
            )

            if val_metrics[f"{task}_rmse"] < best_rmse:
                best_rmse = val_metrics[f"{task}_rmse"]
                self._save_best_model()
                no_improve = 0
                self.logger.info(f"New {task.upper()} best model - RMSE: {best_rmse:.6f} at Epoch {epoch}")
            else:
                no_improve += 1

            if no_improve >= self.config.patience:
                self.logger.info(f"{task.upper()} early stopping - Epoch {epoch}")
                break

        return {'best_rmse': best_rmse, 'epochs_trained': epoch + 1}

    # =========================
    # Single task epoch
    # =========================
    def _train_single_epoch(self, task: str, loader_key: str, is_training=True) -> Dict[str, float]:
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        total_loss, preds, targets, num_batches = 0.0, [], [], 0

        loader = self.data_loaders[loader_key]
        with torch.set_grad_enabled(is_training):
            for batch in loader:
                if is_training:
                    self.optimizer.zero_grad()

                batch = self._to_device(batch)
                other_batch = None if task == "pk" else batch
                loss_dict, _, _ = self.loss_computation.compute_loss(
                    self.model,
                    batch if task == "pk" else None,
                    other_batch,
                    self.mode,
                    is_training=is_training
                )
                if is_training:
                    loss_dict['total'].backward()
                    self.optimizer.step()

                total_loss += loss_dict[task].item()
                batch_dict, target = (
                    ({"x": batch[0], "y": batch[1]}, batch[1])
                    if isinstance(batch, (list, tuple)) else (batch, batch['y'])
                )

                if self.config.use_mc_dropout:
                    results = self.model.predict_with_uncertainty({task: batch_dict})
                else:
                    results = self.model({task: batch_dict})

                preds.append(results[task]['pred'].detach().cpu())
                targets.append(target.detach().cpu())
                num_batches += 1

        metrics = _reg_metrics(torch.cat(preds).squeeze(), torch.cat(targets).squeeze())
        return {
            f"{task}_loss": total_loss / num_batches,
            **{f"{task}_{k}": float(v) for k, v in metrics.items()}
        }

    # =========================
    # Standard standard training epoch
    # =========================
    def _train_epoch_standard(self) -> Dict[str, float]:
        results = {}
        for task, loader_key in [("pk", "train_pk"), ("pd", "train_pd")]:
            task_metrics = self._train_single_epoch(task, loader_key, is_training=True)
            results.update(task_metrics)

        results['total_loss'] = results['pk_loss'] + results['pd_loss']
        return results

    def _validate_epoch_standard(self, loaders: List[Any]) -> Dict[str, float]:
        results = {}
        for task, loader_key in [("pk", "val_pk"), ("pd", "val_pd")]:
            task_metrics = self._train_single_epoch(task, loader_key, is_training=False)
            results.update(task_metrics)

        results['total_loss'] = results['pk_loss'] + results['pd_loss']
        return results

    # =========================
    # Utilities
    # =========================
    def _get_train_loaders(self):
        return [self.data_loaders['train_pk'], self.data_loaders['train_pd']]

    def _get_val_loaders(self):
        return [self.data_loaders['val_pk'], self.data_loaders['val_pd']]

    def _get_test_loaders(self):
        return [self.data_loaders['test_pk'], self.data_loaders['test_pd']]

    def _to_device(self, batch):
        if torch.is_tensor(batch):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(v) for v in batch)
        return batch

    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        for k, v in train_metrics.items():
            self.training_history[f"train_{k}"].append(v)
        for k, v in val_metrics.items():
            self.training_history[f"val_{k}"].append(v)

        self.logger.info(
            f"Epoch {epoch:4d} | Train | Loss: {train_metrics['total_loss']:.6f} "
            f"| PK RMSE: {train_metrics.get('pk_rmse', 0.0):.6f} "
            f"| PD RMSE: {train_metrics.get('pd_rmse', 0.0):.6f}"
        )
        self.logger.info(
            f"Epoch {epoch:4d} | Valid | Loss: {val_metrics['total_loss']:.6f} "
            f"| PK RMSE: {val_metrics.get('pk_rmse', 0.0):.6f} "
            f"| PD RMSE: {val_metrics.get('pd_rmse', 0.0):.6f}"
        )

    def _check_early_stopping(self, val_loss: float) -> bool:
        if val_loss + 1e-5 < self.best_val_loss:
            self.patience_counter = 0
            self.best_val_loss = val_loss
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience

    def _should_save_model(self, val_metrics: Dict[str, float]) -> bool:
        should_save = False
        if val_metrics['pd_rmse'] < self.best_pd_rmse:
            self.best_pd_rmse = val_metrics['pd_rmse']
            should_save = True
            self.logger.info(f"New PD best model - RMSE: {self.best_pd_rmse:.6f} at Epoch {self.epoch}")
        return should_save

    def _save_best_model(self):
        self.model_save_directory.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving best model to {self.model_save_directory}")
        model_path = self.model_save_directory / "best_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'val_loss': self.best_val_loss,
            'config': vars(self.config) if not isinstance(self.config, dict) else self.config
        }, str(model_path))


    def save_model(self, filename: str):
        self.model_save_directory.mkdir(parents=True, exist_ok=True)
        model_path = self.model_save_directory / filename

        model_for_saving = copy.deepcopy(self.model)

        def remove_batch_input_functions(module):
            for _, child in module.named_children():
                if hasattr(child, 'batched_qnode'):
                    child.batched_qnode = None
                remove_batch_input_functions(child)

        remove_batch_input_functions(model_for_saving)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model': model_for_saving,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'val_loss': self.best_val_loss,
            'config': self.config,
            'training_history': dict(self.training_history)
        }, model_path)
        self.logger.info(f"Model saved: {model_path}")

    def _get_final_results(self) -> Dict[str, Any]:
        self.logger.info("Evaluating on test set...")
        test_metrics = self._validate_epoch_standard(self._get_test_loaders())
        return {
            'best_val_loss': self.best_val_loss,
            'best_pk_rmse': self.best_pk_rmse,
            'best_pd_rmse': self.best_pd_rmse,
            'final_epoch': self.epoch,
            'training_history': dict(self.training_history),
            'model_info': self.model.get_model_info(),
            'test_metrics': test_metrics
        }

    # =========================
    # Classification fine-tuning
    # =========================
    def train_clf(self, freeze_encoder: bool = True) -> Dict[str, Any]:
        """
        Fine-tune classification head only (encoder frozen).
        Requires train_pd / val_pd / test_pd dataloaders to return (x, y_reg, y_clf) or dict with keys.
        """
        self.logger.info("=== Starting classification fine-tuning ===")

        # 1. Freeze encoder
        if freeze_encoder:
            for enc_name in ["pk_encoder", "pd_encoder", "encoder"]:
                if hasattr(self.model, enc_name):
                    for p in getattr(self.model, enc_name).parameters():
                        p.requires_grad = False
            self.logger.info("Encoders frozen. Only head_clf will be trained.")

        # 2. Optimizer only for classification head
        self.optimizer = torch.optim.Adam(
            self.model.head_clf.parameters(),
            lr=getattr(self.config, "clf_lr", 1e-3)
        )

        # 3. Loss function
        criterion = torch.nn.CrossEntropyLoss()

        # 4. Training state
        best_val_loss = float("inf")
        best_val_metrics = {}
        history = {"train_loss": [], "val_loss": [], "val_metrics": []}
        patience = getattr(self.config, "clf_patience", 5)
        no_improve = 0

        # 5. Training loop
        for epoch in range(getattr(self.config, "clf_epochs", 10)):
            # -------- Training --------
            self.model.train()
            train_losses = []

            for batch in self.data_loaders["train_pd"]:
                batch = self._to_device(batch)

                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x, y_reg, y_clf = batch
                elif isinstance(batch, dict):
                    x, y_reg, y_clf = batch["x"], batch.get("y"), batch["y_clf"]
                else:
                    raise ValueError("Unsupported batch format for classification")

                self.optimizer.zero_grad()
                results = self.model({'pd': {'x': x, 'y': y_reg}})
                z = results['pd']['z']
                outputs = self.model._forward_clf(z)
                logits = outputs["pred_clf"]    

                loss = criterion(logits, y_clf)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))
            history["train_loss"].append(avg_train_loss)

            # -------- Validation --------
            self.model.eval()
            val_losses, all_preds, all_targets = [], [], []
            with torch.no_grad():
                for batch in self.data_loaders["val_pd"]:
                    batch = self._to_device(batch)

                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        x, y_reg, y_clf = batch
                    elif isinstance(batch, dict):
                        x, y_reg, y_clf = batch["x"], batch.get("y"), batch["y_clf"]
                    else:
                        raise ValueError("Unsupported batch format for classification")

                    results = self.model({'pd': {'x': x, 'y': y_reg}})
                    z = results['pd']['z']
                    outputs = self.model._forward_clf(z)
                    logits = outputs["pred_clf"]

                    loss = criterion(logits, y_clf)
                    val_losses.append(loss.item())

                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(y_clf.cpu().numpy())

            avg_val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else float("inf")

            if len(all_targets) > 0:
                acc = accuracy_score(all_targets, all_preds)
                f1 = f1_score(all_targets, all_preds, average="weighted")
                prec = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
                rec = recall_score(all_targets, all_preds, average="weighted")
            else:
                acc = f1 = prec = rec = 0.0

            metrics = {"acc": acc, "f1": f1, "precision": prec, "recall": rec}
            history["val_loss"].append(avg_val_loss)
            history["val_metrics"].append(metrics)

            self.logger.info(
                f"[Clf Fine-tune][Epoch {epoch+1}] "
                f"Train loss: {avg_train_loss:.4f} | "
                f"Val loss: {avg_val_loss:.4f} | "
                f"Acc: {acc:.4f} | F1: {f1:.4f}"
            )

            # -------- Early stopping & model save --------
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_metrics = metrics
                model_path = self.model_save_directory / "best_model_clf.pth"
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"- Best classification model saved at Epoch {epoch+1}")
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # 6. Test evaluation (after best model is chosen)
        self.logger.info("=== Evaluating classification on test set ===")
        self.model.eval()
        test_losses, test_preds, test_targets = [], [], []
        with torch.no_grad():
            for batch in self.data_loaders["test_pd"]:
                batch = self._to_device(batch)

                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x, y_reg, y_clf = batch
                elif isinstance(batch, dict):
                    x, y_reg, y_clf = batch["x"], batch.get("y"), batch["y_clf"]
                else:
                    raise ValueError("Unsupported batch format for classification")

                results = self.model({'pd': {'x': x, 'y': y_reg}})
                z = results['pd']['z']
                outputs = self.model._forward_clf(z)
                logits = outputs["pred_clf"]

                loss = criterion(logits, y_clf)
                test_losses.append(loss.item())

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                test_preds.extend(preds)
                test_targets.extend(y_clf.cpu().numpy())

        avg_test_loss = float(np.mean(test_losses)) if len(test_losses) > 0 else float("inf")
        test_acc = accuracy_score(test_targets, test_preds)
        test_f1 = f1_score(test_targets, test_preds, average="weighted")
        test_prec = precision_score(test_targets, test_preds, average="weighted", zero_division=0)
        test_rec = recall_score(test_targets, test_preds, average="weighted")

        test_metrics = {
            "test_loss": avg_test_loss,
            "acc": test_acc,
            "f1": test_f1,
            "precision": test_prec,
            "recall": test_rec,
        }

        self.logger.info(
            f"[Clf Test] Loss: {avg_test_loss:.4f} | "
            f"Acc: {test_acc:.4f} | F1: {test_f1:.4f}"
        )

        return {
            "best_val_loss": best_val_loss,
            "best_val_metrics": best_val_metrics,
            "history": history,
            "test_metrics": test_metrics,
        }
