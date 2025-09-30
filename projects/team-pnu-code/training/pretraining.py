"""
Contrastive pretraining methods for PK/PD training
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List
from pathlib import Path


class ContrastivePretraining:
    def __init__(self, model, config, data_loaders, device, logger):
        self.model = model
        self.config = config
        self.data_loaders = data_loaders

        self.device = device
        self.logger = logger
        self.model_save_directory = None
        
        # PK label integration for better PD pretraining
        self.pk_integrator = PKLabelIntegrator(logger)
        self.cross_modal_learning = CrossModalContrastiveLearning(temperature=config.temperature)
    
    def set_model_save_directory(self, directory: Path):
        """Set model save directory"""
        self.model_save_directory = directory
    
    def contrastive_pretraining(self, epochs: int = 50, patience: int = 20) -> Dict[str, Any]:
        """Contrastive Learning Pretraining Phase with PK-PD Integration"""
        self.logger.info(f"Starting Enhanced Contrastive Pretraining - Epochs: {epochs}")

        self._train_pk_predictor()
        
        pt_history = {'total_loss': [], 'contrastive_loss': [], 'clf_loss': [], 'learning_rate': []}
        
        pt_optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate * 2, 
            weight_decay=1e-4
        )
        pt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pt_optimizer, T_max=epochs)

        best_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        for epoch in range(epochs):
            ct_loss = self._enhanced_pretraining_epoch(pt_optimizer)
            if self.config.use_pt_clf: 
                clf_loss = self._pretraining_epoch_clf(pt_optimizer)
                total_loss = ct_loss + 10* clf_loss
            else:
                clf_loss = 0.0
                total_loss = ct_loss   
                
            pt_history['total_loss'].append(total_loss)
            pt_history['contrastive_loss'].append(ct_loss)
            pt_history['clf_loss'].append(clf_loss)
            pt_history['learning_rate'].append(pt_optimizer.param_groups[0]['lr'])
            
            pt_scheduler.step()
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                self.logger.info(
                    f"Pretraining Epoch {epoch:3d} | "
                    f"Total Loss: {total_loss:.6f} | "
                    f"Contrastive Loss: {ct_loss:.6f} | "
                    f"CLF Loss: {clf_loss:.6f} | "
                    f"Best Loss: {best_loss:.6f} | "
                    f"at Epoch: {best_epoch}"
                )
            
            if total_loss < best_loss:
                best_loss = total_loss
                best_epoch = epoch
                patience_counter = 0
                self._save_pretrained_model()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch} with best loss: {best_loss:.6f}")
                    break
        
        self.logger.info(f"Enhanced Contrastive Pretraining completed - Best Loss: {best_loss:.6f}")
        
        return {
            'best_loss': best_loss,
            'pretraining_history': pt_history,
            'epochs_pretrained': best_epoch + 1
        }
    
    def _compute_contrastive_loss(self, batch_dict: Dict[str, Any], task: str) -> torch.Tensor:
        """Compute contrastive loss for specific task"""
        if not hasattr(self, 'contrastive_learning') or self.contrastive_learning is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        x = batch_dict['x']
        if task == 'pd' and 'pk_labels' in batch_dict:
            pk_labels = batch_dict['pk_labels']
            x = torch.cat([x, pk_labels.unsqueeze(-1) if pk_labels.dim() == 1 else pk_labels], dim=1)
        
        encoder = self._get_encoder(self.model, task)
        
        if encoder is None:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        return self.contrastive_learning.contrastive_loss(x, encoder)
    
    def _compute_contrastive_loss_with_pk_labels(self, batch_dict: Dict[str, Any], task: str) -> torch.Tensor:
        """Compute contrastive loss for PD with PK label integration"""
        if not hasattr(self, 'contrastive_learning') or self.contrastive_learning is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        x = batch_dict['x']
        if task == 'pd' and 'pk_labels' in batch_dict:
            pk_labels = batch_dict['pk_labels']
            x_enhanced = torch.cat([x, pk_labels.unsqueeze(-1) if pk_labels.dim() == 1 else pk_labels], dim=1)
        else:
            x_enhanced = x
        
        encoder = self._get_encoder(self.model, task)
               
        return self.contrastive_learning.contrastive_loss(x_enhanced, encoder)
    
    def _train_pk_predictor(self):
        """Train PK predictor using PK data"""
        self.logger.info("Training PK predictor for enhanced PD pretraining...")
        
        # Get PK data for training predictor
        pk_batches = list(self.data_loaders['train_pk'])
        if not pk_batches:
            self.logger.warning("No PK training data available for predictor training")
            return

        pk_feats, pk_targets, pk_targets_clf = [], [], []
        for batch in pk_batches:
            batch = self._to_device(batch)
            if isinstance(batch, (list, tuple)):
                x, y, y_clf = batch
                batch_dict = {'x': x, 'y': y, 'y_clf': y_clf}
            else:
                batch_dict = batch
            pk_feats.append(batch_dict['x'])
            pk_targets.append(batch_dict['y'])
            pk_targets_clf.append(batch_dict['y_clf'])
        
        # Concatenate all PK data
        pk_features = torch.cat(pk_feats, dim=0)
        pk_targets = torch.cat(pk_targets, dim=0)
        pk_targets_clf = torch.cat(pk_targets_clf, dim=0)
        
        # Train PK predictor
        pk_data = {'x': pk_features, 'y': pk_targets, 'y_clf': pk_targets_clf}
        pd_data = {'x': pk_features, 'y': pk_targets, 'y_clf': pk_targets_clf}  # Dummy for now
        
        self.pk_integrator.train_pk_predictor(pk_data)
        self.logger.info("PK predictor training completed")

    def _enhanced_pretraining_epoch(self, optimizer) -> float:
        """Enhanced pretraining epoch with PK-PD integration (train + val)"""
        self.model.train()
        total_contrastive_loss = 0.0
        num_batches = 0
        
        # Get enhanced PD batches with PK labels (train + val)
        enhanced_pd_batches = self._get_enhanced_pd_batches()
        
        # Process PK batches (train + val)
        all_pk_batches = list(self.data_loaders.get('train_pk', [])) + list(self.data_loaders.get('val_pk', []))
        for batch_pk in all_pk_batches:
            optimizer.zero_grad()
            
            batch_pk = self._to_device(batch_pk)
            batch_pk_dict = self._convert_batch_to_dict(batch_pk, 'pk')
            
            if batch_pk_dict is not None:
                pk_contrastive_loss = self._compute_contrastive_loss(batch_pk_dict, 'pk')
                pk_contrastive_loss.backward()
                optimizer.step()
                
                total_contrastive_loss += pk_contrastive_loss.item()
                num_batches += 1
        
        # Process enhanced PD batches
        for enhanced_pd_batch in enhanced_pd_batches:
            optimizer.zero_grad()
            total_pd_loss = self._compute_contrastive_loss(enhanced_pd_batch, 'pd')

            total_pd_loss.backward()
            optimizer.step()
            
            total_contrastive_loss += total_pd_loss.item()
            num_batches += 1
        
        return total_contrastive_loss / num_batches if num_batches > 0 else 0.0

    def _get_enhanced_pd_batches(self) -> List[Dict[str, Any]]:
        """Get enhanced PD batches with PK labels"""
        enhanced_batches = []
        
        for batch_pd in self.data_loaders['train_pd']:
            batch_pd = self._to_device(batch_pd)
            batch_pd_dict = self._convert_batch_to_dict(batch_pd, 'pd')
            enhanced_batch = self.pk_integrator.enhance_pd_batch_with_pk_labels(batch_pd_dict)
            enhanced_batches.append(enhanced_batch)
        
        return enhanced_batches

    def _pretraining_epoch_clf(self, optimizer) -> float:
        """Pretraining epoch with classification task"""
        self.model.train()
        total_pd_loss = 0.0
        num_batches = 0

        enhanced_pd_batches = self._get_enhanced_pd_batches()
        for enhanced_pd_batch in enhanced_pd_batches:
            optimizer.zero_grad()
            pd_loss = self._compute_classification_loss(enhanced_pd_batch, 'pd')
            
            pd_loss.backward()
            optimizer.step()
            
            total_pd_loss += pd_loss.item()
            num_batches += 1
            
        return total_pd_loss / num_batches if num_batches > 0 else 0.0


    def _get_pd_batches_with_pk(self) -> List[Dict[str, Any]]:
        """Get enhanced PD batches with PK labels (train + val)"""
        enhanced_batches = []
        all_pd_batches = list(self.data_loaders.get('train_pd', [])) + list(self.data_loaders.get('val_pd', []))
        
        for batch_pd in all_pd_batches:
            batch_pd = self._to_device(batch_pd)
            batch_pd_dict = self._convert_batch_to_dict(batch_pd, 'pd')
            # Enhance with PK labels
            enhanced_batch = self.pk_integrator.enhance_pd_batch_with_pk_labels(batch_pd_dict)
            enhanced_batches.append(enhanced_batch)
        
        return enhanced_batches
    

        
    def _compute_cross_modal_loss(self, enhanced_pd_batch: Dict[str, Any]) -> torch.Tensor:
        """Compute cross-modal contrastive loss"""
        if not hasattr(self, 'contrastive_learning') or self.contrastive_learning is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        pd_x = enhanced_pd_batch['x']
        pk_labels = enhanced_pd_batch.get('pk_labels', None)
        
        pk_encoder = self._get_encoder(self.model, 'pk')
        pd_encoder = self._get_encoder(self.model, 'pd')
        
        try:
            if pk_labels is not None:
                z_pk = pk_labels.unsqueeze(-1) if pk_labels.dim() == 1 else pk_labels
            else:
                z_pk = pk_encoder(pd_x)
            
            z_pd = pd_encoder(pd_x)
            cross_modal_loss = self.cross_modal_learning.compute_cross_modal_loss(z_pk, z_pd)
            
            return cross_modal_loss
            
        except Exception as e:
            self.logger.debug(f"Cross-modal learning failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def _get_pretraining_loaders(self) -> List[Any]:
        """Get data loaders for pretraining"""
        return [self.data_loaders['train_pk'], self.data_loaders['train_pd']]
    
    def _convert_batch_to_dict(self, batch: Any, task: str) -> Dict[str, Any]:
        """Convert batch to dictionary format"""
        if isinstance(batch, (list, tuple)):
            x, y, y_clf = batch
            return {'x': x, 'y': y, 'y_clf': y_clf}
        elif isinstance(batch, dict):
            return batch
        else:
            return {'x': batch, 'y': None, 'y_clf': None}
    
    def _save_pretrained_model(self):
        """Save pretrained model"""
        if self.model_save_directory is None:
            return
        
        pretrained_path = self.model_save_directory / "pretrained_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'pretraining_phase': True,
            'config': self.config
        }, pretrained_path)
    
    def load_pretrained_model(self):
        """Load pretrained model"""
        if self.model_save_directory is None:
            return
        
        pretrained_path = self.model_save_directory / "pretrained_model.pth"
        if pretrained_path.exists():
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Pretrained model loaded successfully")
        else:
            self.logger.warning("No pretrained model found")
    
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device"""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [v.to(self.device) if torch.is_tensor(v) else v for v in batch]
        else:
            return batch.to(self.device)
    
    def set_contrastive_learning(self, contrastive_learning):
        """Set contrastive learning instance"""
        self.contrastive_learning = contrastive_learning

    def _compute_classification_loss(self, batch_dict: Dict[str, Any], task: str) -> torch.Tensor:

        x = torch.cat([batch_dict['x'], batch_dict['pk_labels']], dim=1).to(self.device)
        y_clf = batch_dict['y_clf'].long().view(-1).to(self.device)  # [B]

        encoder = self._get_encoder(self.model, task)

        z = encoder(x)
        logits = self.model.head_clf(z)  # [B, 2]

        loss = F.cross_entropy(logits['pred'], y_clf)
        return loss

    def _get_encoder(self, model, task: str):
        """Get encoder for specific task"""
        if task == 'pk':
            if hasattr(model, 'pk_encoder'):
                return model.pk_encoder
            else:
                return model.encoder
        elif task == 'pd':
            if hasattr(model, 'pd_encoder'):
                return model.pd_encoder
            else:
                return model.encoder
        else:
            raise ValueError(f"Unknown task: {task}")
    

class PKPredictor(nn.Module):
    """Simple linear model to predict PK labels from PK features"""
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class PKLabelIntegrator:
    """Handles training of PK predictor and integration of PK labels into PD batches."""
    def __init__(self, logger):
        self.pk_predictor = None
        self.logger = logger

    def train_pk_predictor(self, pk_data: Dict[str, torch.Tensor], epochs: int = 50):
        """
        Trains a simple PK predictor using PK data.
        pk_data: {'x': pk_features, 'y': pk_targets}
        """
        self.logger.info("Training PK predictor for PD pretraining...")
        
        pk_feats = pk_data['x']
        pk_targets = pk_data['y']

        input_dim = pk_feats.shape[-1]
        self.pk_predictor = PKPredictor(input_dim).to(pk_feats.device)
        optimizer = optim.Adam(self.pk_predictor.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Simple training loop for the predictor
        for epoch in range(epochs):  # Small number of epochs
            optimizer.zero_grad()
            predictions = self.pk_predictor(pk_feats)
            loss = criterion(predictions.squeeze(), pk_targets.squeeze())
            loss.backward()
            optimizer.step()
        self.logger.info("PK predictor trained successfully")

    def enhance_pd_batch_with_pk_labels(self, pd_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhances a PD batch by adding predicted PK labels.
        Requires pk_predictor to be trained.
        """
        if self.pk_predictor is None:
            self.logger.warning("PK predictor not trained. Cannot enhance PD batch with PK labels.")
            return pd_batch
        pk_pred = self.pk_predictor(pd_batch['x'])
        pd_batch_new = pd_batch.copy()
        pd_batch_new['pk_labels'] = pk_pred.detach() # Detach to prevent gradients flowing back to predictor during CL
        return pd_batch_new


class CrossModalContrastiveLearning:
    """
    Handles cross-modal contrastive learning between PK and PD representations.
    """
    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature

    def compute_cross_modal_loss(self, pk_features: torch.Tensor, pd_features: torch.Tensor) -> torch.Tensor:
        """
        Computes InfoNCE loss between PK and PD features.
        Assumes pk_features and pd_features are already representations (e.g., from encoders).
        """
        import torch.nn.functional as F
        
        batch_size = pk_features.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=pk_features.device, requires_grad=True)

        # Normalize features
        pk_features = F.normalize(pk_features, dim=1)
        pd_features = F.normalize(pd_features, dim=1)

        # Compute similarity matrix between PK and PD features
        # Shape: (batch_size, batch_size)
        similarity_matrix = torch.matmul(pk_features, pd_features.T) / self.temperature

        # Positive pairs are (pk_i, pd_i)
        labels = torch.arange(batch_size, device=pk_features.device)

        # InfoNCE Loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
