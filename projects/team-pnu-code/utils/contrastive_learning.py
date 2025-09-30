"""
PK/PD Augmentation & SimCLR Contrastive Learning
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Union
from .data_augmentation import PKPDDataAugmentation
    
class PKPDContrastiveLearning:
    def __init__(self,
                 temperature: float = 0.1,
                 aug_method: str = None,  # Unified aug_method (same as supervised training)
                 time_jitter_std: float = 0.1,
                 noise_std: float = 0.05,
                 dropout_rate: float = 0.1,
                 scale_range: Tuple[float, float] = (0.8, 1.2)):
        self.temperature = temperature
        self.aug_method = aug_method

        self.augmentation = PKPDDataAugmentation(
            aug_method=self.aug_method,
            jitter_std=time_jitter_std, 
            gaussian_noise_std=noise_std, 
            dropout_rate=dropout_rate,
            scale_range=scale_range
        )
    
    def augment_data(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation using the unified PKPDDataAugmentation class
        """
        if self.aug_method is None:
            # Default to random augmentation
            x_aug, _ = self.augmentation._apply_random_augmentation(x, torch.zeros(x.size(0), device=x.device))
        else:
            # Use specified augmentation method
            x_aug, _ = self.augmentation.apply_augmentation(x, torch.zeros(x.size(0), device=x.device))
        return x_aug
    
    def create_augmented_pairs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # first augmentation
        x1 = self.augment_data(x)
        
        # second augmentation (different augmentation)
        x2 = self.augment_data(x)
        
        return x1, x2
    
    def simclr_contrastive_loss(self, x: torch.Tensor, encoder) -> torch.Tensor:
        batch_size = x.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        
        x1, x2 = self.create_augmented_pairs(x)
        
        z1 = encoder(x1)
        z2 = encoder(x2)
        # z1 is tuple, use first element
        if isinstance(z1, tuple):
            z1 = z1[0]
        if isinstance(z2, tuple):
            z2 = z2[0]

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        representations = torch.cat([z1, z2], dim=0)
        
        # similarity matrix calculation
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # Positive pairs: (z1_i, z2_i) and (z2_i, z1_i)
        labels = torch.arange(batch_size, device=x.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # remove self-similarity
        mask = torch.eye(2 * batch_size, device=x.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # InfoNCE Loss calculation
        logits = F.log_softmax(similarity_matrix, dim=1)
        loss = F.nll_loss(logits, labels)
        
        return loss
    
    def simclr_contrastive_loss_with_projector(self, x: torch.Tensor, encoder, projector) -> torch.Tensor:
        batch_size = x.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        
        # Augmented pairs creation
        x1, x2 = self.create_augmented_pairs(x)
        
        # Encoder + Projector to extract representation
        z1 = encoder(x1)
        z2 = encoder(x2)
        
        # Projector application
        h1 = projector(z1)
        h2 = projector(z2)
        
        # L2 normalization
        h1 = F.normalize(h1, dim=1)
        h2 = F.normalize(h2, dim=1)
        
        # all representations combined
        representations = torch.cat([h1, h2], dim=0)
        
        # similarity matrix calculation
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # Positive pairs
        labels = torch.arange(batch_size, device=x.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # remove self-similarity
        mask = torch.eye(2 * batch_size, device=x.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # InfoNCE Loss calculation
        logits = F.log_softmax(similarity_matrix, dim=1)
        loss = F.nll_loss(logits, labels)
        
        return loss
    
    def contrastive_loss(self, x: torch.Tensor, encoder, projector=None) -> torch.Tensor:
        if projector is not None:
            return self.simclr_contrastive_loss_with_projector(x, encoder, projector)
        else:
            return self.simclr_contrastive_loss(x, encoder)


def create_pkpd_contrastive_learning(config) -> PKPDContrastiveLearning:
    aug_method = getattr(config, 'aug_method', None)  # Unified aug_method
    temperature = getattr(config, 'temperature', 0.1)
    time_jitter_std = getattr(config, 'time_jitter_std', 0.1)
    noise_std = getattr(config, 'noise_std', 0.05)
    dropout_rate = getattr(config, 'dropout_rate', 0.1)
    scale_range = getattr(config, 'scale_range', (0.8, 1.2))
    
    return PKPDContrastiveLearning(
        temperature=temperature,
        aug_method=aug_method,
        time_jitter_std=time_jitter_std,
        noise_std=noise_std,
        dropout_rate=dropout_rate,
        scale_range=scale_range
    )