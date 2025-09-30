"""
Data Augmentation utilities for PK/PD training
"""

import torch
import torch.nn.functional as F
import random
from typing import Tuple, Dict, Any


class PKPDDataAugmentation:
    def __init__(self,
                 aug_method: str = None,
                 mixup_alpha: float = 0.3,
                 jitter_std: float = 0.05,
                 time_shift_ratio: float = 0.1,
                 mixup_prob: float = 0.1,
                 gaussian_noise_std: float = 0.02,
                 scale_range: tuple = (0.8, 1.2),
                 dropout_rate: float = 0.1,
                 time_warp_factor: float = 0.1,
                 amplitude_scale_range: tuple = (0.9, 1.1),
                 cutmix_alpha: float = 1.0,
                 cutmix_prob: float = 0.1,
                 label_smooth_eps: float = 0.1,
                 random_erase_prob: float = 0.1,
                 feature_dropout_prob: float = 0.1):

        self.aug_method = aug_method
        self.mixup_alpha = mixup_alpha
        self.jitter_std = jitter_std
        self.time_shift_ratio = time_shift_ratio
        self.mixup_prob = mixup_prob

        self.gaussian_noise_std = gaussian_noise_std
        self.scale_range = scale_range
        self.dropout_rate = dropout_rate
        self.time_warp_factor = time_warp_factor
        self.amplitude_scale_range = amplitude_scale_range
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob
        self.label_smooth_eps = label_smooth_eps
        self.random_erase_prob = random_erase_prob
        self.feature_dropout_prob = feature_dropout_prob
        
    def apply_augmentation(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.aug_method:
            return x, y

        if self.aug_method == "mixup":
            return self._apply_mixup(x, y)
        elif self.aug_method == "jitter":
            return self._apply_jitter(x, y)
        elif self.aug_method == "jitter_mixup":
            return self._apply_jitter_mixup(x, y)
        elif self.aug_method == "gaussian_noise":
            return self._apply_gaussian_noise(x, y)
        elif self.aug_method == "scaling":
            return self._apply_scaling(x, y)
        elif self.aug_method == "time_warp":
            return self._apply_time_warp(x, y)
        elif self.aug_method == "feature_dropout":
            return self._apply_feature_dropout(x, y)
        elif self.aug_method == "cutmix":
            return self._apply_cutmix(x, y)
        elif self.aug_method == "random_erase":
            return self._apply_random_erase(x, y)
        elif self.aug_method == "label_smooth":
            return self._apply_label_smoothing(x, y)
        elif self.aug_method == "amplitude_scale":
            return self._apply_amplitude_scale(x, y)
        elif self.aug_method == "enhanced_mixup":
            return self._apply_enhanced_mixup(x, y)
        elif self.aug_method == "random":
            return self._apply_random_augmentation(x, y)
        elif self.aug_method == "pk_curve":
            return self._apply_pk_curve_augmentation(x, y)
        elif self.aug_method == "pd_response":
            return self._apply_pd_response_augmentation(x, y)
        else:
            return x, y
    
    def _apply_mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup augmentation"""
        if x.size(0) < 2 or random.random() > self.mixup_prob:
            return x, y
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Mixup lambda
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample()
        
        # Mix samples
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def _apply_jitter(self, x: torch.Tensor, y: torch.Tensor, task: str = "pk") -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply jitter augmentation (task-aware)."""
        x_jittered = x.clone()
        y_jittered = y.clone()

        # Time shift (both PK and PD)
        if x.size(1) > 0:  # Assuming first column is TIME
            time_shift = torch.randn(x.size(0), 1).to(x.device) * self.time_shift_ratio
            x_jittered[:, 0] = (x[:, 0] + time_shift.squeeze()).clamp(min=0)

        if task == "pk":
            noise = torch.randn_like(y) * self.jitter_std * y.std()
            y_jittered = (y + noise).clamp(min=0)

        elif task == "pd":
            y_jittered = y

        return x_jittered, y_jittered

    
    def _apply_jitter_mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply jitter + mixup augmentation"""
        # First apply jitter
        x_jittered, y_jittered = self._apply_jitter(x, y)
        
        # Then apply mixup (returns 4 values, but we only need the first 2)
        mixup_result = self._apply_mixup(x_jittered, y_jittered)
        if len(mixup_result) == 4:
            mixed_x, y_a, y_b, lam = mixup_result
            # For simplicity, return the mixed input and first target
            return mixed_x, y_a
        else:
            return mixup_result


    def _apply_gaussian_noise(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Gaussian noise to PK/PD data"""
        # Add noise to features
        x_noise = torch.randn_like(x) * self.gaussian_noise_std
        x_aug = x + x_noise

        # Add noise to target (concentration/DV values)
        if y.numel() > 0:
            y_noise = torch.randn_like(y) * self.gaussian_noise_std * y.std()
            y_aug = y + y_noise
            y_aug = torch.clamp(y_aug, min=0)  # Ensure positive values
        else:
            y_aug = y

        return x_aug, y_aug

    def _apply_scaling(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random scaling to PK/PD data"""
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        x_aug = x * scale_factor

        # Scale target proportionally
        y_aug = y * scale_factor

        return x_aug, y_aug

    def _apply_time_warp(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply time warping to PK/PD time series"""
        if x.size(1) < 2:  # Need at least time and one feature
            return x, y

        # Assume first column is TIME
        time_col = x[:, 0].clone()
        other_cols = x[:, 1:].clone()

        # Apply time warping
        warp_factor = 1.0 + random.uniform(-self.time_warp_factor, self.time_warp_factor)
        warped_time = time_col * warp_factor

        # Reconstruct
        x_aug = torch.cat([warped_time.unsqueeze(1), other_cols], dim=1)

        return x_aug, y

    def _apply_feature_dropout(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply feature dropout to PK/PD features"""
        if x.size(1) <= 1:  # Need at least one feature to drop
            return x, y

        # Create dropout mask for features
        feature_mask = torch.rand(x.size(1)) > self.feature_dropout_prob
        feature_mask = feature_mask.float().to(x.device)

        # Keep at least one feature
        if feature_mask.sum() == 0:
            feature_mask[random.randint(0, x.size(1)-1)] = 1.0

        # Apply mask
        x_aug = x * feature_mask.unsqueeze(0)

        return x_aug, y

    def _apply_cutmix(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix augmentation"""
        if x.size(0) < 2 or random.random() > self.cutmix_prob:
            return x, y

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        # CutMix lambda
        lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample()

        # Random box coordinates
        cut_rat = torch.sqrt(1. - lam)
        cut_w = (x.size(1) * cut_rat).int().clamp(min=1)
        cut_h = (x.size(0) * cut_rat).int().clamp(min=1)

        # Random center
        cx = torch.randint(0, x.size(1) - cut_w + 1, (1,)).item()
        cy = torch.randint(0, x.size(0) - cut_h + 1, (1,)).item()

        # Apply CutMix
        x_aug = x.clone()
        x_aug[cy:cy+cut_h, cx:cx+cut_w] = x[index[cy:cy+cut_h], cx:cx+cut_w]

        # Interpolate labels
        y_a, y_b = y, y[index]
        y_aug = lam * y_a + (1 - lam) * y_b

        return x_aug, y_aug

    def _apply_random_erase(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Random Erasing to PK/PD features"""
        if random.random() > self.random_erase_prob:
            return x, y

        # Random erase parameters
        erase_ratio = random.uniform(0.1, 0.3)
        erase_area = x.numel() * erase_ratio
        erase_aspect_ratio = random.uniform(0.3, 3.0)

        # Calculate erase dimensions
        erase_h = int(torch.sqrt(torch.tensor(erase_area * erase_aspect_ratio)))
        erase_w = int(torch.sqrt(torch.tensor(erase_area / erase_aspect_ratio)))

        erase_h = min(erase_h, x.size(0))
        erase_w = min(erase_w, x.size(1))

        # Random position
        x_start = random.randint(0, x.size(0) - erase_h)
        y_start = random.randint(0, x.size(1) - erase_w)

        # Apply erasing
        x_aug = x.clone()
        x_aug[x_start:x_start+erase_h, y_start:y_start+erase_w] = 0

        return x_aug, y

    def _apply_label_smoothing(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply label smoothing to targets"""
        # Label smoothing for regression
        if y.numel() > 0:
            # Add small noise for smoothing
            noise = torch.randn_like(y) * self.label_smooth_eps
            y_aug = y + noise
            y_aug = torch.clamp(y_aug, min=0)  # Ensure positive values
        else:
            y_aug = y

        return x, y_aug

    def _apply_amplitude_scale(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply amplitude scaling to PK/PD signals"""
        scale_factor = random.uniform(self.amplitude_scale_range[0], self.amplitude_scale_range[1])

        # Scale features (excluding time column if present)
        if x.size(1) > 1:
            x_aug = x.clone()
            x_aug[:, 1:] = x_aug[:, 1:] * scale_factor  # Skip time column
        else:
            x_aug = x * scale_factor

        # Scale target
        y_aug = y * scale_factor

        return x_aug, y_aug

    def _apply_enhanced_mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply enhanced mixup with multiple samples"""
        if x.size(0) < 3 or random.random() > self.mixup_prob:
            return x, y

        batch_size = x.size(0)
        indices = torch.randperm(batch_size).to(x.device)

        # Mix with 3 samples
        lam1 = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample()
        lam2 = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample()

        # Weighted combination of 3 samples
        mixed_x = (lam1 * x + (1-lam1) * lam2 * x[indices[:batch_size//2]] +
                  (1-lam1) * (1-lam2) * x[indices[batch_size//2:]])
        y_a, y_b, y_c = y, y[indices[:batch_size//2]], y[indices[batch_size//2:]]
        mixed_y = lam1 * y_a + (1-lam1) * lam2 * y_b + (1-lam1) * (1-lam2) * y_c

        return mixed_x, mixed_y


    def _apply_random_augmentation(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random augmentation for contrastive learning (legacy method)"""
        augmentations = [
            self._apply_gaussian_noise,
            self._apply_scaling,
            self._apply_feature_dropout,
            self._apply_time_warp,
            self._apply_jitter,
            self._apply_amplitude_scale
        ]

        # Randomly select 2-3 augmentations
        num_augs = random.randint(2, 3)
        selected_augs = random.sample(augmentations, num_augs)

        x_aug, y_aug = x, y
        for aug in selected_augs:
            x_aug, y_aug = aug(x_aug, y_aug)

        return x_aug, y_aug

    def _apply_pk_curve_augmentation(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """PK curve specific augmentation (multiple augmentation combinations)"""
        # 1. Time axis noise
        x_aug, y_aug = self._apply_jitter(x, y)

        # 2. Gaussian noise
        x_aug, y_aug = self._apply_gaussian_noise(x_aug, y_aug)

        # 3. Scaling
        x_aug, y_aug = self._apply_scaling(x_aug, y_aug)

        return x_aug, y_aug

    def _apply_pd_response_augmentation(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """PD response specific augmentation (only input x is transformed, y is kept stable)"""
        x_aug = x.clone()

        # 1. Time axis jitter (time shift)
        if x.size(1) > 0:
            time_shift = torch.randn(x.size(0), 1).to(x.device) * self.time_shift_ratio
            x_aug[:, 0] = (x[:, 0] + time_shift.squeeze()).clamp(min=0)

        # 2. Gaussian noise (only x)
        x_aug, _ = self._apply_gaussian_noise(x_aug, y)

        # 3. Feature dropout
        x_aug, _ = self._apply_feature_dropout(x_aug, y)

        # 4. Time warp
        x_aug, _ = self._apply_time_warp(x_aug, y)

        # 5. y is kept as is (option: only add very small noise)
        y_aug = y.clone()
        y_aug = y_aug + 0.01 * torch.randn_like(y_aug)  # For stability, add very small noise
        y_aug = torch.clamp(y_aug, min=0.0)

        return x_aug, y_aug

def create_data_augmentation(config) -> PKPDDataAugmentation:
    """Create data augmentation instance from config"""
    return PKPDDataAugmentation(
        aug_method=getattr(config, 'aug_method', None),
        mixup_alpha=getattr(config, 'mixup_alpha', 0.3),
        jitter_std=getattr(config, 'jitter_std', 0.05),
        time_shift_ratio=getattr(config, 'time_shift_ratio', 0.1),
        mixup_prob=getattr(config, 'mixup_prob', 0.1),
        # Enhanced augmentation parameters
        gaussian_noise_std=getattr(config, 'gaussian_noise_std', 0.02),
        scale_range=getattr(config, 'scale_range', (0.8, 1.2)),
        dropout_rate=getattr(config, 'dropout_rate', 0.1),
        time_warp_factor=getattr(config, 'time_warp_factor', 0.1),
        amplitude_scale_range=getattr(config, 'amplitude_scale_range', (0.9, 1.1)),
        cutmix_alpha=getattr(config, 'cutmix_alpha', 1.0),
        cutmix_prob=getattr(config, 'cutmix_prob', 0.1),
        label_smooth_eps=getattr(config, 'label_smooth_eps', 0.1),
        random_erase_prob=getattr(config, 'random_erase_prob', 0.1),
        feature_dropout_prob=getattr(config, 'feature_dropout_prob', 0.1)
    )
