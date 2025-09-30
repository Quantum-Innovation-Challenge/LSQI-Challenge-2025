"""
Helper functions for PK/PD modeling
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pandas as pd
# Additional utility functions
import os
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from itertools import cycle
from models.heads import MSEHead
from models.encoders import MLPEncoder, ResMLPEncoder, MoEEncoder, ResMLPMoEEncoder
from models.encoders import QResMLPMoEEncoder, QMoEEncoder, QResMLPEncoder, QMLPEncoder, CNNEncoder

def generate_run_name(config) -> str:
    # Create human-readable timestamp (YYMMDD_HHMM)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    
    if config.encoder_pk or config.encoder_pd:
        pk_encoder = config.encoder_pk or config.encoder
        pd_encoder = config.encoder_pd or config.encoder
        encoder_name = f"{pk_encoder}-{pd_encoder}"
    else:
        encoder_name = config.encoder
    
    postfix_parts = []
    if getattr(config, 'use_fe', False):
        postfix_parts.append("fe")
    if getattr(config, 'use_contrast', False):
        postfix_parts.append("contrast")
    if getattr(config, 'threshold', None):
        postfix_parts.append(f"clf_threshold")
    postfix = "_".join(postfix_parts) if postfix_parts else ""
    run_name = f"{config.mode}_{encoder_name}_s{config.random_state}_{timestamp}_{postfix}"
    return run_name

def scaling_and_prepare_loader(
    data,
    features,
    batch_size,
    target_col,
    *,
    use_scaling=True,
    is_clf=False,
    threshold=3.3
):    # Extract features and target
    if isinstance(data, dict):
        train_data = data.get("train", pd.DataFrame())
        val_data   = data.get("val", pd.DataFrame())
        test_data  = data.get("test", pd.DataFrame())
    else:
        train_data = data
        val_data   = pd.DataFrame(columns=train_data.columns)
        test_data  = pd.DataFrame(columns=train_data.columns)


    X_train = train_data[features].values.astype(np.float32)
    y_train = train_data[target_col].values.reshape(-1, 1)
    y_train_clf = (y_train <= threshold).astype(np.int64).reshape(-1)
     
    scaler_y = StandardScaler() if use_scaling else None
    scaler_X = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    if scaler_y is not None:
        y_train = scaler_y.fit_transform(y_train)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_train_clf, dtype=torch.long)
    )

    # helper to process val/test
    def _process_subset(df, is_clf=False, threshold=3.3):
        if df.empty:
            return TensorDataset(
                torch.empty(0, len(features), dtype=torch.float32),
                torch.empty(0, 1, dtype=torch.float32),   # y_reg
                torch.empty(0, dtype=torch.long)          # y_clf
            )
        
        X = df[features].values.astype(np.float32)
        y = df[target_col].values.reshape(-1, 1)
        y_clf = (y <= threshold).astype(np.int64).reshape(-1)

        X = scaler_X.transform(X)
        if not is_clf and scaler_y is not None:
            y = scaler_y.transform(y)
        
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long if is_clf else torch.float32),
            torch.tensor(y_clf, dtype=torch.long if is_clf else None)
        )
    # TensorDataset
    val_dataset  = _process_subset(val_data, is_clf, threshold)
    test_dataset = _process_subset(test_data, is_clf, threshold)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, )
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return scaler_X, scaler_y, train_loader, val_loader, test_loader

class ReIter:
    """Re-iterable wrapper for data loaders"""
    def __init__(self, func, *loaders):
        self.func = func
        self.loaders = loaders
    
    def __iter__(self):
        return self.func(*self.loaders)

def roundrobin_loaders(*loaders):
    """Round-robin through multiple loaders"""
    # Create iterators for each loader
    iterators = [iter(loader) for loader in loaders]
    # Cycle through the iterators
    for iterator in cycle(iterators):
        try:
            yield next(iterator)
        except StopIteration:
            # If one loader is exhausted, remove it from the cycle
            iterators = [it for it in iterators if it is not iterator]
            if not iterators:
                break

def rr_val(*loaders):
    """Round-robin validation"""
    return roundrobin_loaders(*loaders)

def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_device(device_id=0):
    """Return available device with specified device ID"""
    if torch.cuda.is_available():
        if device_id >= torch.cuda.device_count():
            print(f"Warning: Device {device_id} not available. Using device 0 instead.")
            device_id = 0
        return torch.device(f"cuda:{device_id}")
    else:
        return torch.device("cpu")


def count_parameters(model):
    """Calculate number of parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_feature_dimensions(config, branch="pk", pk_features=None, pd_features=None):
    """Get feature dimensions based on configuration and branch"""
    if branch == "pk":
        return len(pk_features)
    elif branch == "pd":
        return len(pd_features)
    else:
        return len(pd_features)  # Default to PD dimensions

def get_pk_input_dim(config):
    """Get PK input dimension"""
    return get_feature_dimensions(config, "pk")


def get_pd_input_dim(config):
    """Get PD input dimension"""
    return get_feature_dimensions(config, "pd")