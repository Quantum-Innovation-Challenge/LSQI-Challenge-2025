#!/usr/bin/env python3
import sys
import time
import argparse
from pathlib import Path
import json
import pickle
import os
import random
import numpy as np
import torch

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import parse_args
from utils.logging import setup_logging, get_logger
from data.loaders import load_estdata, separate_pkpd
from utils.helpers import scaling_and_prepare_loader, get_device, generate_run_name
from utils.factory import create_model, create_trainer
from data.splits import prepare_for_split
from auto_visualization import AutoVisualizer
from pathlib import Path

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    """Main function with deterministic data splitting"""
    
    config = parse_args()
    set_seed(config.random_state) 

    
    # Generate run name
    if not config.run_name:
        config.run_name = generate_run_name(config)
    
    # Setup logging with hierarchical directory structure
    if config.encoder_pk or config.encoder_pd:
        pk_encoder = config.encoder_pk or config.encoder
        pd_encoder = config.encoder_pd or config.encoder
        encoder_name = f"{pk_encoder}-{pd_encoder}"
    else:
        encoder_name = config.encoder
    
    # Create hierarchical log directory: logs/{run_name}/{mode}/{encoder}/s{seed}/
    if config.run_name:
        log_dir = f"{config.output_dir}/{config.run_name}/{config.mode}/{encoder_name}/s{config.random_state}/logs"
    else:
        # Fallback to the old structure if no run_name
        log_dir = f"{config.output_dir}/logs/{config.mode}/{encoder_name}/s{config.random_state}"
    
    log_file = setup_logging(log_dir, config.verbose, config.run_name)
    logger = get_logger(__name__)
    
    # Flush output
    sys.stdout.flush()
    sys.stderr.flush()
    
    logger.info("=== PK/PD Modeling with Deterministic Splitting ===")
    logger.info(f"Run name: {config.run_name}")
    
    # Encoder information
    if config.encoder_pk or config.encoder_pd:
        logger.info(f"Mode: {config.mode} | PK Encoder: {pk_encoder} | PD Encoder: {pd_encoder} | Epochs: {config.epochs}")
    else:
        logger.info(f"Mode: {config.mode} | Encoder: {config.encoder} | Epochs: {config.epochs}")
    
    logger.info(f"Batch size: {config.batch_size} | Learning rate: {config.learning_rate}")
    
    device = get_device(config.device_id)
    logger.info(f"Device: {device}")
    
    # === 1. Data loading ===
    logger.info(f"\n=== 1. Data loading ===")
    logger.info(f"CSV file: {config.csv_path}")
    
    df_all, df_obs, df_dose = load_estdata(config.csv_path)
    logger.info(f"Data loading completed - Total: {df_all.shape}, Observed: {df_obs.shape}, Dose: {df_dose.shape}")
    
    # === 2. Data preprocessing ===
    logger.info(f"\n=== 2. Data preprocessing ===")
    
    df_final, pk_df, pd_df, pd_feats, pk_feats = separate_pkpd(df_obs, df_dose, config.use_fe, config.use_perkg)

    logger.info(f"Preprocessing completed - Final data shape: {df_final.shape}")
    logger.info(f"PK data: {pk_df.shape}, PD data: {pd_df.shape}")

    # === 3. Data splitting ===
    logger.info(f"\n=== 3. Data splitting ===")
    # Use robust data splitting with multiple strategies
    pk_splits, pd_splits, global_splits, _ = prepare_for_split(
        df_final=df_final, df_dose=df_dose,
        split_strategy=config.split_strategy,
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.random_state,
        dose_bins=4,
        id_universe="union",
        verbose=True,
    )
    logger.info(f"Data splitting completed - Strategy: {config.split_strategy}")
    
    # Clean splits structure
    splits = {'pk': pk_splits, 'pd': pd_splits, 'global': global_splits, 'pk_feats': pk_feats, 'pd_feats': pd_feats}

    # === 4. Data loader creation ===
    logger.info(f"\n=== 4. Data loader creation ===")
    
    pk_scaler, pk_target_scaler, train_loader_pk, valid_loader_pk, test_loader_pk = scaling_and_prepare_loader(
        splits['pk'], pk_feats, batch_size=config.batch_size, target_col="DV", is_clf=False, threshold=config.threshold
    )
    
    # PD data loader (optimized with target scaler)
    pd_scaler, pd_target_scaler, train_loader_pd, valid_loader_pd, test_loader_pd = scaling_and_prepare_loader(
        splits['pd'], pd_feats, batch_size=config.batch_size, target_col="DV", is_clf= config.use_clf, threshold=config.threshold
        )
    
    loaders = {
        "train_pk": train_loader_pk, "val_pk": valid_loader_pk, "test_pk": test_loader_pk,
        "train_pd": train_loader_pd, "val_pd": valid_loader_pd, "test_pd": test_loader_pd,
    }
    
    logger.info("Data loader creation completed")
    logger.info(f"PK - Train: {len(train_loader_pk.dataset)}, Val: {len(valid_loader_pk.dataset)}, Test: {len(test_loader_pk.dataset)}")
    logger.info(f"PD - Train: {len(train_loader_pd.dataset)}, Val: {len(valid_loader_pd.dataset)}, Test: {len(test_loader_pd.dataset)}")
    
    # === 5. Model creation ===
    logger.info(f"\n=== 5. Model creation ===")
    
    model = create_model(config, loaders, pk_feats, pd_feats)
    logger.info(f"Model creation completed - Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # === 6. Trainer creation and training ===
    logger.info(f"\n=== 6. Training start ===")
    
    trainer = create_trainer(config, model, loaders, device)
    
    start_time = time.time()
    results = trainer.train()
    training_time = time.time() - start_time
    logger.info(f"Training completed - Time: {training_time:.2f} seconds")
    
    if config.clf_finetune:
        results_clf = trainer.train_clf()
        training_time = time.time() - start_time
        logger.info(f"Classification fine-tuning completed - Time: {training_time:.2f} seconds")

    # Handle different result structures for different modes
    if config.mode == "separate":
        logger.info(f"PK training completed - Best RMSE: {results['pk']['best_rmse']:.6f}")
        logger.info(f"PD training completed - Best RMSE: {results['pd']['best_rmse']:.6f}")
        
    # Log test performance for separate mode
    if 'test_metrics' in results:
        test_metrics = results['test_metrics']
        logger.info(f"Test Performance - PK RMSE: {test_metrics.get('pk_rmse', 'N/A'):.6f}, PD RMSE: {test_metrics.get('pd_rmse', 'N/A'):.6f}")
        logger.info(f"Test Performance - PK R²: {test_metrics.get('pk_r2', 'N/A'):.6f}, PD R²: {test_metrics.get('pd_r2', 'N/A'):.6f}")
    else:
        logger.info(f"Best validation loss: {results['best_val_loss']:.6f}")
        logger.info(f"Best PK RMSE: {results['best_pk_rmse']:.6f}")
        logger.info(f"Best PD RMSE: {results['best_pd_rmse']:.6f}")
        
        # Log test performance for other modes
        if 'test_metrics' in results:
            test_metrics = results['test_metrics']
            logger.info(f"Test Performance - PK RMSE: {test_metrics.get('pk_rmse', 'N/A'):.6f}, PD RMSE: {test_metrics.get('pd_rmse', 'N/A'):.6f}")
            logger.info(f"Test Performance - PK R²: {test_metrics.get('pk_r2', 'N/A'):.6f}, PD R²: {test_metrics.get('pd_r2', 'N/A'):.6f}")
    
    # === 7. Results saving ===
    logger.info(f"\n=== 7. Results saving ===")
    
    # Create hierarchical directory structure for all outputs
    # Structure: runs/{run_name}/{mode}/{encoder}/s{seed}/
    if config.encoder_pk or config.encoder_pd:
        pk_encoder = config.encoder_pk or config.encoder
        pd_encoder = config.encoder_pd or config.encoder
        encoder_name = f"{pk_encoder}-{pd_encoder}"
    else:
        encoder_name = config.encoder
    
    # Unified structure: {run_name}/{mode}/{encoder}/s{random_state}/
    run_dir = f"{config.output_dir}/{config.run_name}/{config.mode}/{encoder_name}/s{config.random_state}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Model saving
    model_path = f"{run_dir}/model.pth"
    # Temporarily override the trainer's save directory
    original_save_dir = trainer.model_save_directory
    trainer.model_save_directory = Path(run_dir)
    trainer.save_model("model.pth")
    trainer.model_save_directory = original_save_dir  # Restore original
    logger.info(f"Model saved: {model_path}")
    
    # Configuration saving
    config_path = f"{run_dir}/config.json"
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2)
    logger.info(f"Configuration saved: {config_path}")
    
    # Scaler saving (now with consistent scalers including target scalers)
    scaler_path = f"{run_dir}/scalers.pkl"
    # Scaler saving (dict type)
    scalers = {
        "pk_scaler": pk_scaler,
        "pd_scaler": pd_scaler,
        "pk_target_scaler": pk_target_scaler,
        "pd_target_scaler": pd_target_scaler,
    }
    with open(scaler_path, "wb") as f:
        pickle.dump(scalers, f)
    logger.info(f"Consistent scalers (including target scalers) saved: {scaler_path}")

    # Feature list saving
    features_path = f"{run_dir}/features.pkl"
    with open(features_path, "wb") as f:
        pickle.dump({"pk": pk_feats, "pd": pd_feats}, f)
    logger.info(f"Feature lists saved: {features_path}")
    
    # Save split information for reproducibility
    split_info = {
        'pk_train_subjects': sorted(pk_splits['train']['ID'].unique().tolist()),
        'pk_val_subjects': sorted(pk_splits['val']['ID'].unique().tolist()),
        'pk_test_subjects': sorted(pk_splits['test']['ID'].unique().tolist()),
        'pd_train_subjects': sorted(pd_splits['train']['ID'].unique().tolist()),
        'pd_val_subjects': sorted(pd_splits['val']['ID'].unique().tolist()),
        'pd_test_subjects': sorted(pd_splits['test']['ID'].unique().tolist()),
        'split_method': 'deterministic',
        'random_state': config.random_state
    }
    
    split_info_path = f"{run_dir}/split_info.json"
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"Split information saved: {split_info_path}")
    
    # Results saving
    results_path = f"{run_dir}/results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved: {results_path}")
    
    # Create symlink for backward compatibility
    try:
        # Create symlinks in the old structure for easy access
        old_model_path = f"{config.output_dir}/{config.run_name}/{config.mode}/{encoder_name}/s{config.random_state}/model.pth"
        os.makedirs(os.path.dirname(old_model_path), exist_ok=True)
        if not os.path.exists(old_model_path):
            os.symlink(os.path.abspath(model_path), old_model_path)
            logger.info(f"Symlink created: {old_model_path} -> {model_path}")
    except Exception as e:
        logger.warning(f"Could not create symlink: {e}")
    
    logger.info(f"All outputs saved to: {run_dir}")
    
    # === 8. Auto Visualization ===
    logger.info(f"\n=== 8. Auto Visualization ===")
    
    visualizer = AutoVisualizer(str(run_dir), device)
    visualizer.run_auto_visualization()
    logger.info(" Auto Visualization completed!")
    
    logger.info("=== Execution completed with deterministic splitting ===")
    return 0


if __name__ == "__main__":
    exit(main())
