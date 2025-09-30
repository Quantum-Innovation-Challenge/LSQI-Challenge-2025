#!/usr/bin/env python3
"""
AutoVisualization Script
"""

import sys
import json
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.helpers import get_device
from utils.factory import create_model
from data.loaders import load_estdata, use_feature_engineering


class AutoVisualizer:
    def __init__(self, run_dir: str, device: str = "auto", tag: str = "all"):
        self.run_dir = Path(run_dir)
        self.tag = tag

        # Device 설정
        if isinstance(device, torch.device):
            if device.type == 'cuda':
                device_id = int(device.index) if device.index is not None else 0
            else:
                device_id = 0
        elif isinstance(device, str):
            if device == 'auto':
                device_id = 0
            elif device == 'cpu':
                device_id = 0
            elif device.startswith('cuda:'):
                device_id = int(device.split(':')[1])
            else:
                device_id = 0
        else:
            device_id = device
        self.device = get_device(device_id)

        # File paths
        self.model_path = self.run_dir / "model.pth"
        self.config_path = self.run_dir / "config.json"
        self.scalers_path = self.run_dir / "scalers.pkl"
        self.split_info_path = self.run_dir / "split_info.json"
        self.results_path = self.run_dir / "results.json"
        self.features_path = self.run_dir / "features.pkl"

        # Loaded data
        self.config = None
        self.model = None
        self.scalers = None
        self.split_info = None
        self.results = None

        print(f" AutoVisualizer initialized: {self.run_dir}")

    def load_experiment_data(self):
        print(" Loading experiment data...")
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        print(f"    Config loaded")

        with open(self.scalers_path, "rb") as f:
            self.scalers = pickle.load(f)
        print(f"    Scaler loaded: {list(self.scalers.keys())}")

        with open(self.split_info_path, 'r') as f:
            self.split_info = json.load(f)
        print(f"    Split info loaded")

        with open(self.results_path, 'r') as f:
            self.results = json.load(f)
        print(f"    Results loaded")

    def load_model(self):
        print(" Loading model...")
        from config import Config
        cfg = Config()
        for k, v in self.config.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        with open(self.features_path, "rb") as f:
            feats = pickle.load(f)
        pk_feats = feats["pk"]
        pd_feats = feats["pd"]

        self.model = create_model(cfg, None, pk_feats, pd_feats)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("    Loaded model state_dict")
        else:
            self.model.load_state_dict(checkpoint)
            print("    Loaded model weights")

        self.model.to(self.device).eval()
        print(f"    Model ready on {self.device}")

    def prepare_datasets(self):
        print(" Dataset preparation...")
        df_all, df_obs, df_dose = load_estdata(self.config['csv_path'])

        if self.config.get('use_fe', False):
            df_final, pk_feats, pd_feats = use_feature_engineering(
                df_obs=df_obs,
                df_dose=df_dose,
                use_perkg=self.config.get('use_perkg', False),
                target="dv",
                allow_future_dose=self.config.get('allow_future_dose', True),
                time_windows=self.config.get('time_windows', [24, 48, 72, 96, 120, 144, 168])
            )
        else:
            df_final = df_obs.copy()
            pk_feats = ['BW', 'COMED', 'DOSE', 'TIME']
            pd_feats = ['BW', 'COMED', 'DOSE', 'TIME']

        pk_df = df_final[df_final["DVID"] == 1].copy()
        pd_df = df_final[df_final["DVID"] == 2].copy()

        meta_info = df_obs[['ID', 'BW', 'COMED', 'DOSE']].drop_duplicates()
        for col in ['BW', 'COMED', 'DOSE']:
            pk_df = pk_df.drop(columns=col, errors='ignore')
            pd_df = pd_df.drop(columns=col, errors='ignore')
        pk_df = pk_df.merge(meta_info, on='ID', how='left')
        pd_df = pd_df.merge(meta_info, on='ID', how='left')

        # split info 기반 filtering
        pk_train_subjects = self.split_info['pk_train_subjects']
        pd_train_subjects = self.split_info['pd_train_subjects']
        pk_val_subjects = self.split_info['pk_val_subjects']
        pd_val_subjects = self.split_info['pd_val_subjects']
        pk_test_subjects = self.split_info['pk_test_subjects']
        pd_test_subjects = self.split_info['pd_test_subjects']

        pk_train = pk_df[pk_df['ID'].isin(pk_train_subjects)].copy()
        pd_train = pd_df[pd_df['ID'].isin(pd_train_subjects)].copy()
        pk_val = pk_df[pk_df['ID'].isin(pk_val_subjects)].copy()
        pd_val = pd_df[pd_df['ID'].isin(pd_val_subjects)].copy()
        pk_test = pk_df[pk_df['ID'].isin(pk_test_subjects)].copy()
        pd_test = pd_df[pd_df['ID'].isin(pd_test_subjects)].copy()

        print(f"   PK Train: {pk_train.shape}, Val: {pk_val.shape}, Test: {pk_test.shape}")
        print(f"   PD Train: {pd_train.shape}, Val: {pd_val.shape}, Test: {pd_test.shape}")

        return pk_train, pd_train, pk_val, pd_val, pk_test, pd_test, pk_feats, pd_feats

    def make_predictions(self, data, features, target_scaler, data_type):
        print(f" {data_type.upper()} prediction...")
        X = data[features].values
        y_true = data['DV'].values
        scaler = self.scalers['pk_scaler'] if data_type == 'pk' else self.scalers['pd_scaler']
        X_scaled = scaler.transform(X)

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.zeros((len(y_true), 1)).to(self.device)

        self.model.eval()
        with torch.no_grad():
            results = self.model({data_type: {'x': X_tensor, 'y': y_tensor}})
            z = results[data_type]['z']
            preds = results[data_type]['pred']
            y_pred = target_scaler.inverse_transform(preds.cpu().numpy().reshape(-1, 1)).flatten()
            pred_clf, prob_clf = None, None
            if self.config.get("clf_finetune", False) or self.config.get("finetune_clf", False):
                outputs_clf = self.model._forward_clf(z)
                logits = outputs_clf["pred_clf"]
                pred_clf = torch.argmax(logits, dim=1).cpu().numpy()
                prob_clf = torch.softmax(logits, dim=1).cpu().numpy()[:, 1]

        meta_cols = [c for c in ['DOSE', 'BW', 'COMED'] if c in data.columns]
        extra_cols = [c for c in data.columns if c.startswith("DOSE_SUM_PREV") or c.startswith("DECAY_HL") or c.startswith("CUM_DOSE")]
        cols_to_keep = ['ID', 'TIME', 'DV'] + meta_cols + extra_cols

        results_df = data[cols_to_keep].copy()
        results_df['PRED'] = y_pred
        results_df['ERROR'] = results_df['PRED'] - results_df['DV']
        results_df['ABS_ERROR'] = results_df['ERROR'].abs()

        if pred_clf is not None:
            results_df['PRED_CLF'] = pred_clf
            results_df['PROB_CLF'] = prob_clf

        print(f"    {data_type.upper()} prediction completed")
        return results_df

    # -----------------------------
    # Visualization functions
    # -----------------------------
    def _plot_predictions_vs_actual(self, results, data_type, viz_dir):
        plt.figure(figsize=(10, 8))
        plt.scatter(results['DV'], results['PRED'], alpha=0.6, s=50)
        min_val = min(results['DV'].min(), results['PRED'].min())
        max_val = max(results['DV'].max(), results['PRED'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        from sklearn.metrics import r2_score
        r2 = r2_score(results['DV'], results['PRED'])
        plt.xlabel(f'Actual {data_type} Values')
        plt.ylabel(f'Predicted {data_type} Values')
        plt.title(f'{data_type} Predictions vs Actual Values\nR² = {r2:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        viz_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_dir / f'{data_type.lower()}_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pd_by_id_time(self, pd_results, viz_dir):
        n_subj = pd_results['ID'].nunique()

        if n_subj <= 4:
            n_cols = n_subj
        else:
            n_cols = min(6, n_subj)

        n_rows = math.ceil(n_subj / n_cols)

        n_rows = math.ceil(n_subj / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_subj == 1:
            axes = [axes] 
        else:
            axes = axes.flatten()

        for i, subject_id in enumerate(sorted(pd_results['ID'].unique())):
            subject_data = pd_results[pd_results['ID'] == subject_id].sort_values('TIME')
            ax = axes[i]
            ax.plot(subject_data['TIME'], subject_data['DV'], 'o-', label='Actual', linewidth=2, markersize=6)
            ax.plot(subject_data['TIME'], subject_data['PRED'], 's--', label='Predicted', linewidth=2, markersize=6)
            dose_val = subject_data['DOSE'].iloc[0] if 'DOSE' in subject_data.columns else None
            bw_val = subject_data['BW'].iloc[0] if 'BW' in subject_data.columns else None
            ax.set_title(f'ID: {subject_id} | DOSE: {dose_val} | BW: {bw_val}')
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('PD Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # unused subplot
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle('PD Predictions by Subject and Time', fontsize=16)
        plt.tight_layout()
        viz_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_dir / 'pd_by_id_time.png', dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_error_distribution(self, pk_results, pd_results, viz_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.hist(pk_results['ERROR'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('PK Prediction Errors')
        ax1.set_xlabel('Error (Predicted - Actual)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        ax2.hist(pd_results['ERROR'], bins=30, alpha=0.7, color='red', edgecolor='black')
        ax2.set_title('PD Prediction Errors')
        ax2.set_xlabel('Error (Predicted - Actual)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        viz_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_metrics(self, pk_results, pd_results, viz_dir):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        pk_metrics = {
            'RMSE': np.sqrt(mean_squared_error(pk_results['DV'], pk_results['PRED'])),
            'MAE': mean_absolute_error(pk_results['DV'], pk_results['PRED']),
            'R²': r2_score(pk_results['DV'], pk_results['PRED'])
        }
        pd_metrics = {
            'RMSE': np.sqrt(mean_squared_error(pd_results['DV'], pd_results['PRED'])),
            'MAE': mean_absolute_error(pd_results['DV'], pd_results['PRED']),
            'R²': r2_score(pd_results['DV'], pd_results['PRED'])
        }
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        metrics_names = list(pk_metrics.keys())
        metrics_values = list(pk_metrics.values())
        ax1.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax1.set_title('PK Performance Metrics')
        ax1.set_ylabel('Value')
        for i, v in enumerate(metrics_values):
            ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
        metrics_values = list(pd_metrics.values())
        ax2.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax2.set_title('PD Performance Metrics')
        ax2.set_ylabel('Value')
        for i, v in enumerate(metrics_values):
            ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
        plt.tight_layout()
        viz_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        with open(viz_dir / 'performance_metrics.txt', 'w') as f:
            f.write("=== Performance Metrics ===\n\n")
            f.write("PK Metrics:\n")
            for metric, value in pk_metrics.items():
                f.write(f"  {metric}: {value:.6f}\n")
            f.write("\nPD Metrics:\n")
            for metric, value in pd_metrics.items():
                f.write(f"  {metric}: {value:.6f}\n")


    def _plot_pd_by_id_time_ver2(self, pd_results, viz_dir, threshold=3.3):
        """PD by ID/Time with dose timing + classification marker + threshold line"""
        n_subj = pd_results['ID'].nunique()

        if n_subj <= 4:
            n_cols = n_subj
        else:
            n_cols = min(6, n_subj)

        n_rows = math.ceil(n_subj / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_subj == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        handles, labels = None, None

        for i, subject_id in enumerate(sorted(pd_results['ID'].unique())):
            subject_data = pd_results[pd_results['ID'] == subject_id].sort_values('TIME')
            ax = axes[i]

            # 기본 선
            ax.plot(subject_data['TIME'], subject_data['DV'], 'o-', label='Actual', linewidth=2, markersize=6)
            ax.plot(subject_data['TIME'], subject_data['PRED'], 's--', label='Predicted', linewidth=2, markersize=6)

            # threshold 라인
            ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Threshold={threshold}')

            # 🔹 classification marker (강조)
            if 'PRED_CLF' in subject_data.columns:
                clf_pos = subject_data[subject_data['PRED_CLF'] == 1]
                if not clf_pos.empty:
                    ax.scatter(
                        clf_pos['TIME'], clf_pos['PRED'],
                        color='red', s=120, marker='X', edgecolor='black', linewidth=1.2,
                        label='Pred Clf=1'
                    )

            ax.set_ylim(bottom=0)
            dose_val = subject_data['DOSE'].iloc[0] if 'DOSE' in subject_data.columns else None
            bw_val   = subject_data['BW'].iloc[0] if 'BW' in subject_data.columns else None
            ax.set_title(f'ID: {subject_id} | DOSE: {dose_val} | BW: {bw_val}')
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('PD Value')
            ax.grid(True, alpha=0.3)

            if handles is None and labels is None:
                handles, labels = ax.get_legend_handles_labels()

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        # 🔹 범례 (겹치지 않게 아래쪽에 크게 표시)
        fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=10, frameon=False)
        plt.suptitle('PD by Subject (with Threshold + Clf marker)', fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        viz_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_dir / 'pd_by_id_time_ver2.png', dpi=300, bbox_inches='tight')
        plt.close()


    def create_visualizations(self, pk_results, pd_results, tag="train"):
        print(f" Creating visualizations for {tag}...")

        if tag == "train":
            viz_dir = self.run_dir / "visualizations_train"
        elif tag == "valid":
            viz_dir = self.run_dir / "visualizations_valid"
        elif tag == "test":
            viz_dir = self.run_dir / "visualizations_test"
        elif tag == "all":
            viz_dir = self.run_dir / "visualizations_all"
        else:
            viz_dir = self.run_dir / "visualizations" / tag

        viz_dir.mkdir(parents=True, exist_ok=True)

        self._plot_predictions_vs_actual(pk_results, "PK", viz_dir)
        self._plot_predictions_vs_actual(pd_results, "PD", viz_dir)
        self._plot_pd_by_id_time(pd_results, viz_dir)
        self._plot_error_distribution(pk_results, pd_results, viz_dir)
        self._plot_performance_metrics(pk_results, pd_results, viz_dir)

        # ver2 실행 (조건부)
        if 'PRED_CLF' in pd_results.columns:
            self._plot_pd_by_id_time_ver2(pd_results, viz_dir, threshold=3.3)

        print(f"    Visualizations completed: {viz_dir}")


    def run_auto_visualization(self):
        print(" Auto visualization started!")
        try:
            self.load_experiment_data()
            self.load_model()
            pk_train, pd_train, pk_val, pd_val, pk_test, pd_test, pk_feats, pd_feats = self.prepare_datasets()

            if self.tag == "train":
                pk_results = self.make_predictions(pk_train, pk_feats, self.scalers['pk_target_scaler'], 'pk')
                pd_results = self.make_predictions(pd_train, pd_feats, self.scalers['pd_target_scaler'], 'pd')
                self.create_visualizations(pk_results, pd_results, tag="train")

            elif self.tag == "valid":
                pk_results = self.make_predictions(pk_val, pk_feats, self.scalers['pk_target_scaler'], 'pk')
                pd_results = self.make_predictions(pd_val, pd_feats, self.scalers['pd_target_scaler'], 'pd')
                self.create_visualizations(pk_results, pd_results, tag="valid")

            elif self.tag == "test":
                pk_results = self.make_predictions(pk_test, pk_feats, self.scalers['pk_target_scaler'], 'pk')
                pd_results = self.make_predictions(pd_test, pd_feats, self.scalers['pd_target_scaler'], 'pd')
                self.create_visualizations(pk_results, pd_results, tag="test")

            elif self.tag == "all":
                # Train
                pk_results = self.make_predictions(pk_train, pk_feats, self.scalers['pk_target_scaler'], 'pk')
                pd_results = self.make_predictions(pd_train, pd_feats, self.scalers['pd_target_scaler'], 'pd')
                self.create_visualizations(pk_results, pd_results, tag="train")
                # Valid
                pk_results = self.make_predictions(pk_val, pk_feats, self.scalers['pk_target_scaler'], 'pk')
                pd_results = self.make_predictions(pd_val, pd_feats, self.scalers['pd_target_scaler'], 'pd')
                self.create_visualizations(pk_results, pd_results, tag="valid")
                # Test
                pk_results = self.make_predictions(pk_test, pk_feats, self.scalers['pk_target_scaler'], 'pk')
                pd_results = self.make_predictions(pd_test, pd_feats, self.scalers['pd_target_scaler'], 'pd')
                self.create_visualizations(pk_results, pd_results, tag="test")

            print(" Auto visualization completed!")

        except Exception as e:
            print(f" Error occurred: {e}")
            import traceback
            traceback.print_exc()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Auto visualization script')
    parser.add_argument('--run_dir', type=str, required=True,
                       help='Experiment result directory path')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device setting (auto, cuda, cpu)')
    parser.add_argument('--tag', type=str, default='all',
                       choices=['train', 'valid', 'test', 'all'],
                       help='Which dataset split to visualize')
    args = parser.parse_args()
    visualizer = AutoVisualizer(args.run_dir, args.device, args.tag)
    visualizer.run_auto_visualization()


if __name__ == "__main__":
    main()
