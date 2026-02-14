# long_run_start.py
import os
import json
from typing import Optional

import numpy as np
import torch

import TrajectoryGenerator as TG
import data_and_training as DT


def start_long_run(
    *,
    task: str,                 # "pk" or "pd"
    base_dir: str,             # contains mpirank_*
    splits_npz: str,           # splits_90_5_5_seed123.npz
    cfg_json: str,             # JSON with cfg
    out_dir: str,              # where checkpoints/logs go
    run_name: str,             # e.g. "PD_long"
    fit_n: int = 200_000,
    norm_seed: int = 1,
    device: str = "cuda",
    refuse_if_exists: bool = False,
    verbose: bool = True,
):
    """
    Start training from scratch WITHOUT argparse.
    Returns (best_metrics_dict, best_checkpoint_path).
    """
    assert task in ("pk", "pd"), "task must be 'pk' or 'pd'"

    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"{run_name}_best.pt")
    if refuse_if_exists and os.path.exists(best_path):
        raise RuntimeError(f"Refusing to overwrite existing best checkpoint: {best_path}")

    # Load cfg
    with open(cfg_json, "r") as f:
        cfg = json.load(f)

    # Discover shards
    shards = DT.discover_mpirank_shards(base_dir)

    # Load splits
    spl = np.load(splits_npz)
    idx_tr = spl["idx_tr"].astype(np.int64)
    idx_va = spl["idx_va"].astype(np.int64)

    # Build memmaps + dataset
    if task == "pk":
        X = DT.ShardedMemmap(shards, "X_pk.npy")
        y_cls = DT.ShardedMemmap(shards, "y_pk_cls.npy")
        y_reg = DT.ShardedMemmap(shards, "y_pk_reg.npy")
        y_mask = DT.ShardedMemmap(shards, "y_pk_mask.npy")
        reg_dim = TG.PK_PARAM_DIM
        num_classes = 10
        times_t = torch.tensor(TG.PK_TIMES, dtype=torch.float32)
    else:
        X = DT.ShardedMemmap(shards, "X_pd.npy")
        y_cls = DT.ShardedMemmap(shards, "y_pd_cls.npy")
        y_reg = DT.ShardedMemmap(shards, "y_pd_reg.npy")
        y_mask = DT.ShardedMemmap(shards, "y_pd_mask.npy")
        reg_dim = TG.PD_PARAM_DIM
        num_classes = 10
        times_t = torch.tensor(TG.PD_TIMES, dtype=torch.float32)

    # Fit normalizers (train-only, deterministic if idx_tr + seed + fit_n fixed)
    x_norm, y_norm = DT.fit_normalizers(
        dataset_kind=task,
        shard_dirs=shards,
        train_idx=idx_tr,
        fit_n=fit_n,
        seed=norm_seed
    )

    ds = DT.ShardedMultiTaskSeqDataset(X, y_cls, y_reg, y_mask, x_norm, y_norm)

    # Device
    if device == "cuda" and not torch.cuda.is_available():
        if verbose:
            print("CUDA not available; falling back to CPU.")
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(device)

    times_t = times_t.to(torch_device)

    if verbose:
        print(f"[{run_name}] Starting from scratch")
        print(f"  task={task}, device={torch_device}, fit_n={fit_n}, norm_seed={norm_seed}")
        print(f"  out_dir={out_dir}")

    # Start training from scratch
    best, best_ckpt = DT.train_one_config(
        run_name=run_name,
        dataset=ds,
        idx_tr=idx_tr.tolist(),
        idx_va=idx_va.tolist(),
        num_classes=num_classes,
        reg_dim=reg_dim,
        device=torch_device,
        out_dir=out_dir,
        cfg=cfg,
        times_t=times_t,
        resume_training=False,
        resume_ckpt_path=None,
    )

    if verbose:
        print(f"[{run_name}] === DONE (START) ===")
        print("Best:", best)
        print("Best ckpt:", best_ckpt)

    return best, best_ckpt


# Optional CLI entrypoint (keep or delete)
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["pk", "pd"], required=True)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--splits_npz", required=True)
    ap.add_argument("--cfg_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--fit_n", type=int, default=200_000)
    ap.add_argument("--norm_seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--refuse_if_exists", action="store_true")
    args = ap.parse_args()

    start_long_run(
        task=args.task,
        base_dir=args.base_dir,
        splits_npz=args.splits_npz,
        cfg_json=args.cfg_json,
        out_dir=args.out_dir,
        run_name=args.run_name,
        fit_n=args.fit_n,
        norm_seed=args.norm_seed,
        device=args.device,
        refuse_if_exists=args.refuse_if_exists,
        verbose=True,
    )

