import os, time
import numpy as np
from numpy.lib.format import open_memmap
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()

import TrajectoryGenerator as TG  # your module

def generate_and_save(
    out_dir: str,
    target_n: int = 1_000_000,
    batch_size: int = 2048,
    seed: int = 123,
    time_limit_seconds: int = 3600,     # stop after this many seconds
    flush_every_seconds: int = 60       # flush periodically
):
    os.makedirs(out_dir, exist_ok=True)

    pk_T = len(TG.PK_TIMES)
    pd_T = len(TG.PD_TIMES)

    # ---- Create memmap .npy files sized to target_n (safe even if we stop early)
    X_pk_path = os.path.join(out_dir, "X_pk.npy")
    y_pk_cls_path = os.path.join(out_dir, "y_pk_cls.npy")
    y_pk_reg_path = os.path.join(out_dir, "y_pk_reg.npy")
    y_pk_mask_path = os.path.join(out_dir, "y_pk_mask.npy")

    X_pd_path = os.path.join(out_dir, "X_pd.npy")
    y_pd_cls_path = os.path.join(out_dir, "y_pd_cls.npy")
    y_pd_reg_path = os.path.join(out_dir, "y_pd_reg.npy")
    y_pd_mask_path = os.path.join(out_dir, "y_pd_mask.npy")

    X_pk = open_memmap(X_pk_path, mode="w+", dtype="float32", shape=(target_n, 2, pk_T))
    y_pk_cls = open_memmap(y_pk_cls_path, mode="w+", dtype="int16", shape=(target_n,))
    y_pk_reg = open_memmap(y_pk_reg_path, mode="w+", dtype="float32", shape=(target_n, TG.PK_PARAM_DIM))
    y_pk_mask = open_memmap(y_pk_mask_path, mode="w+", dtype="float32", shape=(target_n, TG.PK_PARAM_DIM))

    X_pd = open_memmap(X_pd_path, mode="w+", dtype="float32", shape=(target_n, 2, pd_T))
    y_pd_cls = open_memmap(y_pd_cls_path, mode="w+", dtype="int16", shape=(target_n,))
    y_pd_reg = open_memmap(y_pd_reg_path, mode="w+", dtype="float32", shape=(target_n, TG.PD_PARAM_DIM))
    y_pd_mask = open_memmap(y_pd_mask_path, mode="w+", dtype="float32", shape=(target_n, TG.PD_PARAM_DIM))

    start = time.time()
    last_flush = start
    n = 0
    batch_idx = 0

    while n < target_n:
        elapsed = time.time() - start
        if elapsed >= time_limit_seconds:
            print(f"[STOP] time limit reached at n={n} (elapsed={elapsed:.1f}s)")
            break

        n_batch = min(batch_size, target_n - n)

        # Generate paired PK+PD samples in one call
        data = TG.generate_pk_pd_datasets(N=n_batch, seed=seed + batch_idx)

        # Write slices (cast to float32 to reduce disk space and speed IO)
        X_pk[n:n+n_batch] = data["X_pk_2xT"].astype(np.float32, copy=False)
        y_pk_cls[n:n+n_batch] = data["y_pk_cls"].astype(np.int16, copy=False)
        y_pk_reg[n:n+n_batch] = data["y_pk_reg"].astype(np.float32, copy=False)
        y_pk_mask[n:n+n_batch] = data["y_pk_mask"].astype(np.float32, copy=False)

        X_pd[n:n+n_batch] = data["X_pd_2xT"].astype(np.float32, copy=False)
        y_pd_cls[n:n+n_batch] = data["y_pd_cls"].astype(np.int16, copy=False)
        y_pd_reg[n:n+n_batch] = data["y_pd_reg"].astype(np.float32, copy=False)
        y_pd_mask[n:n+n_batch] = data["y_pd_mask"].astype(np.float32, copy=False)

        n += n_batch
        batch_idx += 1

        # Periodic flush + metadata snapshot
        now = time.time()
        if now - last_flush >= flush_every_seconds:
            for arr in (X_pk, y_pk_cls, y_pk_reg, y_pk_mask, X_pd, y_pd_cls, y_pd_reg, y_pd_mask):
                arr.flush()
            np.savez(os.path.join(out_dir, "meta_partial.npz"),
                     n_written=n,
                     seed=seed,
                     pk_times=TG.PK_TIMES.astype(np.float32),
                     pd_times=TG.PD_TIMES.astype(np.float32))
            print(f"[FLUSH] n={n} elapsed={now-start:.1f}s")
            last_flush = now

    # Final flush + final meta
    for arr in (X_pk, y_pk_cls, y_pk_reg, y_pk_mask, X_pd, y_pd_cls, y_pd_reg, y_pd_mask):
        arr.flush()

    np.savez(os.path.join(out_dir, "meta.npz"),
             n_written=n,
             seed=seed,
             pk_times=TG.PK_TIMES.astype(np.float32),
             pd_times=TG.PD_TIMES.astype(np.float32))

    print(f"[DONE] wrote n={n} samples to {out_dir}")

# Example usage:
# generate_and_save("/pscratch/sd/b/by1997/pkpd/datasets/run1", target_n=1_000_000, time_limit_seconds=7200)

TOTAL = 10_000_000  # total across all ranks (change if you want 40M)
per_rank = int(TOTAL/50)


comm.Barrier()


generate_and_save(
    "./dataset/mpirank_{mpi}".format(mpi=mpirank),
    target_n=per_rank,
    seed=123 + mpirank,            # unique randomness per shard
    batch_size=1024,              # keep per-rank RAM bounded
    time_limit_seconds=11.8*3600,      # stop cleanly and save partial
    flush_every_seconds=60
)
    
comm.Barrier()



