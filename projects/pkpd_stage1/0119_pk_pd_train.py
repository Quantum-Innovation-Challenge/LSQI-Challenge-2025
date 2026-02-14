import os
from mpi4py import MPI
import long_run_start as LRS

comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()


comm.Barrier()

if mpirank == 0:
    best, ckpt = LRS.start_long_run(
        task="pk",
        base_dir="./dataset",
        splits_npz="./dataset/splits_90_5_5_seed123.npz",
        cfg_json="./configs/cfg_final_pk.json",
        out_dir="./long_runs/pk",
        run_name="PK_long",
        fit_n=200_000,
        norm_seed=1,
        device="cuda",
        refuse_if_exists=True,
        verbose=True,
    )

elif mpirank == 1:
    best, ckpt = LRS.start_long_run(
        task="pk",
        base_dir="./dataset",
        splits_npz="./dataset/splits_90_5_5_seed123.npz",
        cfg_json="./configs/cfg_final_pk_deep.json",
        out_dir="./long_runs/pk_deep",
        run_name="PK_long_deep",
        fit_n=200_000,
        norm_seed=1,
        device="cuda",
        refuse_if_exists=True,
        verbose=True,
    )

elif mpirank == 2:
    best, ckpt = LRS.start_long_run(
        task="pd",
        base_dir="./dataset",
        splits_npz="./dataset/splits_90_5_5_seed123.npz",
        cfg_json="./configs/cfg_final_pd.json",
        out_dir="./long_runs/pd",
        run_name="PD_long",
        fit_n=200_000,
        norm_seed=2,
        device="cuda",
        refuse_if_exists=True,
        verbose=True,
    )

elif mpirank == 3:
    best, ckpt = LRS.start_long_run(
        task="pd",
        base_dir="./dataset",
        splits_npz="./dataset/splits_90_5_5_seed123.npz",
        cfg_json="./configs/cfg_final_pd_deep.json",
        out_dir="./long_runs/pd_deep",
        run_name="PD_long_deep",
        fit_n=200_000,
        norm_seed=2,
        device="cuda",
        refuse_if_exists=True,
        verbose=True,
    )

comm.Barrier()
