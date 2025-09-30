from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import math

# =========================================================
# prepare data
# =========================================================
def load_estdata(path: str = "EstData.csv", *, standardize_cols: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return (df_all, df_obs, df_dose).
    - df_all: original
    - df_obs: observed (EVID==0 & MDV==0)
    - df_dose: dosing (EVID==1)
    """
    df = pd.read_csv(path)
    if standardize_cols:
        df.columns = [c.strip().upper() for c in df.columns]
    df_obs = df[(df['EVID'] == 0) & (df['MDV'] == 0)].copy()
    df_dose = df[df['EVID'] == 1].copy()
    return df, df_obs, df_dose


# =========================================================
# PK/PD dataframe separation
# =========================================================
def separate_pkpd(df_obs: pd.DataFrame, df_dose: pd.DataFrame, use_fe: bool, use_perkg: bool):
    """Return (df_final, pk_df, pd_df, pd_features, pk_features)."""
    if use_fe:
        df_final, pk_feats, pd_feats = use_feature_engineering(df_obs, df_dose, use_perkg)
    else:
        print("not applying feature engineering.")
        df_final = df_obs.copy()
        pk_feats = ['BW', 'COMED', 'DOSE', 'TIME']
        pd_feats = ['BW', 'COMED', 'DOSE', 'TIME']

    print(df_final.head(3))
    pk_df = df_final[df_final['DVID'] == 1].copy()
    pd_df = df_final[df_final['DVID'] == 2].copy()
    pd_df = pd_df.copy()

    return df_final, pk_df, pd_df, pd_feats, pk_feats

# =========================================================
# Time window analysis and selection
# =========================================================
def analyze_dose_patterns(dose_df: pd.DataFrame, verbose: bool = False) -> dict:
    """Analyze dose patterns to determine optimal time windows"""
    if dose_df.empty:
        if verbose:
            print("No dose data found, using default windows: [24, 48, 72, 96, 120, 144, 168]")
        return {"avg_interval": 24, "max_interval": 48, "dose_frequency": "daily"}
    
    # Calculate dose intervals per subject
    dose_intervals = []
    for subject_id, subject_doses in dose_df.groupby('ID'):
        times = subject_doses['TIME'].sort_values()
        intervals = times.diff().dropna()
        dose_intervals.extend(intervals.tolist())
    
    if not dose_intervals:
        if verbose:
            print("No dose intervals found, using default windows: [24, 48, 72, 96, 120, 144, 168]")
        return {"avg_interval": 24, "max_interval": 48, "dose_frequency": "daily"}
    
    avg_interval = np.mean(dose_intervals)
    max_interval = np.max(dose_intervals)
    median_interval = np.median(dose_intervals)
    
    # Determine dose frequency pattern
    if avg_interval < 8:
        frequency = "frequent"  # 8 hours apart
    elif avg_interval < 16:
        frequency = "twice_daily"  # 12 hours apart
    elif avg_interval < 32:
        frequency = "daily"  # 24 hours apart
    else:
        frequency = "sparse"  # 32 hours apart
    
    return {
        "avg_interval": avg_interval,
        "max_interval": max_interval,
        "median_interval": median_interval,
        "dose_frequency": frequency,
        "dose_intervals": dose_intervals
    }

def get_optimal_time_windows(dose_df: pd.DataFrame, custom_windows: list = None, verbose: bool = False) -> list:
    """Get optimal time windows based on data analysis or custom settings"""
    
    if custom_windows is not None:
        return custom_windows
    
    # Default windows for EstData.csv (24 hours apart)
    default_windows = [24, 48, 72, 96, 120, 144, 168]
    
    # If no dose data, return default
    if dose_df.empty:
        if verbose:
            print(f"No dose data found, using default windows: {default_windows}")
        return default_windows
    
    # Analyze dose patterns
    pattern_info = analyze_dose_patterns(dose_df)
    avg_interval = pattern_info["avg_interval"]
    max_interval = pattern_info["max_interval"]
    frequency = pattern_info["dose_frequency"]
    
    if verbose:
        print(f"Dose pattern analysis: avg_interval={avg_interval:.2f}h, frequency={frequency}")
    
    # Select windows based on analysis
    if frequency == "frequent":  # 8 hours apart
        windows = [6, 12, 24, 48, 72]
    elif frequency == "twice_daily":  # 12 hours apart
        windows = [12, 24, 48, 72, 96]
    elif frequency == "daily":  # 24 hours apart
        windows = [24, 48, 72, 96, 120, 144, 168]
    elif frequency == "sparse":  # 32 hours apart
        windows = [24, 48, 72, 96, 120, 144, 168, 240, 336]  # Maximum 2 weeks
    else:
        # Default fallback (EstData.csv)
        windows = default_windows
    
    # Adjust windows based on actual data
    # For daily dosing (24h), allow up to 1 week (168h) for cumulative effects
    if frequency == "daily" and avg_interval == 24.0:
        # Keep all daily windows up to 1 week for cumulative effects
        max_reasonable_window = 168  # 1 week
    else:
        # For other patterns, use 3x max interval
        max_reasonable_window = max_interval * 3
    
    windows = [w for w in windows if w <= max_reasonable_window]
    
    # Ensure we have at least 3 windows
    if len(windows) < 3:
        windows = default_windows
    
    return sorted(windows)

# =========================================================
# FE: dose history -> additional features (leakage-safe)
# =========================================================
def features_from_dose_history(
    obs_df: pd.DataFrame,
    dose_df: pd.DataFrame,
    add_pk_baseline: bool = False,
    add_pd_delta: bool = False,   
    target: str = "dv",           
    allow_future_dose: bool = False,
    time_windows: list = None,
    add_decay_features: bool = True,
    half_lives: list = [24, 48, 72],   # hours
    verbose: bool = False
) -> pd.DataFrame:
    required_obs = {"ID", "TIME", "DV", "DVID"}
    required_dose = {"ID", "TIME", "AMT"}
    miss_obs = required_obs - set(obs_df.columns)
    miss_dose = required_dose - set(dose_df.columns)
    if miss_obs:
        raise ValueError(f"obs_df missing columns: {sorted(miss_obs)}")
    if miss_dose:
        raise ValueError(f"dose_df missing columns: {sorted(miss_dose)}")

    if str(target).lower() == "dv" and add_pd_delta:
        raise ValueError("Target is 'DV' but add_pd_delta=True -> data leakage risk.")

    out = obs_df.copy()

    # Pre-group doses per subject (sorted by time)
    dose_by_id = {
        i: d.sort_values("TIME")[["TIME", "AMT"]].to_numpy()
        for i, d in dose_df.groupby("ID", sort=False)
    }

    # Get optimal time windows
    optimal_windows = get_optimal_time_windows(dose_df, time_windows, verbose)
    if verbose:
        print(f"Using time windows: {optimal_windows} hours")
    
    tsld_list, last_amt_list = [], []
    ndose_list, cumdose_list = [], []
    ttnext_list = []
    
    # Dynamic window sum lists
    window_sum_lists = {f"sum{int(w)}h": [] for w in optimal_windows}
    # Decay feature lists
    decay_lists = {f"DECAY_HL{hl}h": [] for hl in half_lives} if add_decay_features else {}

    def window_sum(d_times, csum, t_start, t_end):
        """Cumulative dose in the window (t_start, t_end], inclusive on right)."""
        r = np.searchsorted(d_times, t_end, side="right") - 1
        l = np.searchsorted(d_times, t_start, side="right") - 1
        if r < 0:
            return 0.0
        left_cum = 0.0 if l < 0 else float(csum[l])
        right_cum = float(csum[r])
        return right_cum - left_cum

    # Build row-wise features
    for i, g in out.groupby("ID", sort=False):
        times = g["TIME"].to_numpy()
        if i not in dose_by_id:
            n = len(times)
            tsld_list += [np.nan] * n
            last_amt_list += [0.0] * n
            ndose_list += [0] * n
            cumdose_list += [0.0] * n
            ttnext_list += [np.nan] * n
            for window_key in window_sum_lists:
                window_sum_lists[window_key] += [0.0] * n
            if add_decay_features:
                for key in decay_lists:
                    decay_lists[key] += [0.0] * n
            continue

        dmat = dose_by_id[i]
        d_times = dmat[:, 0]
        d_amts = dmat[:, 1]
        csum = np.cumsum(d_amts)

        for t in times:
            # last dose info up to time t (inclusive)
            idx_last = np.searchsorted(d_times, t, side="right") - 1
            if idx_last >= 0:
                t_last = d_times[idx_last]
                last_amt = float(d_amts[idx_last])
                ndoses = int(idx_last + 1)
                cumdose = float(csum[idx_last])
                tsld = float(t - t_last)
            else:
                last_amt = 0.0
                ndoses = 0
                cumdose = 0.0
                tsld = np.nan

            if allow_future_dose:
                idx_next = idx_last + 1
                if idx_next < len(d_times):
                    t_next = d_times[idx_next]
                    ttnext = float(t_next - t)
                else:
                    ttnext = np.nan
            else:
                ttnext = np.nan

            # rolling dose sums
            for window in optimal_windows:
                window_key = f"sum{int(window)}h"
                window_sum_value = window_sum(d_times, csum, t - window, t)
                window_sum_lists[window_key].append(window_sum_value)

            # decay features
            if add_decay_features:
                for hl in half_lives:
                    k = np.log(2) / hl
                    decay_val = np.exp(-k * tsld) if tsld == tsld else 0.0
                    decay_lists[f"DECAY_HL{hl}h"].append(decay_val)

            tsld_list.append(tsld)
            last_amt_list.append(last_amt)
            ndose_list.append(ndoses)
            cumdose_list.append(cumdose)
            ttnext_list.append(ttnext)

    # Assign engineered columns
    out["TSLD"] = tsld_list
    out["LAST_DOSE_AMT"] = last_amt_list
    out["N_DOSES_UP_TO_T"] = ndose_list
    out["CUM_DOSE_UP_TO_T"] = cumdose_list
    out["TIME_TO_NEXT_DOSE"] = ttnext_list
    out["LAST_DOSE_TIME"] = out["TIME"] - out["TSLD"]

    # TIME transformations
    if verbose:
        print("  + TIME feature enhancement applied (essential transformations)")
    out["TIME_SQUARED"] = out["TIME"] ** 2
    out["TIME_LOG"] = np.log1p(out["TIME"])

    # Window sum features
    for window_key, window_values in window_sum_lists.items():
        hour_value = window_key.replace("sum", "").replace("h", "H")
        column_name = f"DOSE_SUM_PREV{hour_value}"
        out[column_name] = window_values

    # Decay features
    if add_decay_features:
        for col, vals in decay_lists.items():
            out[col] = vals
            if verbose:
                print(f"  + decay feature added: {col}")

    # Baselines
    base_by_id_dvid = (
        out.sort_values(["ID", "TIME"])
           .groupby(["ID", "DVID"])["DV"]
           .transform(lambda s: s.iloc[0] if len(s) else np.nan)
    )
    if add_pd_delta:
        out["PD_BASELINE"] = np.where(out["DVID"].eq(2), base_by_id_dvid, np.nan)
        out["PD_DELTA"] = np.where(out["DVID"].eq(2), out["DV"] - out["PD_BASELINE"], np.nan)
        if verbose:
            print("WARNING: PD_DELTA target uses PD_BASELINE which may cause data leakage!")

    if add_pk_baseline:
        out["PK_BASELINE"] = np.where(out["DVID"].eq(1), base_by_id_dvid, np.nan)
        out["PK_DELTA"] = np.where(out["DVID"].eq(1), out["DV"] - out["PK_BASELINE"], np.nan)
        if verbose:
            print("WARNING: PK_DELTA target uses PK_BASELINE which may cause data leakage!")

    return out



def add_population_baseline(df_obs: pd.DataFrame, baseline_type: str = "median", baseline_value: float = None) -> pd.DataFrame:
    """
    Add population-based baseline for new participants.
    This avoids data leakage by using population statistics instead of individual baselines.
    """
    out = df_obs.copy()
    
    # Calculate or use provided population baseline for PD (DVID==2)
    pd_data = out[out['DVID'] == 2]
    if not pd_data.empty:
        if baseline_value is not None:
            # Use provided baseline value (for inference)
            pop_pd_baseline = baseline_value
            print(f"Using provided population PD baseline: {pop_pd_baseline:.4f}")
        else:
            # Calculate baseline from data (for training)
            if baseline_type == "median":
                pop_pd_baseline = pd_data['DV'].median()
            elif baseline_type == "mean":
                pop_pd_baseline = pd_data['DV'].mean()
            elif baseline_type == "mode":
                pop_pd_baseline = pd_data['DV'].mode().iloc[0] if not pd_data['DV'].mode().empty else pd_data['DV'].median()
            else:
                pop_pd_baseline = pd_data['DV'].median()
            print(f"Calculated population PD baseline ({baseline_type}): {pop_pd_baseline:.4f}")
        
        # Add population baseline for all PD observations
        out["PD_POPULATION_BASELINE"] = np.where(out["DVID"].eq(2), pop_pd_baseline, np.nan)
    
    return out


def calculate_population_baseline(df_obs: pd.DataFrame, baseline_type: str = "median") -> float:
    """
    Calculate population baseline value for saving/loading.
    """
    pd_data = df_obs[df_obs['DVID'] == 2]
    if pd_data.empty:
        return 0.0
    
    if baseline_type == "median":
        return float(pd_data['DV'].median())
    elif baseline_type == "mean":
        return float(pd_data['DV'].mean())
    elif baseline_type == "mode":
        mode_result = pd_data['DV'].mode()
        return float(mode_result.iloc[0]) if not mode_result.empty else float(pd_data['DV'].median())
    else:
        return float(pd_data['DV'].median())


# =========================================================
# Feature Engineering (build feature lists with leakage guards)
# =========================================================
def use_feature_engineering(
    df_obs: pd.DataFrame,
    df_dose: pd.DataFrame,
    use_perkg: bool,
    *,
    target: str = "dv",
    allow_future_dose: bool = False,
    time_windows: list = None,
    add_decay_features: bool = True,
    half_lives: list = [24, 48, 72]
):
    print("applying feature engineering (leakage-safe).")
    df_final = features_from_dose_history(
        obs_df=df_obs,
        dose_df=df_dose,
        add_pk_baseline=False,
        add_pd_delta=(str(target).lower() == "delta"),
        target=target,
        allow_future_dose=allow_future_dose,
        time_windows=time_windows,
        add_decay_features=add_decay_features,
        half_lives=half_lives
    )

    base_feats = [
        'BW', 'COMED', 'DOSE', 'TIME',
        'TSLD', 'LAST_DOSE_TIME', 'LAST_DOSE_AMT',
        'N_DOSES_UP_TO_T', 'CUM_DOSE_UP_TO_T',
        'TIME_SQUARED', 'TIME_LOG'
    ]
    
    # Add WEEKLY feature if it exists in the data
    if 'WEEKLY' in df_final.columns:
        base_feats.append('WEEKLY')
        print("  + WEEKLY feature added for dosing pattern differentiation")

    # Add window features
    window_features = [col for col in df_final.columns if col.startswith('DOSE_SUM_PREV')]
    base_feats.extend(window_features)
    print(f"  + window features added: {window_features}")

    # Add decay features
    if add_decay_features:
        decay_features = [col for col in df_final.columns if col.startswith('DECAY_HL')]
        base_feats.extend(decay_features)
        print(f"  + decay features added: {decay_features}")

    if allow_future_dose:
        base_feats.append('TIME_TO_NEXT_DOSE')

    pk_features = base_feats.copy()
    pd_features = base_feats.copy()

    print("PD_BASELINE feature removed to prevent data leakage for new participants.")
    
    # Optional per-kg features
    if use_perkg:
        bw = df_final['BW'].replace(0, np.nan)
        perkg_cols = ['DOSE', 'LAST_DOSE_AMT', 'CUM_DOSE_UP_TO_T']
        perkg_cols.extend(window_features)
        added = []
        for col in perkg_cols:
            if col in df_final.columns:
                df_final[f'{col}_PER_KG'] = (df_final[col] / bw).fillna(0.0)
                added.append(f'{col}_PER_KG')
        if added:
            pk_features += added
            pd_features += added
            print("  + per-kg features added:", added)

    forbidden = {'DV', 'PD_DELTA', 'PK_DELTA'}
    assert not (forbidden & set(pk_features)), f"Leakage risk in PK features: {forbidden & set(pk_features)}"
    assert not (forbidden & set(pd_features)), f"Leakage risk in PD features: {forbidden & set(pd_features)}"

    df_final.fillna(0, inplace=True)
    return df_final, pk_features, pd_features


# =========================================================
# Custom Dataset
# =========================================================
class CustomDataset(Dataset):
    def __init__(self, X, y, y_clf=None, ids=None, times=None, mask=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.y_clf = torch.tensor(y_clf, dtype=torch.float32)
        self.ids = None if ids is None else torch.tensor(ids, dtype=torch.long)
        self.times = None if times is None else torch.tensor(times, dtype=torch.float32)
        self.mask = None if mask is None else torch.tensor(mask, dtype=torch.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        items = [self.X[idx], self.y[idx]]
        if self.mask is not None:  items.append(self.mask[idx])
        if self.ids is not None:   items.append(self.ids[idx])
        if self.times is not None: items.append(self.times[idx])
        return tuple(items)

# =========================================================
# Daily → Weekly dataset 변환
# =========================================================
def convert_to_weekly_dataset(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily dosing dataset into weekly dosing dataset.
    Rule: Seven days cumulative dose is accumulated on the first day (week start), and the other six days are AMT=0.
    """
    df = df_all.copy()
    df_weekly = []

    for sid, g in df.groupby("ID"):
        g = g.sort_values("TIME").reset_index(drop=True)

        # dose/obs separation
        g_dose = g[g["EVID"] == 1].copy()
        g_obs = g[g["EVID"] == 0].copy()

        # week index calculation
        g_dose["WEEK"] = (g_dose["TIME"] // 168).astype(int)

        weekly_doses = []
        for w, grp in g_dose.groupby("WEEK"):
            week_start = w * 168
            total_amt = grp["AMT"].sum()

            # Record cumulative dose only on the first day (week_start)
            first_row = grp.iloc[0].copy()
            first_row["TIME"] = week_start
            first_row["AMT"] = total_amt
            weekly_doses.append(first_row)

            # The other six days are recorded as AMT=0
            for offset in [24, 48, 72, 96, 120, 144]:
                zero_row = grp.iloc[0].copy()
                zero_row["TIME"] = week_start + offset
                zero_row["AMT"] = 0.0
                weekly_doses.append(zero_row)

        g_dose_weekly = pd.DataFrame(weekly_doses)

        # obs is kept as is
        g_weekly = pd.concat([g_dose_weekly, g_obs], ignore_index=True)
        g_weekly = g_weekly.sort_values("TIME").reset_index(drop=True)

        # Weekly flag added
        g_weekly["WEEKLY"] = 1
        df_weekly.append(g_weekly)

    df_weekly = pd.concat(df_weekly, ignore_index=True)
    return df_weekly
