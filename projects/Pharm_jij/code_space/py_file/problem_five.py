#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare minimal doses at 90% vs 75% success
-------------------------------------------
Reads dose→success summary CSVs (mean curves) produced by prior MC/LSTM runs and
computes, per scenario:
  - Minimal dose for ≥90% success
  - Minimal dose for ≥75% success
  - Absolute reduction (mg) when going from 90% -> 75%
Works for both QD and QW scenarios.

If the curve does not reach the target within its grid, the script:
  - Enforces monotonicity (cumulative max over dose)
  - Returns an interpolated dose if the crossing is bracketed
  - Otherwise EXTRAPOLATES beyond the top of the grid using:
      1) linear extrapolation from the last *increasing* segment; or
      2) logistic (logit) least-squares fit as a fallback.
Outputs:
  - dose_target_comparison.csv
  - dose_target_comparison.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 140

# ----------------------- Default scenarios -----------------------
DEFAULT_SCENARIOS = [
    {"name": "Baseline QD",  "path": "mc_dose_success_summary.csv",
     "dose": "dose_mg",      "succ": "succ_mean", "regimen": "QD"},

    {"name": "Baseline QW",  "path": "lstm_weekly_dose_success_summary.csv",
     "dose": "weekly_dose_mg","succ": "succ_mean", "regimen": "QW"},

    {"name": "No-conmed QD", "path": "qd_no_comed_dose_success_summary.csv",
     "dose": "qd_dose_mg",   "succ": "succ_mean", "regimen": "QD"},

    {"name": "No-conmed QW", "path": "qw_no_comed_dose_success_summary.csv",
     "dose": "qw_dose_mg",   "succ": "succ_mean", "regimen": "QW"},
]

TARGET_STRICT = 0.90
TARGET_LOOSER = 0.75

# --------------------------- Helpers ---------------------------
def load_curve(path: str, dose_col: str, succ_col: str):
    df = pd.read_csv(path)
    if dose_col not in df.columns or succ_col not in df.columns:
        raise ValueError(f"CSV {path} must contain '{dose_col}' and '{succ_col}'.")
    d = pd.to_numeric(df[dose_col], errors="coerce").to_numpy()
    s = pd.to_numeric(df[succ_col], errors="coerce").to_numpy()
    m = np.isfinite(d) & np.isfinite(s)
    d, s = d[m], s[m]
    order = np.argsort(d)
    d, s = d[order], s[order]
    # enforce non-decreasing success vs dose (theory-consistent)
    s = np.maximum.accumulate(s)
    return d, s

def _interp_between(x0, y0, x1, y1, yt):
    # Linear interpolation to x at y=yt (assumes y1>y0, yt in [y0,y1])
    if y1 == y0:
        return x1  # degenerate; just take the upper x
    return x0 + (yt - y0) * (x1 - x0) / (y1 - y0)

def _linear_extrapolate(d, s, target):
    """Use the last *increasing* segment to linearly extrapolate to target."""
    inc_idx = np.where(np.diff(s) > 1e-9)[0]
    if inc_idx.size == 0:
        return np.nan, "fail"
    i = inc_idx[-1]
    x0, y0, x1, y1 = d[i], s[i], d[i+1], s[i+1]
    if y1 >= target:
        return _interp_between(x0, y0, x1, y1, target), "interp"
    if y1 <= y0 + 1e-9:
        return np.nan, "fail"
    slope = (y1 - y0) / max(1e-12, (x1 - x0))
    x_t = x1 + (target - y1) / max(1e-12, slope)
    return float(x_t), "extrap-linear"

def _logit(x):
    x = np.clip(x, 1e-6, 1-1e-6)
    return np.log(x / (1.0 - x))

def _logistic_extrapolate(d, s, target):
    """
    Fit logit(s) ~ a + b*d (least squares), solve for dose at target.
    Returns (dose, 'extrap-logit') or (np.nan, 'fail').
    """
    y = np.clip(s, 1e-6, 1-1e-6)
    z = _logit(y)
    X = np.vstack([np.ones_like(d), d]).T
    try:
        beta, *_ = np.linalg.lstsq(X, z, rcond=None)
        a, b = beta
        if abs(b) < 1e-12:
            return np.nan, "fail"
        zt = _logit(target)
        x_t = (zt - a) / b
        return float(x_t), "extrap-logit"
    except Exception:
        return np.nan, "fail"

def dose_at_target(dose: np.ndarray, succ: np.ndarray, target: float):
    """
    Minimal dose to achieve >= target using:
      1) in-range interpolation (preferred)
      2) linear extrapolation off last increasing segment
      3) logistic regression extrapolation
    Returns (dose_value, method_tag).
    """
    s = succ
    d = dose

    idx = np.where(s >= target)[0]
    if idx.size:
        i = idx[0]
        if i == 0:
            return float(d[0]), "grid"
        return float(_interp_between(d[i-1], s[i-1], d[i], s[i], target)), "grid"

    x_lin, tag = _linear_extrapolate(d, s, target)
    if np.isfinite(x_lin):
        return float(x_lin), tag

    x_log, tag = _logistic_extrapolate(d, s, target)
    if np.isfinite(x_log):
        return float(x_log), tag

    return np.nan, "fail"

# ----------------------------- Main -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare minimal doses at 90% vs 75% success from dose→success CSVs.")
    parser.add_argument("--out-csv", default="dose_target_comparison.csv", help="Output CSV filename")
    parser.add_argument("--out-png", default="dose_target_comparison.png", help="Output PNG filename")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    scenarios = DEFAULT_SCENARIOS
    rows, missing = [], []

    for sc in scenarios:
        path = sc["path"]
        if not os.path.exists(path):
            missing.append(sc["name"])
            continue
        try:
            d, s = load_curve(path, sc["dose"], sc["succ"])
            d90, m90 = dose_at_target(d, s, TARGET_STRICT)
            d75, m75 = dose_at_target(d, s, TARGET_LOOSER)

            rows.append({
                "scenario": sc["name"],
                "regimen": sc["regimen"],
                "dose_col": sc["dose"],
                "path": path,
                "dose_90pct_mg": d90,
                "dose_75pct_mg": d75,
                "reduction_mg": (d90 - d75) if np.isfinite(d90) and np.isfinite(d75) else np.nan,
                "method_90": m90,
                "method_75": m75,
                "grid_min_mg": float(d[0]) if d.size else np.nan,
                "grid_max_mg": float(d[-1]) if d.size else np.nan,
                "max_success_in_grid": float(s.max()) if s.size else np.nan
            })
        except Exception as e:
            missing.append(f"{sc['name']} (error: {e})")

    res = pd.DataFrame(rows)

    # --------------------------- Save & display ----------------------------
    if not res.empty:
        cols = ["scenario","regimen","dose_col","path","dose_90pct_mg","dose_75pct_mg",
                "reduction_mg","method_90","method_75","grid_min_mg","grid_max_mg","max_success_in_grid"]
        res_sorted = res[cols].sort_values(["regimen", "scenario"]).reset_index(drop=True)
        res_sorted.to_csv(args.out_csv, index=False)
        print(f"Saved: {args.out_csv}")
        print(res_sorted.to_string(index=False))
    else:
        print("No scenarios found. Please check file paths and rerun.")
        res_sorted = pd.DataFrame()

    # ------------------------------- Plot ----------------------------------
    if not args.no_plot and not res_sorted.empty:
        COLOR_90 = "#0000CD"  # MediumBlue
        COLOR_75 = "#999999"

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
        legend_added = False

        for ax, reg in zip(axes, ["QD", "QW"]):
            sub = res_sorted[res_sorted["regimen"] == reg]
            if sub.empty:
                ax.set_visible(False)
                continue

            x = np.arange(len(sub))
            w = 0.35

            ax.bar(x - w/2, sub["dose_90pct_mg"], width=w, label="≥90% target", color=COLOR_90)
            ax.bar(x + w/2, sub["dose_75pct_mg"], width=w, label="≥75% target", color=COLOR_75)

            ax.set_xticks(x)
            labels = []
            for i, lab in enumerate(sub["scenario"]):
                star = "*" if sub["method_90"].iloc[i] != "grid" or sub["method_75"].iloc[i] != "grid" else ""
                labels.append(lab + star)
            ax.set_xticklabels(labels, rotation=20, ha="right")

            ax.set_title(f"{reg} minimal dose (grid/extrapolated)")
            ax.set_ylabel("Dose (mg)")
            ax.grid(True, axis="y", alpha=0.3)

            # Annotate delta
            for i, (d90, d75) in enumerate(zip(sub["dose_90pct_mg"], sub["dose_75pct_mg"])):
                if np.isfinite(d90) and np.isfinite(d75):
                    ax.text(i, max(d90, d75) * 1.02, f"Δ={d90-d75:.1f}", ha="center", fontsize=9)

            if not legend_added and reg == "QD":
                ax.legend(loc="upper left", frameon=True)
                legend_added = True

            # y-axis limits per panel (tweak if desired)
            ax.set_ylim(0, max(1.1 * np.nanmax(sub[["dose_90pct_mg","dose_75pct_mg"]].to_numpy()), 1.0))

        # ensure right subplot (QW) has no legend even if Matplotlib tries to be helpful
        leg = axes[1].get_legend()
        if leg is not None:
            leg.remove()

        plt.tight_layout()
        plt.savefig(args.out_png, dpi=240)
        plt.show()
        print(f"Saved: {args.out_png}")

    # ------------------------------ Notes ----------------------------------
    if missing:
        print("\nSkipped scenarios (missing/invalid files):")
        for m in missing:
            print(" -", m)

    print("\nLegend: method_90/method_75 = 'grid' (in-range), 'interp' (in-range interpolation), "
          "'extrap-linear' (beyond grid, linear), 'extrap-logit' (beyond grid, logistic fit). "
          "Labels with '*' indicate at least one extrapolated value.")

if __name__ == "__main__":
    main()
