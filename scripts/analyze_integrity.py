#!/usr/bin/env python3
"""Analyze UWB+IMU integrity trajectory CSVs.

This extends scripts/analyze_comprehensive.py for CSVs produced by both
integrity_uwb_node.cpp (launch/uwb_wls_integrity.launch) and
uwb_imu_tc_integrity_node.cpp (launch/uwb_imu_tc_integrity.launch). It keeps
the usual trajectory/error plots and adds integrity-specific diagnostics:

  - chi-square statistic vs threshold
  - HPL/VPL vs HAL/VAL
  - availability, fault, FDE/exclusion timeline
  - accepted factor count vs input count
  - max standardized residual and factor sigma summaries
  - excluded-pair frequency

TC list-valued CSV columns use "|" separators, matching the node logger. WLS
integrity CSVs log some equivalent diagnostics as scalar summaries instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CORE_INTEGRITY_COLUMNS = {
    "integ_chi2",
    "integ_threshold",
    "integ_fault",
    "integ_hpl",
    "integ_vpl",
    "integ_available",
}

def detect_kind(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if {"tc_x", "tc_y", "tc_z"}.issubset(cols):
        return "tc"
    if {"lc_x", "lc_y", "lc_z"}.issubset(cols):
        return "lc"
    return "unknown"


def has_integrity(df: pd.DataFrame) -> bool:
    return CORE_INTEGRITY_COLUMNS.issubset(set(df.columns))


def numeric_col(df: pd.DataFrame, *names: str, default=np.nan) -> pd.Series:
    for name in names:
        if name in df:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def bool_col(df: pd.DataFrame, *names: str, default=False) -> pd.Series:
    return numeric_col(df, *names, default=int(default)).fillna(0).astype(bool)


def first_existing_col(df: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in df:
            return name
    return None


def err_stats(e: pd.Series | np.ndarray) -> dict:
    arr = pd.Series(e).dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return {k: np.nan for k in ("mean", "median", "rmse", "std", "max", "p95")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "std": float(np.std(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def per_axis_err(df: pd.DataFrame, prefix: str):
    return (
        df[f"{prefix}_x"] - df["gt_x"],
        df[f"{prefix}_y"] - df["gt_y"],
        df[f"{prefix}_z"] - df["gt_z"],
    )


def split_list_cell(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "none":
        return []
    return [x for x in text.split("|") if x]


def split_token_cell(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "none":
        return []
    for sep in (";", "|"):
        text = text.replace(sep, " ")
    return [x for x in text.split() if x]


def parse_float_list(value) -> list[float]:
    out = []
    for item in split_list_cell(value):
        try:
            out.append(float(item))
        except ValueError:
            pass
    return out


def list_abs_max(series: pd.Series) -> np.ndarray:
    vals = []
    for value in series:
        items = parse_float_list(value)
        vals.append(max((abs(x) for x in items), default=np.nan))
    return np.asarray(vals, dtype=float)


def list_mean(series: pd.Series) -> np.ndarray:
    vals = []
    for value in series:
        items = parse_float_list(value)
        vals.append(float(np.mean(items)) if items else np.nan)
    return np.asarray(vals, dtype=float)


def count_label(series: pd.Series, label: str) -> np.ndarray:
    vals = []
    for value in series:
        items = split_list_cell(value)
        vals.append(sum(1 for item in items if item == label))
    return np.asarray(vals, dtype=float)


def pair_counter(series: pd.Series) -> Counter:
    c = Counter()
    for value in series:
        c.update(split_token_cell(value))
    return c


def excluded_count_series(df: pd.DataFrame) -> pd.Series:
    if "integ_excluded_count" in df:
        return numeric_col(df, "integ_excluded_count", default=0).fillna(0)
    if "integ_excluded_idx" in df:
        idx = numeric_col(df, "integ_excluded_idx", default=-1)
        return (idx >= 0).astype(float)
    return pd.Series(0.0, index=df.index)


def input_count_series(df: pd.DataFrame) -> pd.Series:
    return numeric_col(df, "integ_input_count", "raw_meas_count")


def factor_count_series(df: pd.DataFrame) -> pd.Series:
    return numeric_col(df, "integ_factor_count", "used_meas_count")


def max_std_resid_series(df: pd.DataFrame) -> np.ndarray:
    if "integ_factor_std_resid" in df:
        return list_abs_max(df["integ_factor_std_resid"])
    return numeric_col(df, "std_resid_abs_max").to_numpy(dtype=float)


def mean_sigma_series(df: pd.DataFrame) -> np.ndarray:
    if "integ_factor_sigmas" in df:
        return list_mean(df["integ_factor_sigmas"])
    return numeric_col(df, "sigma_mean").to_numpy(dtype=float)


def pair_series(df: pd.DataFrame) -> pd.Series:
    col = first_existing_col(df, "integ_excluded_pairs", "nlos_rejected_pairs")
    if col is None:
        return pd.Series("none", index=df.index, dtype=object)
    return df[col]


def print_stats_table(stats: dict):
    table = pd.DataFrame(stats).T[["mean", "median", "rmse", "std", "max", "p95"]]
    print("\n=== Position error statistics (m) ===")
    print(table.to_string(float_format=lambda v: f"{v:0.4f}"))


def print_integrity_summary(df: pd.DataFrame):
    if not has_integrity(df):
        print("\n[info] no integrity columns found.")
        return

    used = bool_col(df, "integ_used", default=True)
    available = bool_col(df, "integ_available")
    fault = bool_col(df, "integ_fault")
    excluded_count = excluded_count_series(df)
    input_count = input_count_series(df)
    factor_count = factor_count_series(df)

    print("\n=== Integrity summary ===")
    print(f"monitor used rows       : {int(used.sum())}/{len(df)}")
    print(f"available rows          : {int(available.sum())}/{len(df)}")
    print(f"fault rows              : {int(fault.sum())}/{len(df)}")
    print(f"rows with exclusions    : {int((excluded_count > 0).sum())}/{len(df)}")
    print(f"mean input/factor count : {input_count.mean():.2f} / {factor_count.mean():.2f}")
    if "nlos_anchor_removed" in df:
        removed = bool_col(df, "nlos_anchor_removed")
        print(f"NLOS anchor removed rows: {int(removed.sum())}/{len(df)}")
        if "nlos_anchor_id" in df and removed.any():
            counts = numeric_col(df.loc[removed], "nlos_anchor_id").astype(int).value_counts().sort_index()
            detail = ", ".join(f"{idx}:{count}" for idx, count in counts.items())
            print(f"NLOS anchor frequency   : {detail}")

    hpl = numeric_col(df, "integ_hpl")
    vpl = numeric_col(df, "integ_vpl")
    print(f"HPL mean/p95/max        : {hpl.mean():.3f} / {hpl.quantile(.95):.3f} / {hpl.max():.3f}")
    print(f"VPL mean/p95/max        : {vpl.mean():.3f} / {vpl.quantile(.95):.3f} / {vpl.max():.3f}")

    c = pair_counter(pair_series(df))
    if c:
        print("\nTop excluded/NLOS pairs:")
        for pair, n in c.most_common(10):
            print(f"  {pair:>5s}: {n}")


def analyze_trajectory(df: pd.DataFrame, kind: str, title: str, save_path: str | None):
    fused = kind
    fused_label = "TC (UWB+IMU)" if kind == "tc" else "LC (UWB+IMU)"
    t = df["timestamp"].to_numpy()
    fu_ex, fu_ey, fu_ez = per_axis_err(df, fused)

    stats = {"WLS all anchors": err_stats(df["wls_gt_err"])}
    if "wls_clean_gt_err" in df:
        stats["WLS clean LOS"] = err_stats(df["wls_clean_gt_err"])
    stats[fused_label] = err_stats(df[f"{fused}_gt_err"])
    print_stats_table(stats)

    print("\n=== Per-axis mean bias (estimate - GT, m) ===")
    rows = {}
    axis_rows = [("wls", "WLS all anchors")]
    if {"wls_clean_x", "wls_clean_y", "wls_clean_z"}.issubset(df.columns):
        axis_rows.append(("wls_clean", "WLS clean LOS"))
    axis_rows.append((fused, fused_label))
    for prefix, label in axis_rows:
        ex, ey, ez = per_axis_err(df, prefix)
        rows[label] = {
            "dx_mean": ex.mean(), "dy_mean": ey.mean(), "dz_mean": ez.mean(),
            "dx_std": ex.std(), "dy_std": ey.std(), "dz_std": ez.std(),
        }
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:0.4f}"))

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"Trajectory analysis - {title}", fontsize=13)

    ax = fig.add_subplot(2, 3, 1)
    ax.plot(df["gt_x"], df["gt_y"], "k-", lw=2, label="Ground truth")
    ax.plot(df["wls_x"], df["wls_y"], "r.", ms=2, alpha=0.35, label="WLS")
    if {"wls_clean_x", "wls_clean_y"}.issubset(df.columns):
        ax.plot(df["wls_clean_x"], df["wls_clean_y"], "c.", ms=2, alpha=0.45, label="WLS clean")
    ax.plot(df[f"{fused}_x"], df[f"{fused}_y"], "b-", alpha=0.85, label=fused_label)
    ax.set_title("XY trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 2, projection="3d")
    ax.plot(df["gt_x"], df["gt_y"], df["gt_z"], "k-", lw=2, label="GT")
    ax.plot(df["wls_x"], df["wls_y"], df["wls_z"], "r.", ms=1.5, alpha=0.35, label="WLS")
    if {"wls_clean_x", "wls_clean_y", "wls_clean_z"}.issubset(df.columns):
        ax.plot(df["wls_clean_x"], df["wls_clean_y"], df["wls_clean_z"],
                "c.", ms=1.5, alpha=0.45, label="WLS clean")
    ax.plot(df[f"{fused}_x"], df[f"{fused}_y"], df[f"{fused}_z"], "b-", alpha=0.85, label=fused_label)
    ax.set_title("3D trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 3)
    ax.plot(t, df["wls_gt_err"], "r", alpha=0.55, label="WLS")
    if "wls_clean_gt_err" in df:
        ax.plot(t, df["wls_clean_gt_err"], "c", alpha=0.65, label="WLS clean")
    ax.plot(t, df[f"{fused}_gt_err"], "b", alpha=0.85, label=fused_label)
    ax.set_title("Position error")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("|err| (m)")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 4)
    ax.plot(t, fu_ex, "r", label="x")
    ax.plot(t, fu_ey, "g", label="y")
    ax.plot(t, fu_ez, "b", label="z")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"{fused_label} per-axis error")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("err (m)")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(t, df["wls_z"] - df["gt_z"], "r", alpha=0.65, label="WLS z err")
    if "wls_clean_z" in df:
        ax.plot(t, df["wls_clean_z"] - df["gt_z"], "c", alpha=0.70, label="WLS clean z err")
    ax.plot(t, df[f"{fused}_z"] - df["gt_z"], "b", alpha=0.85, label=f"{fused.upper()} z err")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Vertical error focus")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("z err (m)")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 6)
    d = np.sqrt(
        (df[f"{fused}_x"] - df["wls_x"]) ** 2 +
        (df[f"{fused}_y"] - df["wls_y"]) ** 2 +
        (df[f"{fused}_z"] - df["wls_z"]) ** 2
    )
    ax.plot(t, d, "m", alpha=0.85)
    ax.set_title(f"|{fused.upper()} - WLS|")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("m")
    ax.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=130)
        print(f"[saved] {save_path}")


def analyze_integrity(df: pd.DataFrame, title: str, save_path: str | None):
    if not has_integrity(df):
        return

    t = df["timestamp"].to_numpy()
    chi2 = numeric_col(df, "integ_chi2")
    threshold = numeric_col(df, "integ_threshold")
    hpl = numeric_col(df, "integ_hpl")
    vpl = numeric_col(df, "integ_vpl")
    hal = numeric_col(df, "integ_hal")
    val = numeric_col(df, "integ_val")
    available = numeric_col(df, "integ_available", default=0).fillna(0)
    fault = numeric_col(df, "integ_fault", default=0).fillna(0)
    excluded_count = excluded_count_series(df)
    input_count = input_count_series(df)
    factor_count = factor_count_series(df)
    max_std = max_std_resid_series(df)
    mean_sigma = mean_sigma_series(df)
    inflated_count = (count_label(df["integ_factor_labels"], "inflated")
                      if "integ_factor_labels" in df
                      else np.full(len(df), np.nan))

    print_integrity_summary(df)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    fig.suptitle(f"Integrity diagnostics - {title}", fontsize=13)

    ax = axes[0, 0]
    ax.plot(t, chi2, label="chi2", color="tab:blue")
    ax.plot(t, threshold, label="threshold", color="tab:red", linestyle="--")
    ax.set_title("Chi-square global test")
    ax.set_ylabel("statistic")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(t, hpl, label="HPL", color="tab:orange")
    if hal.notna().any():
        ax.plot(t, hal, label="HAL", color="tab:orange", linestyle="--")
    ax.plot(t, vpl, label="VPL", color="tab:purple")
    if val.notna().any():
        ax.plot(t, val, label="VAL", color="tab:purple", linestyle="--")
    ax.set_title("Protection levels vs alert limits")
    ax.set_ylabel("m")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.step(t, available, where="post", label="available", color="tab:green")
    ax.step(t, fault, where="post", label="fault", color="tab:red")
    ax.step(t, excluded_count, where="post", label="excluded count", color="tab:gray")
    ax.set_title("Availability, faults, FDE")
    ax.set_ylabel("state/count")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(t, input_count, label="input TDOAs", color="tab:blue")
    ax.plot(t, factor_count, label="accepted factors", color="tab:green")
    if np.isfinite(inflated_count).any():
        ax.plot(t, inflated_count, label="inflated factors", color="tab:orange")
    ax.set_title("TDOA factor accounting")
    ax.set_ylabel("count")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    ax.plot(t, max_std, color="tab:red", label="max |standardized residual|")
    if "integ_soft_gate" in df:
        ax.plot(t, pd.to_numeric(df["integ_soft_gate"], errors="coerce"),
                "k--", lw=1, label="soft gate")
    if "integ_hard_gate" in df:
        ax.plot(t, pd.to_numeric(df["integ_hard_gate"], errors="coerce"),
                "k:", lw=1, label="hard gate")
    ax.set_title("Standardized residual summary")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("sigma")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    ax.plot(t, mean_sigma, color="tab:blue", label="mean accepted sigma")
    ax.set_title("Accepted factor sigma")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("m")
    ax.grid(True)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=130)
        print(f"[saved] {save_path}")

    counter = pair_counter(pair_series(df))
    if counter:
        pairs, counts = zip(*counter.most_common())
        fig2, ax = plt.subplots(figsize=(10, 5))
        ax.bar(pairs, counts, color="tab:red", alpha=0.8)
        ax.set_title(f"Excluded/NLOS TDOA pair frequency - {title}")
        ax.set_xlabel("anchor pair")
        ax.set_ylabel("exclusion count")
        ax.grid(True, axis="y")
        fig2.tight_layout()
        if save_path:
            pair_path = save_path.replace("_integrity.png", "_excluded_pairs.png")
            fig2.savefig(pair_path, dpi=130)
            print(f"[saved] {pair_path}")


def process_file(path: str, save_dir: str | None):
    print(f"\n########## {os.path.basename(path)} ##########")
    df = pd.read_csv(path)
    kind = detect_kind(df)
    if "timestamp" not in df:
        print("[skip] missing timestamp")
        return
    duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    print(f"kind={kind}, rows={len(df)}, duration={duration:.2f}s, integrity={has_integrity(df)}")

    if kind not in ("tc", "lc"):
        print(f"[skip] not a trajectory CSV (cols={list(df.columns)[:6]}...)")
        return

    base = os.path.splitext(os.path.basename(path))[0]
    traj_path = os.path.join(save_dir, f"{base}_trajectory.png") if save_dir else None
    integ_path = os.path.join(save_dir, f"{base}_integrity.png") if save_dir else None
    analyze_trajectory(df, kind, base, traj_path)
    analyze_integrity(df, base, integ_path)


def collect_files(targets: list[str]) -> list[str]:
    files = []
    for target in targets:
        if os.path.isdir(target):
            files.extend(sorted(glob(os.path.join(target, "*.csv"))))
        elif os.path.isfile(target):
            files.append(target)
        else:
            print(f"[warn] not found: {target}", file=sys.stderr)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Analyze UWB-IMU WLS/LC and TC integrity trajectory CSV files.")
    output_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "output"))
    default_inputs = [
        os.path.join(output_dir, "uwb_wls_integrity.csv"),
        os.path.join(output_dir, "uwb_imu_tc_integrity.csv"),
        os.path.join(output_dir, "uwb_imu_tc_integrity_trajectory.csv"),
    ]
    parser.add_argument("inputs", nargs="*", default=[],
                        help="CSV files or directories. Default: existing launch output CSVs.")
    parser.add_argument("--save-dir", default=None,
                        help="Directory to save figures as PNG.")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not call plt.show().")
    args = parser.parse_args()

    inputs = args.inputs
    if not inputs:
        inputs = [path for path in default_inputs if os.path.exists(path)] or default_inputs

    files = collect_files(inputs)
    if not files:
        print("No CSV files to process.")
        sys.exit(1)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for path in files:
        try:
            process_file(path, args.save_dir)
        except Exception as exc:
            print(f"[error] {path}: {exc}", file=sys.stderr)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
