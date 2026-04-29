#!/usr/bin/env python3
"""Generate plots and summary metrics for the UWB/IMU fusion report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:
    import imageio

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-uwb-imu-fusion")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rmse(values: pd.Series) -> float:
    return float(np.sqrt(np.nanmean(np.square(values))))


def error_stats(values: pd.Series) -> dict[str, float]:
    clean = values.dropna()
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "rmse": rmse(clean),
        "std": float(clean.std()),
        "p75": float(np.percentile(clean, 75)),
        "p95": float(np.percentile(clean, 95)),
        "max": float(clean.max()),
    }


def save_metrics(df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    wls = error_stats(df["wls_gt_err"])
    tc = error_stats(df["tc_gt_err"])
    duration = float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0])
    sample_dt = np.diff(df["timestamp"].to_numpy())
    tc_better = float((df["tc_gt_err"] < df["wls_gt_err"]).mean())
    low_imu = df["imu_n"] < 2

    metrics = {
        "cycles": int(len(df)),
        "duration_s": duration,
        "mean_cycle_period_s": float(np.mean(sample_dt)),
        "median_cycle_period_s": float(np.median(sample_dt)),
        "wls": wls,
        "tc": tc,
        "tc_better_fraction": tc_better,
        "rmse_delta_m": float(tc["rmse"] - wls["rmse"]),
        "mean_delta_m": float(tc["mean"] - wls["mean"]),
        "low_imu_cycles": int(low_imu.sum()),
        "low_imu_fraction": float(low_imu.mean()),
        "imu_n_mean": float(df["imu_n"].mean()),
        "imu_n_median": float(df["imu_n"].median()),
        "mean_abs_z_error_wls": float(np.mean(np.abs(df["wls_z"] - df["gt_z"]))),
        "mean_abs_z_error_tc": float(np.mean(np.abs(df["tc_z"] - df["gt_z"]))),
        "final_tc_bias_accel": [
            float(df["tc_bias_ax"].iloc[-1]),
            float(df["tc_bias_ay"].iloc[-1]),
            float(df["tc_bias_az"].iloc[-1]),
        ],
        "final_tc_bias_gyro": [
            float(df["tc_bias_gx"].iloc[-1]),
            float(df["tc_bias_gy"].iloc[-1]),
            float(df["tc_bias_gz"].iloc[-1]),
        ],
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    rows = [
        ("WLS UWB-only", wls),
        ("TC UWB+IMU", tc),
    ]
    with (out_dir / "metrics_table.tex").open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Estimator & Mean & Median & RMSE & 75\\% & 95\\% & Max \\\\\n")
        f.write("\\midrule\n")
        for name, stats in rows:
            f.write(
                f"{name} & {stats['mean']:.3f} & {stats['median']:.3f} & "
                f"{stats['rmse']:.3f} & {stats['p75']:.3f} & "
                f"{stats['p95']:.3f} & {stats['max']:.3f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    with (out_dir / "metrics_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Result Metrics\n\n")
        f.write(f"- Cycles: {metrics['cycles']}\n")
        f.write(f"- Duration: {metrics['duration_s']:.3f} s\n")
        f.write(f"- WLS RMSE: {wls['rmse']:.3f} m\n")
        f.write(f"- TC RMSE: {tc['rmse']:.3f} m\n")
        f.write(f"- TC lower-error cycle fraction: {tc_better:.3f}\n")
        f.write(f"- Low-IMU cycles: {metrics['low_imu_cycles']}\n")

    return metrics


def style_axes(ax) -> None:
    ax.grid(True, color="#d7dde5", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_trajectory(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.plot(df["gt_x"], df["gt_y"], color="#111111", linewidth=2.2, label="Ground truth")
    ax.scatter(df["wls_x"], df["wls_y"], s=8, color="#d1495b", alpha=0.35, label="WLS")
    ax.plot(df["tc_x"], df["tc_y"], color="#0077b6", linewidth=1.8, label="TC-FGO")
    ax.scatter(df["gt_x"].iloc[0], df["gt_y"].iloc[0], s=55, color="#2a9d8f", marker="o", label="Start")
    ax.scatter(df["gt_x"].iloc[-1], df["gt_y"].iloc[-1], s=60, color="#f4a261", marker="s", label="End")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Planar trajectory")
    style_axes(ax)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_xy.png", dpi=220)
    plt.close(fig)


def plot_errors(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.2), sharex=True)
    axes[0].plot(df["timestamp"], df["wls_gt_err"], color="#d1495b", alpha=0.62, label="WLS")
    axes[0].plot(df["timestamp"], df["tc_gt_err"], color="#0077b6", linewidth=1.35, label="TC-FGO")
    axes[0].set_ylabel("3D error [m]")
    axes[0].set_title("Position error over time")
    axes[0].legend(loc="upper right")
    style_axes(axes[0])

    wls_tc_pos_err = np.linalg.norm(
        df[["wls_x", "wls_y", "wls_z"]].to_numpy()
        - df[["tc_x", "tc_y", "tc_z"]].to_numpy(),
        axis=1,
    )

    def stat_line(values: np.ndarray | pd.Series) -> tuple[float, float, float, float, float, float]:
        arr = np.asarray(values, dtype=float)
        return (
            float(np.nanmean(arr)),
            float(np.nanmedian(arr)),
            float(np.sqrt(np.nanmean(arr * arr))),
            float(np.nanstd(arr, ddof=1)),
            float(np.nanmax(arr)),
            float(np.nanpercentile(arr, 95)),
        )

    wls_s = stat_line(df["wls_gt_err"])
    tc_s = stat_line(df["tc_gt_err"])
    sep_s = stat_line(wls_tc_pos_err)
    stats_text = (
        "Position error statistics (m)\n"
        "                 mean  median   rmse    std    max    p95\n"
        f"WLS (UWB only) {wls_s[0]:.4f}  {wls_s[1]:.4f} {wls_s[2]:.4f} "
        f"{wls_s[3]:.4f} {wls_s[4]:.4f} {wls_s[5]:.4f}\n"
        f"TC (UWB+IMU)  {tc_s[0]:.4f}  {tc_s[1]:.4f} {tc_s[2]:.4f} "
        f"{tc_s[3]:.4f} {tc_s[4]:.4f} {tc_s[5]:.4f}\n"
        "\n"
        "WLS-TC position separation (m)\n"
        f"mean={sep_s[0]:.4f}, median={sep_s[1]:.4f}, rmse={sep_s[2]:.4f}, "
        f"std={sep_s[3]:.4f}, max={sep_s[4]:.4f}, p95={sep_s[5]:.4f}"
    )
    axes[0].text(
        0.012,
        0.965,
        stats_text,
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#b8c0cc", alpha=0.88),
    )

    axes[1].plot(df["timestamp"], wls_tc_pos_err, color="#6a4c93", linewidth=1.15, label="||WLS - TC||")
    axes[1].fill_between(df["timestamp"], 0, wls_tc_pos_err, color="#6a4c93", alpha=0.18)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("WLS-TC position [m]")
    axes[1].set_title("Position separation between WLS and TC")
    axes[1].legend(loc="upper right")
    style_axes(axes[1])
    fig.tight_layout()
    fig.savefig(out_dir / "error_timeseries.png", dpi=220)
    plt.close(fig)


def plot_cdf(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for col, label, color in [
        ("wls_gt_err", "WLS", "#d1495b"),
        ("tc_gt_err", "TC-FGO", "#0077b6"),
    ]:
        x = np.sort(df[col].dropna().to_numpy())
        y = np.linspace(0.0, 1.0, len(x))
        ax.plot(x, y, color=color, linewidth=2.0, label=label)
    ax.set_xlabel("3D error [m]")
    ax.set_ylabel("empirical CDF")
    ax.set_title("Error distribution")
    style_axes(ax)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "error_cdf.png", dpi=220)
    plt.close(fig)


def plot_state_diagnostics(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.2), sharex=True)
    axes[0].plot(df["timestamp"], df["gt_z"], color="#111111", linewidth=1.8, label="GT z")
    axes[0].plot(df["timestamp"], df["wls_z"], color="#d1495b", alpha=0.55, label="WLS z")
    axes[0].plot(df["timestamp"], df["tc_z"], color="#0077b6", label="TC z")
    axes[0].set_ylabel("z [m]")
    axes[0].set_title("Vertical channel")
    axes[0].legend(loc="upper right")
    style_axes(axes[0])

    speed = np.linalg.norm(df[["tc_vx", "tc_vy", "tc_vz"]].to_numpy(), axis=1)
    axes[1].plot(df["timestamp"], speed, color="#6a4c93", linewidth=1.2)
    axes[1].set_ylabel("TC speed [m/s]")
    style_axes(axes[1])

    axes[2].plot(df["timestamp"], df["imu_n"], color="#264653", linewidth=1.0)
    axes[2].axhline(2, color="#e76f51", linestyle="--", linewidth=1.0, label="dropout threshold")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("IMU samples / cycle")
    axes[2].legend(loc="upper right")
    style_axes(axes[2])
    fig.tight_layout()
    fig.savefig(out_dir / "state_diagnostics.png", dpi=220)
    plt.close(fig)


def plot_error_histogram(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    bins = np.linspace(0, max(df["wls_gt_err"].max(), df["tc_gt_err"].max()), 36)
    ax.hist(df["wls_gt_err"], bins=bins, color="#d1495b", alpha=0.48, label="WLS")
    ax.hist(df["tc_gt_err"], bins=bins, color="#0077b6", alpha=0.48, label="TC-FGO")
    ax.set_xlabel("3D error [m]")
    ax.set_ylabel("cycle count")
    ax.set_title("Error histogram")
    style_axes(ax)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "error_histogram.png", dpi=220)
    plt.close(fig)


def plot_animation(df: pd.DataFrame, out_dir: Path) -> None:
    frames = []
    n_frames = 72
    idxs = np.linspace(8, len(df), n_frames).astype(int)
    x_min = min(df["gt_x"].min(), df["wls_x"].min(), df["tc_x"].min()) - 0.25
    x_max = max(df["gt_x"].max(), df["wls_x"].max(), df["tc_x"].max()) + 0.25
    y_min = min(df["gt_y"].min(), df["wls_y"].min(), df["tc_y"].min()) - 0.25
    y_max = max(df["gt_y"].max(), df["wls_y"].max(), df["tc_y"].max()) + 0.25

    for i, stop in enumerate(idxs):
        sub = df.iloc[:stop]
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        ax.plot(sub["gt_x"], sub["gt_y"], color="#111111", linewidth=2.0, label="GT")
        ax.scatter(sub["wls_x"], sub["wls_y"], s=8, color="#d1495b", alpha=0.35, label="WLS")
        ax.plot(sub["tc_x"], sub["tc_y"], color="#0077b6", linewidth=1.7, label="TC")
        ax.scatter(sub["tc_x"].iloc[-1], sub["tc_y"].iloc[-1], color="#0077b6", s=45)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Trajectory replay, t={sub['timestamp'].iloc[-1]:.1f}s")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        style_axes(ax)
        if i == 0:
            ax.legend(loc="best")
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())
        plt.close(fig)

    imageio.mimsave(out_dir / "trajectory_replay.gif", frames, duration=0.08)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["gt_x", "gt_y", "gt_z", "wls_x", "wls_y", "wls_z", "tc_x", "tc_y", "tc_z"]
    )

    metrics = save_metrics(df, args.out_dir)
    plot_trajectory(df, args.out_dir)
    plot_errors(df, args.out_dir)
    plot_cdf(df, args.out_dir)
    plot_state_diagnostics(df, args.out_dir)
    plot_error_histogram(df, args.out_dir)
    plot_animation(df, args.out_dir)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
