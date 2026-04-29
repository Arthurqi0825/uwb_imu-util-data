#!/usr/bin/env python3
"""Generate trajectory XY plot and replay GIF from a UWB/IMU trajectory CSV.

Both outputs include:
  - ground truth trajectory
  - WLS UWB-only trajectory
  - TC UWB+IMU trajectory
"""

from __future__ import annotations

import argparse
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


REQUIRED_COLUMNS = [
    "timestamp",
    "gt_x",
    "gt_y",
    "wls_x",
    "wls_y",
    "tc_x",
    "tc_y",
]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")


def style_axes(ax) -> None:
    ax.grid(True, color="#d7dde5", linewidth=0.7, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_xy(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    ax.plot(df["gt_x"], df["gt_y"], color="#111111", linewidth=2.3, label="Ground truth")
    ax.scatter(df["wls_x"], df["wls_y"], s=8, color="#d1495b", alpha=0.35, label="WLS UWB-only")
    ax.plot(df["tc_x"], df["tc_y"], color="#0077b6", linewidth=1.9, label="TC UWB+IMU")

    ax.scatter(df["gt_x"].iloc[0], df["gt_y"].iloc[0], s=58, color="#2a9d8f", marker="o", label="Start")
    ax.scatter(df["gt_x"].iloc[-1], df["gt_y"].iloc[-1], s=62, color="#f4a261", marker="s", label="End")

    x_cols = ["gt_x", "wls_x", "tc_x"]
    y_cols = ["gt_y", "wls_y", "tc_y"]
    x_min = float(df[x_cols].min().min())
    x_max = float(df[x_cols].max().max())
    y_min = float(df[y_cols].min().min())
    y_max = float(df[y_cols].max().max())
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    half_range = 0.5 * max(x_max - x_min, y_max - y_min) + 0.25
    ax.set_xlim(x_mid - half_range, x_mid + half_range)
    ax.set_ylim(y_mid - half_range, y_mid + half_range)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory comparison: GT / WLS / TC", fontsize=18, fontweight="bold")
    style_axes(ax)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def render_frame(df: pd.DataFrame, stop: int, limits: tuple[float, float, float, float]) -> np.ndarray:
    sub = df.iloc[:stop]
    x_min, x_max, y_min, y_max = limits

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot(sub["gt_x"], sub["gt_y"], color="#111111", linewidth=2.1, label="GT")
    ax.scatter(sub["wls_x"], sub["wls_y"], s=8, color="#d1495b", alpha=0.38, label="WLS")
    ax.plot(sub["tc_x"], sub["tc_y"], color="#0077b6", linewidth=1.8, label="TC")

    ax.scatter(sub["gt_x"].iloc[-1], sub["gt_y"].iloc[-1], color="#111111", s=34, marker="o")
    ax.scatter(sub["wls_x"].iloc[-1], sub["wls_y"].iloc[-1], color="#d1495b", s=34, marker="o")
    ax.scatter(sub["tc_x"].iloc[-1], sub["tc_y"].iloc[-1], color="#0077b6", s=42, marker="o")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Trajectory replay, t={sub['timestamp'].iloc[-1]:.1f}s")
    style_axes(ax)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = rgba[:, :, :3].copy()
    plt.close(fig)
    return frame


def plot_gif(df: pd.DataFrame, output_path: Path, frames: int, duration: float) -> None:
    frames = max(2, frames)
    stop_indices = np.linspace(8, len(df), frames).astype(int)
    stop_indices = np.clip(stop_indices, 1, len(df))

    x_cols = ["gt_x", "wls_x", "tc_x"]
    y_cols = ["gt_y", "wls_y", "tc_y"]
    x_min = float(df[x_cols].min().min()) - 0.25
    x_max = float(df[x_cols].max().max()) + 0.25
    y_min = float(df[y_cols].min().min()) - 0.25
    y_max = float(df[y_cols].max().max()) + 0.25
    limits = (x_min, x_max, y_min, y_max)

    gif_frames = [render_frame(df, stop, limits) for stop in stop_indices]
    imageio.mimsave(output_path, gif_frames, duration=duration)
    


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate trajectory_xy.png and trajectory_replay.gif with GT, WLS, and TC."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("output/uwb_imu_trajectory_tc_260429.csv"),
        help="Input trajectory CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("report/trajectory_visuals"),
        help="Output directory.",
    )
    parser.add_argument("--frames", type=int, default=90, help="Number of GIF frames.")
    parser.add_argument("--duration", type=float, default=0.08, help="GIF frame duration in seconds.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv).replace([np.inf, -np.inf], np.nan)
    validate_columns(df)
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)

    xy_path = args.out_dir / "trajectory_xy.png"
    gif_path = args.out_dir / "trajectory_replay.gif"

    plot_xy(df, xy_path)
    plot_gif(df, gif_path, frames=args.frames, duration=args.duration)

    print(f"Wrote {xy_path}")
    print(f"Wrote {gif_path}")


if __name__ == "__main__":
    main()
