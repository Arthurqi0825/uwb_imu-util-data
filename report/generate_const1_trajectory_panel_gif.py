#!/usr/bin/env python3
"""Generate a 2x2 animated trajectory diagnostics GIF for the const1 dataset.

Panels:
  - 3D trajectory with anchors
  - Overall residuals/errors: WLS->GT and TC->GT
  - XY trajectory with anchors
  - Vertical channel: GT, TC, and WLS z
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


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CSV_DEFAULT = PACKAGE_ROOT / "output/260427_const1_3_2_uwb_imu_trajectory_tc.csv"
OUT_DEFAULT = PACKAGE_ROOT / "report/260427_const1_3_2_trajectory_panel_long.gif"

ANCHORS = np.array(
    [
        [-2.4175, -4.0208, 0.1818],
        [-2.8205, 3.5250, 2.5874],
        [3.4819, 3.3050, 0.1545],
        [3.4507, -3.7181, 2.6693],
        [-3.2776, -3.8690, 2.6739],
        [3.2655, -3.6511, 0.1752],
        [3.8321, 3.6521, 2.6249],
        [-2.7228, 3.2191, 0.1583],
    ],
    dtype=float,
)

REQUIRED_COLUMNS = [
    "timestamp",
    "gt_x",
    "gt_y",
    "gt_z",
    "wls_x",
    "wls_y",
    "wls_z",
    "tc_x",
    "tc_y",
    "tc_z",
]

COLORS = {
    "gt": "#111111",
    "wls": "#d1495b",
    "tc": "#0077b6",
    "anchor": "#f4a261",
    "cursor": "#5c677d",
}


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")


def style_axes(ax) -> None:
    ax.grid(True, color="#d7dde5", linewidth=0.7, alpha=0.85)
    if hasattr(ax, "spines"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def equal_3d_limits(points: np.ndarray, pad: float = 0.35) -> tuple[tuple[float, float], ...]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    half_range = 0.5 * float(np.max(maxs - mins)) + pad
    return tuple((float(c - half_range), float(c + half_range)) for c in center)


def equal_xy_limits(points: np.ndarray, pad: float = 0.35) -> tuple[float, float, float, float]:
    mins = points[:, :2].min(axis=0)
    maxs = points[:, :2].max(axis=0)
    center = 0.5 * (mins + maxs)
    half_range = 0.5 * float(np.max(maxs - mins)) + pad
    return (
        float(center[0] - half_range),
        float(center[0] + half_range),
        float(center[1] - half_range),
        float(center[1] + half_range),
    )


def add_anchor_labels_3d(ax) -> None:
    for idx, (x, y, z) in enumerate(ANCHORS):
        ax.text(x, y, z + 0.08, f"A{idx}", color="#9a5b12", fontsize=8)


def add_anchor_labels_2d(ax) -> None:
    for idx, (x, y, _z) in enumerate(ANCHORS):
        ax.text(x + 0.05, y + 0.05, f"A{idx}", color="#9a5b12", fontsize=8)


def render_frame(
    df: pd.DataFrame,
    stop: int,
    frame_idx: int,
    frame_count: int,
    limits_3d: tuple[tuple[float, float], ...],
    limits_xy: tuple[float, float, float, float],
    residual_ylim: tuple[float, float],
    z_ylim: tuple[float, float],
) -> np.ndarray:
    sub = df.iloc[:stop]
    t_now = float(sub["timestamp"].iloc[-1])
    azim = -62.0 + 32.0 * frame_idx / max(1, frame_count - 1)

    fig = plt.figure(figsize=(14.0, 10.0), dpi=110)
    ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_res = fig.add_subplot(2, 2, 2)
    ax_xy = fig.add_subplot(2, 2, 3)
    ax_z = fig.add_subplot(2, 2, 4)

    ax_3d.plot(sub["gt_x"], sub["gt_y"], sub["gt_z"], color=COLORS["gt"], linewidth=2.0, label="GT")
    ax_3d.plot(sub["wls_x"], sub["wls_y"], sub["wls_z"], color=COLORS["wls"], linewidth=1.3, alpha=0.75, label="WLS")
    ax_3d.plot(sub["tc_x"], sub["tc_y"], sub["tc_z"], color=COLORS["tc"], linewidth=1.8, label="TC")
    ax_3d.scatter(ANCHORS[:, 0], ANCHORS[:, 1], ANCHORS[:, 2], color=COLORS["anchor"], marker="^", s=48, label="Anchor")
    ax_3d.scatter(sub["gt_x"].iloc[-1], sub["gt_y"].iloc[-1], sub["gt_z"].iloc[-1], color=COLORS["gt"], s=32)
    ax_3d.scatter(sub["wls_x"].iloc[-1], sub["wls_y"].iloc[-1], sub["wls_z"].iloc[-1], color=COLORS["wls"], s=32)
    ax_3d.scatter(sub["tc_x"].iloc[-1], sub["tc_y"].iloc[-1], sub["tc_z"].iloc[-1], color=COLORS["tc"], s=38)
    add_anchor_labels_3d(ax_3d)
    ax_3d.set_xlim(*limits_3d[0])
    ax_3d.set_ylim(*limits_3d[1])
    ax_3d.set_zlim(*limits_3d[2])
    ax_3d.set_xlabel("x [m]")
    ax_3d.set_ylabel("y [m]")
    ax_3d.set_zlabel("z [m]")
    ax_3d.set_title(f"3D trajectory, t={t_now:.1f}s")
    ax_3d.view_init(elev=24.0, azim=azim)
    ax_3d.legend(loc="upper left", frameon=True, fontsize=8)

    ax_res.plot(sub["timestamp"], sub["wls_gt_err"], color=COLORS["wls"], linewidth=1.4, label="WLS->GT")
    ax_res.plot(sub["timestamp"], sub["tc_gt_err"], color=COLORS["tc"], linewidth=1.6, label="TC->GT")
    ax_res.axvline(t_now, color=COLORS["cursor"], linewidth=1.0, linestyle="--", alpha=0.75)
    ax_res.scatter(t_now, sub["wls_gt_err"].iloc[-1], color=COLORS["wls"], s=28)
    ax_res.scatter(t_now, sub["tc_gt_err"].iloc[-1], color=COLORS["tc"], s=34)
    ax_res.set_xlim(float(df["timestamp"].min()), float(df["timestamp"].max()))
    ax_res.set_ylim(*residual_ylim)
    ax_res.set_xlabel("time [s]")
    ax_res.set_ylabel("3D residual [m]")
    ax_res.set_title("Overall residuals")
    style_axes(ax_res)
    ax_res.legend(loc="upper right", frameon=True, fontsize=8)

    ax_xy.plot(sub["gt_x"], sub["gt_y"], color=COLORS["gt"], linewidth=2.0, label="GT")
    ax_xy.plot(sub["wls_x"], sub["wls_y"], color=COLORS["wls"], linewidth=1.2, alpha=0.75, label="WLS")
    ax_xy.plot(sub["tc_x"], sub["tc_y"], color=COLORS["tc"], linewidth=1.7, label="TC")
    ax_xy.scatter(ANCHORS[:, 0], ANCHORS[:, 1], color=COLORS["anchor"], marker="^", s=48, label="Anchor")
    ax_xy.scatter(sub["gt_x"].iloc[-1], sub["gt_y"].iloc[-1], color=COLORS["gt"], s=30)
    ax_xy.scatter(sub["wls_x"].iloc[-1], sub["wls_y"].iloc[-1], color=COLORS["wls"], s=30)
    ax_xy.scatter(sub["tc_x"].iloc[-1], sub["tc_y"].iloc[-1], color=COLORS["tc"], s=36)
    add_anchor_labels_2d(ax_xy)
    ax_xy.set_xlim(limits_xy[0], limits_xy[1])
    ax_xy.set_ylim(limits_xy[2], limits_xy[3])
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("XY trajectory")
    style_axes(ax_xy)
    ax_xy.legend(loc="upper right", frameon=True, fontsize=8)

    ax_z.plot(sub["timestamp"], sub["gt_z"], color=COLORS["gt"], linewidth=2.0, label="GT z")
    ax_z.plot(sub["timestamp"], sub["tc_z"], color=COLORS["tc"], linewidth=1.7, label="TC z")
    ax_z.plot(sub["timestamp"], sub["wls_z"], color=COLORS["wls"], linewidth=1.2, alpha=0.75, label="WLS z")
    ax_z.axvline(t_now, color=COLORS["cursor"], linewidth=1.0, linestyle="--", alpha=0.75)
    ax_z.scatter(t_now, sub["gt_z"].iloc[-1], color=COLORS["gt"], s=28)
    ax_z.scatter(t_now, sub["tc_z"].iloc[-1], color=COLORS["tc"], s=34)
    ax_z.scatter(t_now, sub["wls_z"].iloc[-1], color=COLORS["wls"], s=28)
    ax_z.set_xlim(float(df["timestamp"].min()), float(df["timestamp"].max()))
    ax_z.set_ylim(*z_ylim)
    ax_z.set_xlabel("time [s]")
    ax_z.set_ylabel("z [m]")
    ax_z.set_title("Vertical channel")
    style_axes(ax_z)
    ax_z.legend(loc="upper right", frameon=True, fontsize=8)

    fig.suptitle("Const1 GT / TC / WLS trajectory diagnostics", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = rgba[:, :, :3].copy()
    plt.close(fig)
    return frame


def prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).replace([np.inf, -np.inf], np.nan)
    validate_columns(df)
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)

    if "wls_gt_err" not in df.columns:
        df["wls_gt_err"] = np.linalg.norm(
            df[["wls_x", "wls_y", "wls_z"]].to_numpy() - df[["gt_x", "gt_y", "gt_z"]].to_numpy(),
            axis=1,
        )
    if "tc_gt_err" not in df.columns:
        df["tc_gt_err"] = np.linalg.norm(
            df[["tc_x", "tc_y", "tc_z"]].to_numpy() - df[["gt_x", "gt_y", "gt_z"]].to_numpy(),
            axis=1,
        )

    return df.dropna(subset=["wls_gt_err", "tc_gt_err"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the const1 2x2 animated trajectory GIF.")
    parser.add_argument(
        "csv_input",
        nargs="?",
        type=Path,
        help="Input trajectory CSV. This is equivalent to --csv.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Input trajectory CSV.")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT, help="Output GIF path.")
    parser.add_argument("--frames", type=int, default=180, help="Number of GIF frames.")
    parser.add_argument("--duration", type=float, default=0.10, help="GIF frame duration in seconds.")
    args = parser.parse_args()

    csv_path = args.csv or args.csv_input or CSV_DEFAULT
    df = prepare_dataframe(csv_path)
    if len(df) < 2:
        raise ValueError("CSV must contain at least two valid samples.")

    trajectory_points = np.vstack(
        [
            df[["gt_x", "gt_y", "gt_z"]].to_numpy(),
            df[["wls_x", "wls_y", "wls_z"]].to_numpy(),
            df[["tc_x", "tc_y", "tc_z"]].to_numpy(),
            ANCHORS,
        ]
    )
    limits_3d = equal_3d_limits(trajectory_points)
    limits_xy = equal_xy_limits(trajectory_points)
    residual_top = float(df[["wls_gt_err", "tc_gt_err"]].quantile(0.995).max()) + 0.08
    residual_ylim = (0.0, max(0.25, residual_top))
    z_pad = 0.12
    z_ylim = (
        float(df[["gt_z", "wls_z", "tc_z"]].min().min()) - z_pad,
        float(df[["gt_z", "wls_z", "tc_z"]].max().max()) + z_pad,
    )

    frame_count = max(2, args.frames)
    stop_indices = np.linspace(8, len(df), frame_count).astype(int)
    stop_indices = np.clip(stop_indices, 1, len(df))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        render_frame(
            df,
            int(stop),
            frame_idx,
            frame_count,
            limits_3d,
            limits_xy,
            residual_ylim,
            z_ylim,
        )
        for frame_idx, stop in enumerate(stop_indices)
    ]
    imageio.mimsave(args.out, frames, duration=args.duration)
    print(f"Wrote {args.out}")
    print(f"Frames: {frame_count}, duration: {args.duration:.3f}s, playback: {frame_count * args.duration:.1f}s")


if __name__ == "__main__":
    main()
