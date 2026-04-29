#!/usr/bin/env python3
"""Generate an animated GIF for the vertical channel diagnostic.

The animation mirrors the first subplot of state_diagnostics.png:
GT z, WLS z, and TC z over time, with a moving time cursor.
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


REQUIRED_COLUMNS = ["timestamp", "gt_z", "wls_z", "tc_z"]


def style_axes(ax) -> None:
    ax.grid(True, color="#d7dde5", linewidth=0.7, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_frame(
    df: pd.DataFrame,
    stop: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> np.ndarray:
    sub = df.iloc[:stop]
    t_now = float(sub["timestamp"].iloc[-1])

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(sub["timestamp"], sub["gt_z"], color="#111111", linewidth=2.1, label="GT z")
    ax.plot(sub["timestamp"], sub["wls_z"], color="#d1495b", linewidth=1.4, alpha=0.75, label="WLS z")
    ax.plot(sub["timestamp"], sub["tc_z"], color="#0077b6", linewidth=1.7, label="TC z")

    ax.scatter(t_now, sub["gt_z"].iloc[-1], color="#111111", s=35, zorder=5)
    ax.scatter(t_now, sub["wls_z"].iloc[-1], color="#d1495b", s=35, zorder=5)
    ax.scatter(t_now, sub["tc_z"].iloc[-1], color="#0077b6", s=42, zorder=5)
    ax.axvline(t_now, color="#555555", linewidth=1.0, linestyle="--", alpha=0.75)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"Vertical channel replay, t={t_now:.1f}s")
    style_axes(ax)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = rgba[:, :, :3].copy()
    plt.close(fig)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate animated vertical-channel GIF with GT z, WLS z, and TC z."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("output/uwb_imu_trajectory_tc_260429.csv"),
        help="Input trajectory CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("report/260429/vertical_channel_replay.gif"),
        help="Output GIF path.",
    )
    parser.add_argument("--frames", type=int, default=100, help="Number of GIF frames.")
    parser.add_argument("--duration", type=float, default=0.08, help="GIF frame duration in seconds.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv).replace([np.inf, -np.inf], np.nan)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    stop_indices = np.linspace(8, len(df), max(2, args.frames)).astype(int)
    stop_indices = np.clip(stop_indices, 1, len(df))

    z_min = float(df[["gt_z", "wls_z", "tc_z"]].min().min()) - 0.08
    z_max = float(df[["gt_z", "wls_z", "tc_z"]].max().max()) + 0.08
    xlim = (float(df["timestamp"].min()), float(df["timestamp"].max()))
    ylim = (z_min, z_max)

    frames = [render_frame(df, stop, xlim, ylim) for stop in stop_indices]
    imageio.mimsave(args.out, frames, duration=args.duration)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
