#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot UWB+IMU TC integrity outputs")
    p.add_argument(
        "--csv",
        default="uwb_imu_fusion/output/uwb_tdoa_imu_tc_integrity.csv",
        help="Path to CSV file",
    )
    p.add_argument(
        "--plots",
        nargs="+",
        default=["all"],
        choices=["trajectory", "error", "integrity", "all"],
        help="Which figure groups to generate",
    )
    p.add_argument("--show", action="store_true", help="Display plots interactively")
    p.add_argument("--save-dir", default="", help="Directory to save png files")
    return p.parse_args()


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Error components (use TC + WLS as requested; add WLS_clean if you want)
    for tag in ("wls", "wls_clean", "tc"):
        df[f"{tag}_ex"] = df[f"{tag}_x"] - df["gt_x"]
        df[f"{tag}_ey"] = df[f"{tag}_y"] - df["gt_y"]
        df[f"{tag}_ez"] = df[f"{tag}_z"] - df["gt_z"]

        # overall xyz norm
        df[f"{tag}_e_xyz"] = np.sqrt(
            df[f"{tag}_ex"] ** 2 + df[f"{tag}_ey"] ** 2 + df[f"{tag}_ez"] ** 2
        )
        # xy norm
        df[f"{tag}_e_xy"] = np.sqrt(df[f"{tag}_ex"] ** 2 + df[f"{tag}_ey"] ** 2)
        # z abs error
        df[f"{tag}_e_z"] = df[f"{tag}_ez"].abs()

    return df


def plot_trajectory(df: pd.DataFrame, out_dir: Path | None = None):
    t = df["timestamp"].to_numpy()

    # 1) XY trajectory
    fig_xy, ax1 = plt.subplots(figsize=(7, 6))
    ax1.plot(df["gt_x"], df["gt_y"], linewidth=2.2, label="GT")
    ax1.plot(df["wls_x"], df["wls_y"], alpha=0.9, label="WLS")
    ax1.plot(df["tc_x"], df["tc_y"], alpha=0.9, label="TC")
    ax1.set_title("Trajectory (XY)")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.axis("equal")
    ax1.grid(alpha=0.25)
    ax1.legend()
    if out_dir:
        fig_xy.savefig(out_dir / "01_trajectory_xy.png", dpi=220, bbox_inches="tight")

    # 1b) Z trajectory over time
    fig_z, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(t, df["gt_z"], linewidth=2.2, label="GT z")
    ax2.plot(t, df["wls_z"], label="WLS z")
    ax2.plot(t, df["tc_z"], label="TC z")
    ax2.set_title("Trajectory on Z")
    ax2.set_xlabel("timestamp [s]")
    ax2.set_ylabel("z [m]")
    ax2.grid(alpha=0.25)
    ax2.legend()
    if out_dir:
        fig_z.savefig(out_dir / "02_trajectory_z.png", dpi=220, bbox_inches="tight")

    return [fig_xy, fig_z]


def plot_error(df: pd.DataFrame, out_dir: Path | None = None):
    t = df["timestamp"].to_numpy()

    # 2) Overall error
    fig_all, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(t, df["wls_e_xyz"], label="WLS overall")
    ax1.plot(t, df["tc_e_xyz"], label="TC overall")
    ax1.set_title("Overall Position Error")
    ax1.set_xlabel("timestamp [s]")
    ax1.set_ylabel("error norm [m]")
    ax1.grid(alpha=0.25)
    ax1.legend()
    if out_dir:
        fig_all.savefig(out_dir / "03_error_overall.png", dpi=220, bbox_inches="tight")

    # 2b) Error on XY
    fig_xy, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(t, df["wls_e_xy"], label="WLS error XY")
    ax2.plot(t, df["tc_e_xy"], label="TC error XY")
    ax2.set_title("Position Error on XY")
    ax2.set_xlabel("timestamp [s]")
    ax2.set_ylabel("xy error norm [m]")
    ax2.grid(alpha=0.25)
    ax2.legend()
    if out_dir:
        fig_xy.savefig(out_dir / "04_error_xy.png", dpi=220, bbox_inches="tight")

    # 2c) Error on Z
    fig_z, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(t, df["wls_e_z"], label="WLS error Z")
    ax3.plot(t, df["tc_e_z"], label="TC error Z")
    ax3.set_title("Position Error on Z")
    ax3.set_xlabel("timestamp [s]")
    ax3.set_ylabel("abs z error [m]")
    ax3.grid(alpha=0.25)
    ax3.legend()
    if out_dir:
        fig_z.savefig(out_dir / "05_error_z.png", dpi=220, bbox_inches="tight")

    return [fig_all, fig_xy, fig_z]


def plot_integrity(df: pd.DataFrame, out_dir: Path | None = None):
    t = df["timestamp"].to_numpy()

    # 3) Integrity related
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(t, df["integ_chi2"], label="integ_chi2")
    ax1.plot(t, df["integ_threshold"], label="threshold", linestyle="--")
    ax1.plot(t, df["integ_soft_gate"], label="soft gate", alpha=0.8)
    ax1.plot(t, df["integ_hard_gate"], label="hard gate", alpha=0.8)
    ax1.set_title("Integrity: Chi-square and Gate")
    ax1.set_xlabel("timestamp [s]")
    ax1.set_ylabel("chi2 / gate values")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(t, df["integ_fault"], "k", alpha=0.5, label="fault")
    ax2.plot(t, df["integ_ring_ok"], "g", alpha=0.4, label="ring_ok")
    ax2.plot(t, df["integ_available"], "c", alpha=0.4, label="available")
    ax2.set_ylabel("integrity flags (0/1)")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper right")

    if out_dir:
        fig.savefig(out_dir / "06_integrity.png", dpi=220, bbox_inches="tight")

    return [fig]


def main():
    args = parse_args()
    df = load_data(args.csv)

    out_dir = Path(args.save_dir) if args.save_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    requested = args.plots
    if "all" in requested:
        requested = ["trajectory", "error", "integrity"]

    figs = []
    if "trajectory" in requested:
        figs.extend(plot_trajectory(df, out_dir))
    if "error" in requested:
        figs.extend(plot_error(df, out_dir))
    if "integrity" in requested:
        figs.extend(plot_integrity(df, out_dir))

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()