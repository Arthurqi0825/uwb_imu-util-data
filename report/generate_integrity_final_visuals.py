#!/usr/bin/env python3
"""Generate final TC integrity report visuals.

Default input:
  ../output/260513_const4_1_2_1_uwb_imu_tc_integrity_final.csv

Outputs are written under:
  integrity_final_<today>/figures/

Generated assets:
  - static XY trajectory figure
  - static position/z/integrity error figure
  - XY trajectory GIF
  - Z trajectory + vertical error GIF
  - position error GIF
  - integrity diagnostics GIF
  - summary CSV
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter


SCRIPT_DIR = Path(__file__).resolve().parent
PKG_DIR = SCRIPT_DIR.parent
DEFAULT_CSV = PKG_DIR / "output" / "260513_const4_1_2_1_uwb_imu_tc_integrity_final.csv"


def finite_series(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name not in df:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def stats(values: pd.Series | np.ndarray) -> dict[str, float]:
    arr = pd.Series(values).dropna().to_numpy(dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "std": float(np.std(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def prepare_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = [
        "timestamp", "gt_x", "gt_y", "gt_z",
        "wls_x", "wls_y", "wls_z", "tc_x", "tc_y", "tc_z",
        "wls_gt_err", "tc_gt_err",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    for prefix in ("wls", "tc"):
        df[f"{prefix}_x_err"] = df[f"{prefix}_x"] - df["gt_x"]
        df[f"{prefix}_y_err"] = df[f"{prefix}_y"] - df["gt_y"]
        df[f"{prefix}_z_err"] = df[f"{prefix}_z"] - df["gt_z"]

    return df.dropna(subset=required).reset_index(drop=True)


def make_output_dir(base: Path | None, tag: str) -> tuple[Path, Path]:
    out_dir = base if base is not None else SCRIPT_DIR / f"{tag}_{date.today().isoformat()}"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, fig_dir


def save_summary(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for label, err_col in [("WLS (UWB only)", "wls_gt_err"), ("TC (UWB+IMU)", "tc_gt_err")]:
        row = {"estimator": label}
        row.update(stats(df[err_col]))
        rows.append(row)

    for prefix, label in [("wls", "WLS"), ("tc", "TC")]:
        row = {"estimator": f"{label} per-axis"}
        for axis in "xyz":
            s = df[f"{prefix}_{axis}_err"]
            row[f"{axis}_bias_mean"] = float(s.mean())
            row[f"{axis}_rmse"] = float(np.sqrt(np.mean(s * s)))
            row[f"{axis}_p95_abs"] = float(s.abs().quantile(0.95))
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_dir / "summary_metrics.csv", index=False)


def static_xy(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(df["gt_x"], df["gt_y"], "k-", lw=2.2, label="Ground truth")
    ax.plot(df["wls_x"], df["wls_y"], color="tab:red", lw=1.2, alpha=0.55, label="WLS")
    ax.plot(df["tc_x"], df["tc_y"], color="tab:blue", lw=1.8, label="TC")
    ax.set_title("XY Trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def static_error_integrity(df: pd.DataFrame, path: Path) -> None:
    t = df["timestamp"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

    axes[0, 0].plot(t, df["wls_gt_err"], color="tab:red", alpha=0.7, label="WLS")
    axes[0, 0].plot(t, df["tc_gt_err"], color="tab:blue", label="TC")
    axes[0, 0].set_title("3D Position Error")
    axes[0, 0].set_ylabel("m")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(t, df["wls_z_err"], color="tab:red", alpha=0.7, label="WLS z err")
    axes[0, 1].plot(t, df["tc_z_err"], color="tab:blue", label="TC z err")
    axes[0, 1].axhline(0, color="k", lw=0.8)
    axes[0, 1].set_title("Vertical Error")
    axes[0, 1].set_ylabel("m")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(t, finite_series(df, "integ_chi2"), color="tab:blue", label="chi2")
    axes[1, 0].plot(t, finite_series(df, "integ_threshold"), "r--", label="threshold")
    axes[1, 0].set_title("Integrity Chi-square")
    axes[1, 0].set_ylabel("statistic")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(t, finite_series(df, "integ_hpl"), color="tab:orange", label="HPL")
    axes[1, 1].plot(t, finite_series(df, "integ_vpl"), color="tab:purple", label="VPL")
    if "integ_hal" in df:
        axes[1, 1].plot(t, finite_series(df, "integ_hal"), color="tab:orange", ls="--", label="HAL")
    if "integ_val" in df:
        axes[1, 1].plot(t, finite_series(df, "integ_val"), color="tab:purple", ls="--", label="VAL")
    axes[1, 1].set_title("Protection Levels")
    axes[1, 1].set_ylabel("m")
    axes[1, 1].legend(fontsize=8)

    axes[2, 0].step(t, finite_series(df, "integ_available", 0), where="post", label="available")
    axes[2, 0].step(t, finite_series(df, "integ_fault", 0), where="post", label="fault")
    axes[2, 0].step(t, finite_series(df, "integ_excluded_count", 0), where="post", label="excluded count")
    axes[2, 0].set_title("Availability / Fault / Exclusion")
    axes[2, 0].set_xlabel("time (s)")
    axes[2, 0].legend(fontsize=8)

    axes[2, 1].plot(t, finite_series(df, "integ_input_count"), label="input")
    axes[2, 1].plot(t, finite_series(df, "integ_factor_count"), label="accepted")
    axes[2, 1].set_title("TDOA Factor Count")
    axes[2, 1].set_xlabel("time (s)")
    axes[2, 1].set_ylabel("count")
    axes[2, 1].legend(fontsize=8)

    for ax in axes.ravel():
        ax.grid(True, alpha=0.35)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def frame_indices(n_rows: int, max_frames: int) -> np.ndarray:
    return np.unique(np.linspace(0, n_rows - 1, min(max_frames, n_rows)).astype(int))


def save_animation(fig, update, frames: np.ndarray, path: Path, fps: int) -> None:
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def xy_trajectory_gif(df: pd.DataFrame, path: Path, max_frames: int, fps: int) -> None:
    idx = frame_indices(len(df), max_frames)
    fig, ax = plt.subplots(figsize=(7, 7))
    pad = 0.15
    ax.set_xlim(min(df.gt_x.min(), df.wls_x.min(), df.tc_x.min()) - pad,
                max(df.gt_x.max(), df.wls_x.max(), df.tc_x.max()) + pad)
    ax.set_ylim(min(df.gt_y.min(), df.wls_y.min(), df.tc_y.min()) - pad,
                max(df.gt_y.max(), df.wls_y.max(), df.tc_y.max()) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("XY Trajectory Replay")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.35)

    gt_line, = ax.plot([], [], "k-", lw=2, label="GT")
    wls_line, = ax.plot([], [], color="tab:red", lw=1.3, alpha=0.7, label="WLS")
    tc_line, = ax.plot([], [], color="tab:blue", lw=1.7, label="TC")
    gt_dot, = ax.plot([], [], "ko", ms=5)
    wls_dot, = ax.plot([], [], "o", color="tab:red", ms=5)
    tc_dot, = ax.plot([], [], "o", color="tab:blue", ms=5)
    text = ax.text(0.02, 0.96, "", transform=ax.transAxes)
    ax.legend(loc="best", fontsize=8)

    def update(k):
        i = idx[k]
        s = df.iloc[:i + 1]
        gt_line.set_data(s.gt_x, s.gt_y)
        wls_line.set_data(s.wls_x, s.wls_y)
        tc_line.set_data(s.tc_x, s.tc_y)
        gt_dot.set_data([df.gt_x.iloc[i]], [df.gt_y.iloc[i]])
        wls_dot.set_data([df.wls_x.iloc[i]], [df.wls_y.iloc[i]])
        tc_dot.set_data([df.tc_x.iloc[i]], [df.tc_y.iloc[i]])
        text.set_text(f"t = {df.timestamp.iloc[i]:.1f} s")
        return gt_line, wls_line, tc_line, gt_dot, wls_dot, tc_dot, text

    save_animation(fig, update, idx, path, fps)


def z_trajectory_error_gif(df: pd.DataFrame, path: Path, max_frames: int, fps: int) -> None:
    idx = frame_indices(len(df), max_frames)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    t0, t1 = df.timestamp.iloc[0], df.timestamp.iloc[-1]
    zmin = min(df.gt_z.min(), df.wls_z.min(), df.tc_z.min()) - 0.15
    zmax = max(df.gt_z.max(), df.wls_z.max(), df.tc_z.max()) + 0.15
    err_lim = max(df.wls_z_err.abs().max(), df.tc_z_err.abs().max()) * 1.1

    axes[0].set_xlim(t0, t1)
    axes[0].set_ylim(zmin, zmax)
    axes[0].set_title("Z Trajectory Replay")
    axes[0].set_ylabel("z (m)")
    axes[1].set_xlim(t0, t1)
    axes[1].set_ylim(-err_lim, err_lim)
    axes[1].set_title("Vertical Error Replay")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("z error (m)")
    axes[1].axhline(0, color="k", lw=0.8)
    for ax in axes:
        ax.grid(True, alpha=0.35)

    gt_z, = axes[0].plot([], [], "k-", lw=2, label="GT")
    wls_z, = axes[0].plot([], [], color="tab:red", alpha=0.7, label="WLS")
    tc_z, = axes[0].plot([], [], color="tab:blue", label="TC")
    wls_e, = axes[1].plot([], [], color="tab:red", alpha=0.7, label="WLS z err")
    tc_e, = axes[1].plot([], [], color="tab:blue", label="TC z err")
    text = axes[1].text(0.02, 0.9, "", transform=axes[1].transAxes)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    def update(k):
        i = idx[k]
        s = df.iloc[:i + 1]
        gt_z.set_data(s.timestamp, s.gt_z)
        wls_z.set_data(s.timestamp, s.wls_z)
        tc_z.set_data(s.timestamp, s.tc_z)
        wls_e.set_data(s.timestamp, s.wls_z_err)
        tc_e.set_data(s.timestamp, s.tc_z_err)
        text.set_text(f"t = {df.timestamp.iloc[i]:.1f} s")
        return gt_z, wls_z, tc_z, wls_e, tc_e, text

    save_animation(fig, update, idx, path, fps)


def position_error_gif(df: pd.DataFrame, path: Path, max_frames: int, fps: int) -> None:
    idx = frame_indices(len(df), max_frames)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.set_xlim(df.timestamp.iloc[0], df.timestamp.iloc[-1])
    ax.set_ylim(0, max(df.wls_gt_err.max(), df.tc_gt_err.max()) * 1.1)
    ax.set_title("3D Position Error Replay")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("error (m)")
    ax.grid(True, alpha=0.35)

    wls_line, = ax.plot([], [], color="tab:red", alpha=0.75, label="WLS")
    tc_line, = ax.plot([], [], color="tab:blue", label="TC")
    text = ax.text(0.02, 0.9, "", transform=ax.transAxes)
    ax.legend(fontsize=8)

    def update(k):
        i = idx[k]
        s = df.iloc[:i + 1]
        wls_line.set_data(s.timestamp, s.wls_gt_err)
        tc_line.set_data(s.timestamp, s.tc_gt_err)
        text.set_text(f"t = {df.timestamp.iloc[i]:.1f} s")
        return wls_line, tc_line, text

    save_animation(fig, update, idx, path, fps)


def integrity_gif(df: pd.DataFrame, path: Path, max_frames: int, fps: int) -> None:
    idx = frame_indices(len(df), max_frames)
    t0, t1 = df.timestamp.iloc[0], df.timestamp.iloc[-1]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax in axes:
        ax.set_xlim(t0, t1)
        ax.grid(True, alpha=0.35)

    chi2 = finite_series(df, "integ_chi2")
    thresh = finite_series(df, "integ_threshold")
    hpl = finite_series(df, "integ_hpl")
    vpl = finite_series(df, "integ_vpl")
    avail = finite_series(df, "integ_available", 0)
    fault = finite_series(df, "integ_fault", 0)
    excluded = finite_series(df, "integ_excluded_count", 0)

    axes[0].set_ylim(0, max(chi2.max(), thresh.max()) * 1.1)
    axes[0].set_title("Chi-square Integrity Test")
    axes[0].set_ylabel("stat")
    axes[1].set_ylim(0, max(hpl.max(), vpl.max(), finite_series(df, "integ_val", 0).max()) * 1.1)
    axes[1].set_title("Protection Levels")
    axes[1].set_ylabel("m")
    axes[2].set_ylim(-0.1, max(1.2, excluded.max() + 0.5))
    axes[2].set_title("Availability / Fault / Exclusion")
    axes[2].set_xlabel("time (s)")

    chi2_l, = axes[0].plot([], [], color="tab:blue", label="chi2")
    thr_l, = axes[0].plot([], [], "r--", label="threshold")
    hpl_l, = axes[1].plot([], [], color="tab:orange", label="HPL")
    vpl_l, = axes[1].plot([], [], color="tab:purple", label="VPL")
    avail_l, = axes[2].step([], [], where="post", color="tab:green", label="available")
    fault_l, = axes[2].step([], [], where="post", color="tab:red", label="fault")
    excl_l, = axes[2].step([], [], where="post", color="tab:gray", label="excluded")
    text = axes[2].text(0.02, 0.87, "", transform=axes[2].transAxes)
    for ax in axes:
        ax.legend(fontsize=8)

    def update(k):
        i = idx[k]
        ts = df.timestamp.iloc[:i + 1]
        chi2_l.set_data(ts, chi2.iloc[:i + 1])
        thr_l.set_data(ts, thresh.iloc[:i + 1])
        hpl_l.set_data(ts, hpl.iloc[:i + 1])
        vpl_l.set_data(ts, vpl.iloc[:i + 1])
        avail_l.set_data(ts, avail.iloc[:i + 1])
        fault_l.set_data(ts, fault.iloc[:i + 1])
        excl_l.set_data(ts, excluded.iloc[:i + 1])
        text.set_text(f"t = {df.timestamp.iloc[i]:.1f} s")
        return chi2_l, thr_l, hpl_l, vpl_l, avail_l, fault_l, excl_l, text

    save_animation(fig, update, idx, path, fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final TC integrity figures and GIFs.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help=f"Input TC integrity CSV. Default: {DEFAULT_CSV}")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory. Default: report/integrity_final_<today>")
    parser.add_argument("--tag", default="integrity_final",
                        help="Output directory prefix when --out-dir is not set.")
    parser.add_argument("--max-frames", type=int, default=140,
                        help="Maximum frames per GIF.")
    parser.add_argument("--fps", type=int, default=12,
                        help="GIF frames per second.")
    args = parser.parse_args()

    df = prepare_df(args.csv)
    out_dir, fig_dir = make_output_dir(args.out_dir, args.tag)

    save_summary(df, out_dir)
    static_xy(df, fig_dir / "xy_trajectory.png")
    static_error_integrity(df, fig_dir / "error_and_integrity.png")
    xy_trajectory_gif(df, fig_dir / "xy_trajectory.gif", args.max_frames, args.fps)
    z_trajectory_error_gif(df, fig_dir / "z_trajectory_error.gif", args.max_frames, args.fps)
    position_error_gif(df, fig_dir / "position_error.gif", args.max_frames, args.fps)
    integrity_gif(df, fig_dir / "integrity_diagnostics.gif", args.max_frames, args.fps)

    print(f"[done] wrote report assets to: {out_dir}")
    print(f"[done] figures/GIFs: {fig_dir}")


if __name__ == "__main__":
    main()
