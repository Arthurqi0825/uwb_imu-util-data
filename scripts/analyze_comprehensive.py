#!/usr/bin/env python3
"""Analyzer/plotter for UWB-IMU fusion trajectory CSV outputs.

Auto-detects TC vs LC trajectory CSVs by header columns and produces a
single 2x2 figure per file containing:
  (1) XY trajectory
  (2) 3D trajectory
  (3) Position error vs time (WLS vs fused)
  (4) Per-axis trajectory error (x, y, z) of the fused estimate

Also prints summary error statistics to stdout.
"""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------- helpers -----------------------------------

def detect_kind(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if {'tc_x', 'tc_y', 'tc_z'}.issubset(cols):
        return 'tc'
    if {'lc_x', 'lc_y', 'lc_z'}.issubset(cols):
        return 'lc'
    return 'unknown'


def err_stats(e: pd.Series) -> dict:
    e = e.dropna().to_numpy()
    return {
        'mean':   float(np.mean(e)),
        'median': float(np.median(e)),
        'rmse':   float(np.sqrt(np.mean(e ** 2))),
        'std':    float(np.std(e)),
        'max':    float(np.max(e)),
        'p95':    float(np.percentile(e, 95)),
    }


def per_axis_err(df: pd.DataFrame, prefix: str):
    return (
        df[f'{prefix}_x'] - df['gt_x'],
        df[f'{prefix}_y'] - df['gt_y'],
        df[f'{prefix}_z'] - df['gt_z'],
    )


def print_stats_table(stats: dict):
    table = pd.DataFrame(stats).T[['mean', 'median', 'rmse', 'std', 'max', 'p95']]
    print('\n=== Position error statistics (m) ===')
    print(table.to_string(float_format=lambda v: f'{v:0.4f}'))


# ------------------------------ trajectory ---------------------------------

def analyze_trajectory(df: pd.DataFrame, kind: str, title: str, save_path):
    fused = kind  # 'tc' or 'lc'
    fused_label = 'TC (UWB+IMU)' if kind == 'tc' else 'LC (UWB+IMU)'
    t = df['timestamp'].to_numpy()
    fu_ex, fu_ey, fu_ez = per_axis_err(df, fused)

    print_stats_table({
        'WLS (UWB only)': err_stats(df['wls_gt_err']),
        fused_label:      err_stats(df[f'{fused}_gt_err']),
    })

    fig = plt.figure(figsize=(13, 10))
    fig.suptitle(f'Trajectory analysis — {title}', fontsize=13)

    # (1) XY trajectory
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(df['gt_x'], df['gt_y'], 'k-', lw=2, label='Ground truth')
    ax.plot(df['wls_x'], df['wls_y'], 'r.', ms=2, alpha=0.3, label='WLS')
    ax.plot(df[f'{fused}_x'], df[f'{fused}_y'], 'b-', alpha=0.85, label=fused_label)
    ax.set_title('XY trajectory'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_aspect('equal', adjustable='datalim'); ax.grid(True); ax.legend()

    # (2) 3D trajectory
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.plot(df['gt_x'], df['gt_y'], df['gt_z'], 'k-', lw=2, label='Ground truth')
    ax.plot(df['wls_x'], df['wls_y'], df['wls_z'], 'r.', ms=1.5, alpha=0.3, label='WLS')
    ax.plot(df[f'{fused}_x'], df[f'{fused}_y'], df[f'{fused}_z'],
            'b-', alpha=0.85, label=fused_label)
    ax.set_title('3D trajectory')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.legend(loc='upper left', fontsize=8)

    # (3) Total position error vs time
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(t, df['wls_gt_err'], 'r', alpha=0.5, label='WLS error')
    ax.plot(t, df[f'{fused}_gt_err'], 'b', alpha=0.85, label=f'{fused_label} error')
    ax.set_title('Position error vs time')
    ax.set_xlabel('t (s)'); ax.set_ylabel('|err| (m)')
    ax.grid(True); ax.legend()

    # (4) Per-axis (x, y, z) error of fused estimate
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(t, fu_ex, 'r', alpha=0.85, label='err x')
    ax.plot(t, fu_ey, 'g', alpha=0.85, label='err y')
    ax.plot(t, fu_ez, 'b', alpha=0.85, label='err z')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f'{fused_label}: per-axis error')
    ax.set_xlabel('t (s)'); ax.set_ylabel('err (m)')
    ax.grid(True); ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f'[saved] {save_path}')


# --------------------------------- main ------------------------------------

def process_file(path: str, save_dir):
    print(f'\n########## {os.path.basename(path)} ##########')
    df = pd.read_csv(path)
    kind = detect_kind(df)
    duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
    print(f'kind={kind}, rows={len(df)}, duration={duration:.2f}s')

    if kind not in ('tc', 'lc'):
        print(f'[skip] not a trajectory CSV (cols={list(df.columns)[:6]}…)')
        return

    base = os.path.splitext(os.path.basename(path))[0]
    save_path = os.path.join(save_dir, f'{base}_analysis.png') if save_dir else None
    analyze_trajectory(df, kind, base, save_path)


def main():
    p = argparse.ArgumentParser(description='UWB-IMU trajectory CSV analyzer.')
    default_out = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output'))
    p.add_argument('inputs', nargs='*',
                   help=f'CSV files or directories. Default: {default_out}')
    p.add_argument('--save-dir', default=None,
                   help='Directory to save figures (PNG). Omit to only show.')
    p.add_argument('--no-show', action='store_true', help='Do not call plt.show()')
    args = p.parse_args()

    targets = args.inputs or [default_out]
    files = []
    for t in targets:
        if os.path.isdir(t):
            files.extend(sorted(glob(os.path.join(t, '*.csv'))))
        elif os.path.isfile(t):
            files.append(t)
        else:
            print(f'[warn] not found: {t}', file=sys.stderr)

    if not files:
        print('No CSV files to process.'); sys.exit(1)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for f in files:
        try:
            process_file(f, args.save_dir)
        except Exception as e:
            print(f'[error] {f}: {e}', file=sys.stderr)

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
