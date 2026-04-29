# Technical Report Assets

This directory contains the generated report and reproducible result assets for `uwb_imu_fusion.launch`.

## Main Output

- `technical_report.tex`: LaTeX technical report.
- `metrics.json`: exact numerical metrics from `output/260427_const1_3_2_uwb_imu_trajectory_tc.csv`.
- `metrics_table.tex`: table included by the LaTeX report.
- `generate_report_assets.py`: script used to regenerate plots and metrics.

## Result Summary

| Estimator | Mean (m) | Median (m) | RMSE (m) | 95% (m) | Max (m) |
|---|---:|---:|---:|---:|---:|
| WLS UWB-only | 0.186 | 0.142 | 0.225 | 0.444 | 0.556 |
| TC UWB+IMU | 0.192 | 0.128 | 0.235 | 0.457 | 0.570 |

Dataset: 1,828 cycles over 52.944 s. TC has lower error on 46.8% of cycles. The median error improves with TC, while WLS has slightly better mean and RMSE on this run.

## Figures

![WLS TC comparison diagnostics](wls_tc_comparison_diagnostics.png)

![XY trajectory](trajectory_xy.png)

![Error over time](error_timeseries.png)

![Error CDF](error_cdf.png)

![State diagnostics](state_diagnostics.png)

![Error histogram](error_histogram.png)

## Animation

![Trajectory replay](trajectory_replay.gif)

## Regeneration

```bash
python3 report/generate_report_assets.py \
  --csv output/260427_const1_3_2_uwb_imu_trajectory_tc.csv \
  --out-dir report
```
