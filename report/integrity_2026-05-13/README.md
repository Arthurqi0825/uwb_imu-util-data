# Integrity Report - 2026-05-13

Generated from:

- `output/uwb_imu_tc_integrity.csv`
- `output/260513_const4_1_2_1uwb_wls_integrity.csv`

## Executive Summary

The current TC integrity run is slightly better than its own WLS baseline in 3D RMSE: TC `0.1711 m` vs WLS `0.1821 m`. The LC/WLS integrity run still has the best fused result in this report: LC `0.1434 m` vs its WLS baseline `0.1768 m`.

TC integrity did not exclude measurements in the current log, while the WLS integrity run performed substantial NLOS/FDE filtering. This is visible in the exclusion summary and is still the main area for future TC improvement.

## Position Error Statistics

| file                                     | estimator      | mean   | median | rmse   | std    | max    | p95    |
| ---------------------------------------- | -------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| uwb_imu_tc_integrity.csv                 | WLS (UWB only) | 0.1534 | 0.1319 | 0.1821 | 0.0983 | 1.1533 | 0.3478 |
| uwb_imu_tc_integrity.csv                 | TC (UWB+IMU)   | 0.1513 | 0.1336 | 0.1711 | 0.0798 | 0.7408 | 0.3139 |
| 260513_const4_1_2_1uwb_wls_integrity.csv | WLS (UWB only) | 0.1500 | 0.1251 | 0.1768 | 0.0935 | 0.6474 | 0.3460 |
| 260513_const4_1_2_1uwb_wls_integrity.csv | LC (UWB+IMU)   | 0.1236 | 0.1120 | 0.1434 | 0.0727 | 0.5441 | 0.2503 |

## Integrity Summary

| file                                     | kind | rows | duration_s | available_rows | fault_rows | hpl_mean | hpl_p95 | vpl_mean | vpl_p95 | rows_with_exclusions |
| ---------------------------------------- | ---- | ---- | ---------- | -------------- | ---------- | -------- | ------- | -------- | ------- | -------------------- |
| uwb_imu_tc_integrity.csv                 | tc   | 1585 | 107.4049   | 1580           | 0          | 0.3773   | 0.5610  | 0.6945   | 1.0922  | 0                    |
| 260513_const4_1_2_1uwb_wls_integrity.csv | lc   | 2953 | 70.0206    | 2748           | 53         | 0.8130   | 0.9985  | 1.2308   | 1.8499  | 811                  |

## Pair Exclusions

- TC excluded pairs: none
- WLS/LC excluded or NLOS pairs: 2-3 (455), anchor2:1-2 (358), anchor3:2-3 (194), 3-4 (194), 6-7 (192), anchor6:5-6 (124), anchor3:3-4 (97), 0-1 (96)

## Visuals

- Overview: [figures/integrity_comparison_overview.png](figures/integrity_comparison_overview.png)
- TC trajectory analysis: [figures/uwb_imu_tc_integrity_trajectory.png](figures/uwb_imu_tc_integrity_trajectory.png)
- TC integrity diagnostics: [figures/uwb_imu_tc_integrity_integrity.png](figures/uwb_imu_tc_integrity_integrity.png)
- WLS/LC trajectory analysis: [figures/260513_const4_1_2_1uwb_wls_integrity_trajectory.png](figures/260513_const4_1_2_1uwb_wls_integrity_trajectory.png)
- WLS/LC integrity diagnostics: [figures/260513_const4_1_2_1uwb_wls_integrity_integrity.png](figures/260513_const4_1_2_1uwb_wls_integrity_integrity.png)
- WLS/LC excluded pairs: [figures/260513_const4_1_2_1uwb_wls_integrity_excluded_pairs.png](figures/260513_const4_1_2_1uwb_wls_integrity_excluded_pairs.png)
- Animation GIF: [figures/tc_integrity_trajectory.gif](figures/tc_integrity_trajectory.gif)

## Notes

The current TC CSV reflects the corrected WLS-following vertical prior: the earlier ~1 m height lock problem is gone. Remaining TC error is mostly horizontal spread plus residual WLS-Z bias, while LC benefits from explicit NLOS rejection in the WLS integrity pipeline.
