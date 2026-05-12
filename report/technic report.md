# Key Improvement: WLS-Derived Velocity Regularization for TC-FGO

## Background

Analysis of `uwb_imu_fusion/output/260427_const1_3_2_uwb_imu_trajectory_tc.csv`
showed that WLS and TC-FGO have similar overall error, but TC-FGO has a clear
failure transient around `12.4-12.8 s`.

Summary from the CSV:

- WLS mean error: `0.186 m`, RMSE: `0.225 m`
- TC-FGO mean error: `0.192 m`, RMSE: `0.235 m`
- Worst observed TC degradation: at `t=12.572405 s`, WLS error was `0.052 m`
  while TC-FGO error rose to `0.449 m`
- During that transient, TC velocity reached about `2.5 m/s`, while the local
  ground-truth motion was much slower

## Root Cause

The previous TC-FGO update added a velocity prior centered on
`pred_nav.velocity()`:

```cpp
graph.addPrior(V(k), pred_nav.velocity(),
               gtsam::noiseModel::Isotropic::Sigma(3, tc_update_vel_sigma_));
```

This does not actually correct an IMU velocity spike. It reinforces the same
IMU-predicted velocity that can already be wrong. Since TDOA directly observes
position but not velocity, attitude, or IMU bias, a bad velocity state can persist
long enough to pull the TC pose away from the accurate WLS position.

Dropout handling had a related issue: when IMU samples were missing, the velocity
prior used the stale TC velocity. That kept phantom velocity alive across dropout
cycles even when WLS was still valid.

## Implemented Change

The TC branch now stores the last valid WLS position and timestamp. On each new
valid WLS cycle, it computes a bounded WLS-derived velocity reference:

```cpp
v_wls = (wls_p(k) - wls_p(k-1)) / dt
```

The reference is accepted only when:

- WLS velocity prior is enabled
- a previous valid WLS sample exists
- `dt` is positive and below `tc_wls_velocity_max_dt`
- the speed is capped by `tc_wls_velocity_max_speed`

When valid, TC-FGO uses this WLS-derived velocity as the velocity prior mean
instead of the self-reinforcing IMU prediction. During IMU dropout, the same WLS
velocity reference is preferred over stale TC velocity.

New parameters:

```xml
<arg name="tc_use_wls_velocity_prior" default="true" />
<arg name="tc_wls_velocity_sigma"     default="0.25" />
<arg name="tc_wls_velocity_max_speed" default="1.2"  />
<arg name="tc_wls_velocity_max_dt"    default="0.25" />
```

## Expected Benefit

This keeps TC-FGO from developing unrealistic velocity during the exact failure
mode observed in `260427_const1_3_2`. It preserves the IMU factor for smoothing
and short-term propagation, but adds an external velocity sanity reference from
the UWB position track.

The change is intentionally conservative: WLS finite differences can be noisy, so
the inferred velocity is capped and ignored when the WLS time gap is stale.

## Follow-Up Validation

Rerun the same bag/log and compare:

- `tc_gt_err` around `12.4-12.8 s`
- `tc_vx`, `tc_vy`, `tc_vz` peak magnitude
- `WLS-TC` position separation
- number of `WLS velocity prior active` log messages

The target outcome is for the TC velocity peak near `12.5 s` to stay close to the
physical trajectory speed and for TC error to remain near the WLS error instead
of rising toward `0.45 m`.

## 260511 Follow-Up: Vertical Channel Fix

The `260511` output showed that the large velocity blow-up was mostly removed,
but TC-FGO remained slightly worse than WLS because of vertical lag during the
initial climb. The first WLS height was wrong, and the graph was using a tight
startup Z prior:

```xml
[ 0.5, 0.5, 0.5, 0.05, 0.05, 0.02 ]
```

This effectively pinned TC to a bad initial Z. The launch prior was loosened to:

```xml
[ 0.5, 0.5, 0.5, 0.05, 0.05, 0.20 ]
```

The WLS-derived velocity prior was also changed to regularize XY only. WLS
finite-difference Z velocity was very noisy in the 260511 run, so using it as a
tight Z velocity prior caused TC to lag below WLS/GT. Z velocity now remains
IMU-driven with the normal `tc_update_vel_sigma` instead of the tighter WLS
velocity sigma.
