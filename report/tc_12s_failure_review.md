# TC Error Spike Around 12 s

## Symptom

The large TC error near 12 s is not caused by WLS failure. In the main failure window,
`12.35-12.75 s`, WLS remains close to ground truth while TC drifts away:

| Window | WLS mean error | TC mean error | Mean TC-WLS delta |
|---|---:|---:|---:|
| 11-12 s | 0.078 m | 0.055 m | -0.023 m |
| 12.35-12.75 s | 0.081 m | 0.421 m | +0.340 m |
| 13-14 s | 0.091 m | 0.092 m | +0.001 m |

Worst example:

| Time | WLS error | TC error | WLS-TC XY gap | IMU samples |
|---:|---:|---:|---:|---:|
| 12.572405 s | 0.052 m | 0.449 m | 0.395 m | 17 |

This is a TC-specific transient.

## Most Likely Cause

The TC velocity and attitude become physically implausible before and during the spike.

From approximately `12.0-12.42 s`, ground truth moves at about `0.30 m/s`, while the TC estimate moves at about `1.20 m/s`. During `12.35-12.75 s`, the logged TC state reports velocities near:

```text
vx ~= -2.1 m/s
vy ~= -0.87 m/s
vz ~= -0.64 m/s
speed ~= 2.4 m/s
```

That is much larger than the ground-truth motion. The attitude estimate is also unstable. Around this interval, roll/pitch briefly reach tens of degrees even though the mean gyro norm is only about `0.15 rad/s`.

Example sampled rows:

```text
t=12.007 s: roll=-52.3 deg, pitch=-28.8 deg, yaw= 73.6 deg
t=12.090 s: roll= 38.8 deg, pitch=-29.8 deg, yaw=-45.0 deg
t=12.223 s: roll= 37.6 deg, pitch= 25.2 deg, yaw=-105.8 deg
t=12.356 s: roll= 34.0 deg, pitch=-14.8 deg, yaw=-51.0 deg
```

This strongly suggests an attitude/IMU-preintegration issue rather than a UWB geometry issue. Once attitude tilts incorrectly, gravity is projected into horizontal acceleration, producing excessive velocity and position drift. The WLS prior is present, but it is not strong enough to immediately pull the graph back while the IMU factor and velocity prior keep supporting the bad dynamic state.

## Code Paths Involved

Relevant implementation points:

- `imuCallback()` converts IMU units and buffers samples.
- `tcUpdateGraph()` resets/replays IMU preintegration, predicts `pred_nav`, adds `ImuFactor`, then adds velocity and bias priors.
- `addTcWlsPositionPrior()` adds the WLS pose prior with `xy_sigma=0.25 m`.
- `wlsPassesInnovationGate()` only rejects WLS if WLS and TC differ by more than `0.8 m` in XY.

In the bad window the WLS-TC XY gap is about `0.21-0.41 m`, so it passes the `0.8 m` gate. However, the TC state still remains far from WLS/GT, meaning the current graph weighting allows the IMU/dynamic state to dominate for several cycles.

## Recommended Improvements

1. Add a diagnostic log column for attitude.

   The current CSV logs quaternion, but not roll/pitch/yaw or tilt magnitude. Add `tc_roll`, `tc_pitch`, `tc_yaw`, and `tc_tilt_deg` so attitude failures are directly visible.

2. Add a diagnostic log column for WLS-TC innovation.

   Log `wls_tc_xy_gap`, `wls_tc_z_gap`, and whether the WLS prior was accepted. This will make it obvious when the graph is ignoring a good WLS estimate.

3. Add an attitude sanity gate.

   If roll/pitch changes too much relative to integrated gyro magnitude, temporarily reduce IMU authority or add a soft attitude prior to the previous/predicted attitude.

4. Add a speed or acceleration sanity gate.

   When TC speed becomes implausible relative to recent WLS/GT-derived motion, increase IMU noise or strengthen the WLS prior for a few cycles.

5. Consider stronger WLS correction when innovation is moderate but persistent.

   Current normal WLS prior is `xy_sigma=0.25 m`. In this failure, the WLS-TC gap is large enough to matter but below the `0.8 m` rejection gate. A middle mode could use stronger WLS anchoring when:

   ```text
   WLS is stable across several cycles
   WLS-TC XY gap > 0.25 m
   TC speed is implausibly high
   ```

6. Revisit IMU factor weighting during high-attitude-transient periods.

   The configured accelerometer noise, `0.04 m/s^2`, matches static noise but may be too confident during dynamic motion or unobservable attitude periods. A larger acceleration sigma or adaptive IMU noise may prevent gravity-projection errors from dominating the graph.

## Bottom Line

The 12 s error spike is best explained as a TC attitude/velocity transient. WLS is accurate in the same interval, but TC develops an excessive velocity and large roll/pitch excursions. The immediate code improvement is better diagnostics; the estimator improvement is adaptive trust: when attitude/velocity becomes physically inconsistent but WLS remains stable, temporarily weaken IMU propagation and/or strengthen the WLS position prior.
