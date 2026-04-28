# UWB-IMU Fusion Technical Report

## 1. Overview

This package implements a ROS node for fusing:

- UWB TDOA measurements
- IMU measurements
- optional ground-truth pose for evaluation

The node publishes two motion estimates:

- a **WLS-only UWB position estimate**
- a **tightly coupled factor-graph fusion estimate**

The implementation combines a fast front-end based on weighted least squares (WLS) with a back-end based on **GTSAM**, **IMU preintegration**, and **incremental smoothing with iSAM2**.

At a high level:

1. IMU samples are buffered continuously.
2. UWB TDOA samples are grouped into short measurement cycles.
3. Each cycle is solved once with a standalone WLS position estimator.
4. The same cycle is also added to a tightly coupled factor graph using:
   - raw TDOA factors
   - IMU preintegration
   - bias evolution constraints
   - optional WLS and vertical priors

This design gives the system two useful behaviors:

- WLS provides a direct geometric position estimate from UWB alone.
- The tightly coupled graph uses raw UWB and IMU jointly, which is usually smoother and more dynamically consistent.

---

## 2. Node Architecture

The main implementation is in [`src/uwb_imu_fusion_node.cpp`](./src/uwb_imu_fusion_node.cpp).

Core internal components:

- `ImuMeas`
  - stores timestamped IMU acceleration and angular velocity in SI units
- `FgoState`
  - stores the current tightly coupled graph state:
    - pose
    - velocity
    - IMU bias
    - cycle index
    - initialization status
- `TdoaWlsSolver`
  - iterative Gauss-Newton solver for UWB-only position
- `TdoaFactor`
  - custom GTSAM unary factor for one TDOA measurement

Main processing flow:

1. `imuCallback()`
   - converts raw IMU values to SI units
   - performs unit sanity checking
   - buffers IMU samples
   - optionally estimates initial roll/pitch from gravity
2. `uwbCallback()`
   - validates incoming TDOA measurements
   - accumulates them into a cycle
   - triggers `processCycle()` once enough measurements are available
3. `processCycle()`
   - computes cycle timestamp
   - computes WLS estimate
   - initializes or updates tightly coupled factor graph
   - publishes odometry, path, TF, and logs

---

## 3. Measurement and State Models

### 3.1 State representation

For each factor-graph keyframe `k`, the tightly coupled estimator maintains:

- pose `X(k)` as `gtsam::Pose3`
- velocity `V(k)` as `gtsam::Vector3`
- IMU bias `B(k)` as `gtsam::imuBias::ConstantBias`

This is the standard inertial navigation state used by GTSAM IMU factor graphs.

### 3.2 UWB TDOA measurement model

Each TDOA measurement encodes a range difference between two anchors:

`tdoa = d(tag, B) - d(tag, A)`

where:

- `A`, `B` are anchor indices
- `d(tag, anchor)` is Euclidean distance

The implementation uses the validated residual sign convention:

`r_i = (||p - a_B|| - ||p - a_A||) - z_i`

where:

- `p` is tag position
- `a_A`, `a_B` are anchor coordinates
- `z_i` is measured TDOA expressed in meters

This sign convention is used in:

- [`include/wlssolver.h`](./include/wlssolver.h)
- [`include/tdoa_factor.h`](./include/tdoa_factor.h)

Consistency here is critical. A sign mismatch would bias both the WLS solution and the factor graph.

### 3.3 IMU model

The IMU contributes:

- linear acceleration
- angular velocity

The code assumes raw sensor data may arrive in non-SI units and converts it before use:

- acceleration: `g -> m/s^2` using `accel_scale_`
- gyroscope: `deg/s -> rad/s` when `gyro_is_degrees_ = true`

After conversion, IMU samples are integrated using GTSAM preintegration.

---

## 4. WLS Front-End

### 4.1 Purpose

The WLS solver provides a direct position estimate from UWB TDOA data only. It serves as:

- a publishable standalone UWB solution
- an initialization seed
- an optional soft prior for the tightly coupled graph

### 4.2 Algorithm

The WLS solver in [`include/wlssolver.h`](./include/wlssolver.h) uses **iterative Gauss-Newton**.

For each iteration:

1. Build residual vector `r`
2. Build Jacobian matrix `J`
3. Solve normal equations

`(J^T J) dp = -J^T r`

4. Update position estimate

`p <- p + dp`

5. Stop when `||dp|| < tol`

### 4.3 Jacobian

For one TDOA residual:

`r_i = (||p-a_B|| - ||p-a_A||) - z_i`

the Jacobian with respect to position is:

`dr_i/dp = (p-a_B)/||p-a_B|| - (p-a_A)/||p-a_A||`

This is exactly what the implementation constructs.

### 4.4 Numerical safeguards

The solver includes:

- minimum measurement count requirement
- minimum denominator distance to avoid division by zero
- maximum iteration limit

These are simple but important protections for field data.

---

## 5. Tightly Coupled Factor Graph

### 5.1 Motivation

A WLS-only solution uses only instantaneous UWB geometry. It does not exploit motion continuity or inertial dynamics.

The tightly coupled graph instead fuses:

- raw TDOA measurements
- IMU integration between keyframes
- evolving IMU bias
- optional WLS and vertical priors

This allows the estimator to use the full measurement structure rather than treating WLS as the only UWB product.

### 5.2 Back-end engine

The graph is optimized incrementally using **GTSAM iSAM2**.

The node configures:

- IMU preintegration parameters
- TDOA noise model
- robust kernel option
- ISAM2 relinearization behavior

The graph is updated once per UWB cycle.

### 5.3 Factors used

For normal operation, the graph update includes:

1. `ImuFactor`
   - links previous and current pose/velocity using integrated IMU
2. `BetweenFactor` on IMU bias
   - models bias random walk
3. Prior on current velocity
   - regularizes velocity around the IMU-predicted value
4. Prior on current bias
   - regularizes bias evolution
5. One `TdoaFactor` per raw TDOA measurement
6. Optional WLS position prior
7. Optional vertical prior

This makes the approach **tightly coupled**, because raw TDOA measurements are inserted directly as graph factors instead of first collapsing them into a single fused position measurement.

---

## 6. Custom TDOA Factor

The custom factor is defined in [`include/tdoa_factor.h`](./include/tdoa_factor.h).

It is a unary GTSAM factor on `Pose3`, but only the translation part affects the residual.

### 6.1 Residual

The factor predicts:

`predicted = ||p-a_A|| - ||p-a_B||`

and returns:

`error = predicted - measured`

In the node, the measurement is inserted as `-m.tdoa` so that the graph matches the chosen sign convention.

### 6.2 Jacobian

The factor computes:

- gradient of range difference with respect to world position
- translation Jacobian of `Pose3`
- full `1x6` Jacobian with respect to pose tangent variables

This is a correct and practical way to connect a translation-only measurement to a `Pose3` state in GTSAM.

---

## 7. IMU Preintegration

### 7.1 Why preintegration is used

IMU arrives at a much higher rate than UWB. Instead of adding one graph factor per IMU sample, the node integrates all IMU samples between two UWB cycle times into one compact motion constraint.

This is the role of `PreintegratedImuMeasurements`.

### 7.2 Processing steps

For each cycle update:

1. Reset preintegrator using current bias estimate
2. Replay IMU samples in `[last_cycle_t, t_mid]`
3. Integrate each valid sample using its `dt`
4. Build one `ImuFactor`
5. Predict next navigation state from current state and bias

The predicted state is also used as the initial guess for the next graph node.

### 7.3 Bias evolution

Bias is modeled as a random walk through `tcBiasBetween(...)`.

The standard deviation scales with `sqrt(dt)`, which is consistent with continuous-time random walk discretization.

---

## 8. Initialization Strategy

### 8.1 Position initialization

The graph starts from:

- configured initial pose, or
- WLS position if WLS is available on the first cycle

This improves convergence because the initial TDOA geometry can be ambiguous if the seed is poor.

### 8.2 Orientation initialization from gravity

Before graph initialization, the node can estimate initial roll and pitch from the mean accelerometer vector during an initial static window.

The method:

1. accumulate a fixed number of IMU acceleration samples
2. compute mean acceleration
3. estimate:
   - roll from `atan2(a_y, a_z)`
   - pitch from `atan2(-a_x, sqrt(a_y^2 + a_z^2))`
4. preserve yaw from the configured initial pose

This is a common leveling approach and is effective when the platform is approximately static at startup.

---

## 9. Use of Priors

### 9.1 WLS position prior

The tightly coupled graph can optionally add a soft pose prior using the WLS result.

Important design choice:

- XY prior is tighter
- Z prior is looser

This reflects typical anchor geometry: horizontal positioning is often better constrained than vertical positioning.

### 9.2 Vertical prior

A dedicated vertical prior can constrain only the `z` direction while leaving other components effectively unconstrained.

This is useful because:

- TDOA altitude may be weakly observable
- vertical drift can grow if anchor geometry is poor

### 9.3 Velocity and bias regularization

The node also adds priors on:

- velocity at each update
- bias at each update

These are not hard constraints. They act as stabilizers to reduce unbounded drift or implausible bias jumps.

---

## 10. Measurement Conditioning and Dropout Handling

This section describes a key engineering characteristic of the implementation: the graph is not fed raw asynchronous sensor streams directly. Instead, the node performs a layer of measurement conditioning before graph construction, and it changes the graph structure when IMU support becomes unreliable.

These methods are not cosmetic. They are central to why the estimator remains stable on real data.

### 10.1 IMU unit normalization

In `imuCallback()`, the raw IMU message is converted to the units expected by inertial navigation:

- acceleration is scaled into `m/s^2`
- angular velocity is converted into `rad/s` when the sensor publishes `deg/s`

This is essential because GTSAM preintegration assumes a physically consistent motion model. A unit mismatch, especially in the gyroscope, would make orientation propagation diverge quickly and would corrupt the entire graph update.

### 10.2 Gyroscope sanity checking

The node includes an early-stage sanity check on the first batch of gyro samples. It computes the average raw angular-rate norm and compares it against what is physically reasonable for the platform.

Purpose:

- detect the common `deg/s` versus `rad/s` configuration error
- surface the issue before the graph begins accumulating bad inertial information

This is a simple but highly practical estimator-protection mechanism.

### 10.3 IMU buffering and replay

The IMU is received at high rate, but the graph is updated at the lower UWB cycle rate. Instead of inserting every IMU sample directly into the factor graph, the node:

1. buffers IMU measurements continuously
2. waits until a UWB cycle closes
3. replays only the measurements in the interval `[last_cycle_t, t_mid]`
4. integrates them into one preintegrated IMU summary

This is the standard preintegration pattern for asynchronous multisensor fusion. It reduces graph size while preserving the motion information accumulated between keyframes.

### 10.4 UWB cycle batching

The UWB stream is not used one message at a time as a separate graph update. Instead, `uwbCallback()` groups TDOA measurements into a short cycle using:

- timeout logic
- a required number of measurements per cycle
- pair deduplication logic for anchor pairs

The result is a compact measurement batch that represents one graph update time. This improves temporal consistency and gives the WLS and factor graph a better-constrained snapshot of UWB geometry.

### 10.5 WLS front-end as a conditioning stage

Before the tightly coupled update is built, `processCycle()` first runs the UWB-only solver.

This WLS estimate serves three roles:

- direct UWB output for inspection
- seed for graph initialization
- optional soft prior for the tightly coupled state

This is an important front-end/back-end split:

- the front-end extracts a plausible geometric position from the raw UWB batch
- the back-end decides how strongly that position should influence the fused estimate

### 10.6 Gravity-based initial attitude estimation

The node uses an initial accelerometer averaging step to estimate roll and pitch before the first tightly coupled graph initialization.

Technique:

- average the first `N` acceleration samples
- assume the dominant acceleration is gravity
- infer roll and pitch from the gravity direction
- preserve yaw from the configured initial pose

This is a lightweight alignment method that reduces startup inconsistency between the inertial state and the physical sensor frame.

### 10.7 Innovation gating on the WLS prior

The code does not blindly trust the WLS solution. Before using it as a prior in the graph, it checks whether the WLS estimate is close enough to the current tightly coupled estimate in the XY plane.

If the gap is too large, the WLS prior is rejected for that cycle.

This protects the optimizer against:

- stale WLS estimates
- NLOS-driven UWB excursions
- large corrections applied at a poor linearization point

The XY-only gating is deliberate: horizontal geometry is treated as more trustworthy than vertical geometry.

### 10.8 What "IMU dropout" means here

In this code, IMU dropout does not necessarily mean the IMU sensor is fully dead. It means that for the current UWB cycle, the number of usable IMU samples is below the threshold required for a reliable preintegrated update:

`n_imu < min_imu_per_cycle_`

At that point, the code assumes the normal inertial propagation step is not trustworthy enough to construct a standard `ImuFactor` update.

### 10.9 Why a special dropout update is needed

If the implementation simply skipped the cycle or built a weak IMU factor from too little data, several failure modes could appear:

- stale velocity would continue propagating position incorrectly
- graph linearization would be based on poor inertial prediction
- pose could drift significantly during short sensor outages

The code comments explicitly note this risk as phantom drift during missing-IMU intervals.

### 10.10 Dropout update strategy

When IMU support is insufficient, the node switches to `tcDropoutUpdate(...)` instead of `tcUpdateGraph(...)`.

In dropout mode, the graph update is restructured as follows:

1. **No `ImuFactor` is added**
   - because there is not enough inertial information to justify one
2. **Raw TDOA factors are still added**
   - UWB geometry remains available and still constrains position
3. **A WLS-based position prior may be added**
   - but only if it passes the innovation gate
4. **Bias is carried forward with a random-walk / prior combination**
   - bias should not jump just because IMU samples are temporarily absent
5. **Velocity is regularized with a tight prior**
   - this suppresses unrealistic free coasting during the outage
6. **A vertical prior may still be applied**
   - to keep altitude behavior stable

This is a graceful degradation strategy rather than a simple fail-stop.

### 10.11 Interpretation of the dropout velocity prior

One subtle point is that the code does not necessarily force velocity to zero during dropout. Instead, it regularizes velocity around the current estimate with a tighter covariance.

That choice has two benefits:

- it prevents unbounded growth of velocity-driven drift
- it avoids unrealistically slamming a moving platform to zero velocity

So the behavior is better understood as **velocity damping** rather than hard freezing.

### 10.12 Estimation principle behind these methods

Taken together, these techniques follow a common robust-estimation principle:

- when a sensor is reliable, use its physically rich model
- when a sensor becomes weak or inconsistent, reduce its authority
- keep the rest of the graph active using whatever measurements remain trustworthy

In this node, that means:

- good IMU availability -> use preintegration and a full inertial factor
- poor IMU availability -> fall back to UWB geometry, priors, and state regularization

This adaptive graph construction is one of the most important technical ideas in the codebase.

---

## 11. Robustness Mechanisms

This implementation contains several explicit robustness techniques.

### 11.1 Robust TDOA noise

The TDOA factor can be wrapped with a **Huber robust kernel**.

This reduces the impact of:

- NLOS measurements
- multipath outliers
- occasional corrupted TDOA samples

### 11.2 WLS innovation gate

Before using WLS as a prior in the tightly coupled graph, the node checks whether the WLS estimate is sufficiently close to the current tightly coupled estimate in the XY plane.

If the difference is too large, the WLS prior is rejected for that cycle.

Why this helps:

- a bad or stale WLS prior can destabilize the graph
- forcing a large correction at a poor linearization point can hurt iSAM2

The implementation intentionally gates in XY only because Z is considered less reliable.

### 11.3 IMU dropout handling

If too few IMU samples are available in a cycle, the node uses a dedicated dropout update path instead of a normal `ImuFactor`.

During dropout:

- no IMU factor is added
- WLS prior may still be used
- TDOA factors are still added
- bias is carried forward
- velocity is damped with a prior

This prevents the estimator from drifting freely during sensor outages.

### 11.4 Unit sanity checking

The code checks early gyroscope samples to detect the common mistake of treating `deg/s` as `rad/s`.

This is a very practical engineering safeguard. A unit mismatch in gyro data would quickly destroy IMU integration quality.

---

## 12. Logging, Evaluation, and Outputs

The node logs:

- cycle-level trajectory CSV
- raw IMU CSV

Cycle logs include:

- ground truth position when available
- WLS estimate
- tightly coupled estimate
- orientation quaternion
- velocity
- accelerometer bias
- gyroscope bias
- summarized IMU statistics over the cycle
- WLS and tightly coupled position errors relative to ground truth

Published ROS outputs include:

- `uwb_wls`
- `tc_fusion`
- corresponding `Path` topics
- TF for the tightly coupled solution

This makes the package suitable for both online use and offline analysis.

---

## 13. Why This Is a Tightly Coupled Design

The core reason this implementation is tightly coupled is:

- it does **not** use only the WLS position as a measurement for the graph
- it inserts **each raw TDOA measurement directly** into the graph

That means the optimizer jointly reasons over:

- motion dynamics from IMU
- geometry from each anchor pair
- bias evolution
- optional priors

This is more expressive than a loosely coupled design where UWB would first be collapsed into a pose estimate and only that pose would be fused.

---

## 14. Advantages and Limitations

### Advantages

- raw UWB measurements are used directly
- IMU gives motion continuity between sparse UWB cycles
- WLS gives a fast UWB-only estimate and a useful seed
- robust kernel and gating improve resilience to bad measurements
- dropout logic improves behavior during partial sensor failure

### Limitations

- accuracy depends strongly on anchor geometry
- vertical observability may remain weak
- gravity-based initialization assumes a near-static startup
- bias/noise tuning remains important for stable performance
- WLS normal equations are unweighted in the current implementation despite the class name

That last point is worth noting: the solver is called `TdoaWlsSolver`, but the current code effectively performs Gauss-Newton least squares without per-measurement weighting. The robust weighting is applied in the graph back-end, not in the standalone WLS front-end.

---

## 15. Summary

This codebase implements a practical UWB-IMU fusion pipeline with two complementary estimators:

- a Gauss-Newton TDOA position solver
- a tightly coupled IMU + TDOA factor graph

Technically, the key methods are:

- nonlinear least squares for TDOA localization
- custom factor-graph modeling of TDOA
- IMU preintegration
- incremental smoothing with iSAM2
- soft priors and innovation gating for stability
- dropout-specific fallback logic

Overall, the package is an engineering-oriented fusion system designed not just around nominal estimation theory, but also around real data issues such as:

- unit mismatches
- weak Z observability
- multipath outliers
- IMU dropout
- bias convergence

That combination of model-based estimation and field-oriented safeguards is the main technical character of this implementation.
