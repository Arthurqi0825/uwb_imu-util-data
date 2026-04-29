// =============================================================================
// uwb_imu_fusion_node.cpp  —  Tightly Coupled UWB + IMU fusion
//
// PIPELINE
// ─────────────────────────────────────────────────────────────────────────────
//  UWB stream → cycle accumulator (validated, unchanged)
//                    │
//                    ├─► WLS solver  ──────────────────────────────────────────►  uwb_wls / uwb_wls_path
//                    │
//                    └─► TC-FGO  (TdoaFactor × N + ImuFactor + BiasBetween)  ──► tc_fusion / tc_fusion_path
//
//  IMU stream → imu_buf_
//
// TIGHTLY COUPLED (TC)
//   Adds one TdoaFactor per raw TDOA measurement every cycle.
//   Directly fuses range-difference geometry with IMU integration.
//   Richer UWB information but sensitive to initialization and outliers.
//
// SIGN CONVENTION (DO NOT CHANGE — validated on real data)
//   WLS residual:  r(i) = (||p-aB|| - ||p-aA||) - tdoa_measured
//   TdoaFactor:    predicted = distA - distB  →  pass -m.tdoa as measured_
//
// IMU UNIT NOTE
//   The physical IMU publishes specific force in **g** (not m/s²).
//   accel_scale_ = gravity_magnitude  converts g → m/s² before any use:
//     • GTSAM preintegration  (integrateMeasurement)
//     • Initial gravity-alignment atan2 (works on raw g values too, but
//       scaling first makes the norm-sanity-check consistent with m/s²)
//   LOAD ORDER IS CRITICAL: gravity_mag_, imu_accel_in_g_ and gyro_is_degrees_
//   must be fetched BEFORE accel_scale_ default is computed.
//
// WEIGHT PHILOSOPHY
//   UWB/TDOA → moderately trusted + Huber robust kernel to handle NLOS spikes
//              tdoa_sigma = 0.3 m  (balanced: not so tight that multipath
//              dominates; not so loose that the graph ignores TDOA entirely)
//   WLS prior → trusted for XY (0.25 m), moderate for Z (0.35 m)
//              WLS Z is the weakest axis — looser Z sigma prevents vertical lock
//   IMU       → loosened noise model (g-unit MEMS sensor; real noise floor)
//   Bias R/W  → ADAPTIVE — MUST be loose enough to let real bias converge
//               from zero-init.  Frozen bias → quadratic position drift.
//               tc_prior_bias_sigma_ ≥ 0.1 m/s² releases the zero-bias pin.
//   Velocity  → tighter tc_update_vel_sigma to prevent runaway drift
//
// BIAS CONVERGENCE NOTE (critical)
//   tc_.bias initialises to zero.  tc_prior_bias_sigma_ is the prior on B(0);
//   if it is too small (e.g. 0.001) the graph cannot move bias away from zero
//   regardless of evidence.  Real MEMS sensors commonly have offsets of
//   0.05–0.3 m/s² (0.005–0.03 g).  Setting tc_prior_bias_sigma_ = 0.1 m/s²
//   lets the first cycle's TDOA residuals drive bias to a realistic value.
// =============================================================================

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <cf_msgs/Tdoa.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <set>
#include <utility>
#include <vector>
#include <cmath>
#include <string>
#include <numeric>

#include "tdoa_factor.h"
#include "wlssolver.h"

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

namespace uwb_imu_fusion
{
  struct ImuMeas
  {
    double t;
    gtsam::Vector3 acc; // [m/s²] — already converted from g in imuCallback
    gtsam::Vector3 gyr; // [rad/s] — already converted from deg/s if needed
  };

  struct FgoState
  {
    gtsam::Pose3 pose;
    gtsam::Vector3 vel{gtsam::Vector3::Zero()};
    gtsam::imuBias::ConstantBias bias;
    size_t cycle_idx{0};
    bool initialized{false};
    double last_cycle_t{0.0};
  };

  class UwbImuFusionNode
  {
  public:
    UwbImuFusionNode(ros::NodeHandle &nh, ros::NodeHandle &pnh)
        : nh_(nh), pnh_(pnh)
    {
      loadParams();
      setupGtsam();
      openLog();

      // Publishers
      wls_odom_pub_ = nh_.advertise<nav_msgs::Odometry>("uwb_wls", 50);
      tc_odom_pub_ = nh_.advertise<nav_msgs::Odometry>("tc_fusion", 50);
      wls_path_pub_ = nh_.advertise<nav_msgs::Path>("uwb_wls_path", 10);
      tc_path_pub_ = nh_.advertise<nav_msgs::Path>("tc_fusion_path", 10);
      gt_path_pub_ = nh_.advertise<nav_msgs::Path>("uwb_imu_path_gt", 10);

      // Subscribers
      std::string imu_topic, uwb_topic, gt_topic;
      pnh_.param<std::string>("imu_topic", imu_topic, "/imu_data");
      pnh_.param<std::string>("uwb_topic", uwb_topic, "/tdoa_data");
      pnh_.param<std::string>("gt_topic", gt_topic, "/pose_data");
      imu_sub_ = nh_.subscribe(imu_topic, 1000, &UwbImuFusionNode::imuCallback, this);
      uwb_sub_ = nh_.subscribe(uwb_topic, 500, &UwbImuFusionNode::uwbCallback, this);
      gt_sub_ = nh_.subscribe(gt_topic, 50, &UwbImuFusionNode::gtCallback, this);

      wls_guess_ = Eigen::Vector3d(initial_pose_.translation().x(),
                                   initial_pose_.translation().y(),
                                   initial_pose_.translation().z());

      logConfiguration();
    }

    ~UwbImuFusionNode()
    {
      if (traj_log_.is_open())
      {
        traj_log_.flush();
        traj_log_.close();
      }
      if (imu_log_.is_open())
      {
        imu_log_.flush();
        imu_log_.close();
      }
    }

  private:
    void loadParams()
    {
      auto get = [&](const std::string &k, auto &v, auto def)
      {
        if (!nh_.getParam("/uwb_imu_fusion/" + k, v) && !pnh_.getParam(k, v))
          v = def;
      };

      // Anchors
      XmlRpc::XmlRpcValue axv;
      bool ok = nh_.getParam("/uwb_imu_fusion/anchors", axv) ||
                pnh_.getParam("anchors", axv);
      if (ok && axv.getType() == XmlRpc::XmlRpcValue::TypeArray)
      {
        for (int i = 0; i < axv.size(); ++i)
        {
          auto &row = axv[i];
          if (row.getType() == XmlRpc::XmlRpcValue::TypeArray && row.size() == 3)
            anchors_.emplace_back((double)row[0], (double)row[1], (double)row[2]);
        }
      }
      if (anchors_.empty())
      {
        ROS_ERROR("[uwb_imu_fusion] No anchors!");
        ros::shutdown();
        return;
      }

      std::vector<double> ext_t{0, 0, 0}, ext_q{0, 0, 0, 1};
      if (!nh_.getParam("/uwb_imu_fusion/anchor_extrinsic_translation", ext_t))
        pnh_.getParam("anchor_extrinsic_translation", ext_t);
      if (!nh_.getParam("/uwb_imu_fusion/anchor_extrinsic_rotation", ext_q))
        pnh_.getParam("anchor_extrinsic_rotation", ext_q);
      Eigen::Quaterniond qe(ext_q[3], ext_q[0], ext_q[1], ext_q[2]);
      qe.normalize();
      Eigen::Vector3d te(ext_t[0], ext_t[1], ext_t[2]);
      for (auto &a : anchors_)
      {
        Eigen::Vector3d pw = qe.toRotationMatrix() *
                                 Eigen::Vector3d(a.x(), a.y(), a.z()) +
                             te;
        a = gtsam::Point3(pw.x(), pw.y(), pw.z());
      }

      // ---------------------------------------------------------------------------
      // Shared IMU hardware settings
      //
      // LOAD ORDER IS CRITICAL:
      //   1. Load gyro_is_degrees_, gravity_mag_, imu_accel_in_g_ first.
      //   2. Only then compute the default for accel_scale_.
      //
      // gyro_is_degrees_ DEFAULT = true
      //   Data confirmed: raw gyro values reach 130 deg/s during fast motion.
      //   Interpreted as rad/s that would be 130 rad/s = 7.5 rev/s — physically
      //   impossible.  As deg/s the same peak = 2.3 rad/s = a fast tilt, which
      //   is consistent with the UWB trajectory.  Always set explicitly in the
      //   launch file; the default here guards against the silent 57× error.
      // ---------------------------------------------------------------------------
      get("gyro_is_degrees", gyro_is_degrees_, true); // FIX: was false — sensor publishes deg/s
      get("gravity_magnitude", gravity_mag_, 9.80665);
      get("imu_accel_in_g", imu_accel_in_g_, true);

      // Compute default accel_scale_ AFTER imu_accel_in_g_ and gravity_mag_ are known.
      // FIX: moved out of lambda so the ternary reads the already-loaded members.
      {
        double default_accel_scale = imu_accel_in_g_ ? gravity_mag_ : 1.0;
        get("accel_scale", accel_scale_, default_accel_scale);
      }

      get("estimate_initial_orientation_from_imu",
          estimate_initial_orientation_from_imu_, true);
      get("initial_alignment_imu_samples", initial_alignment_imu_samples_, 100);

      // ---------------------------------------------------------------------------
      // TC IMU noise — values derived from static-period analysis of real data.
      //
      //   Measured accel noise std (static):  ~0.04 m/s²  per axis
      //   Measured gyro  noise std (static):  ~0.0025 rad/s per axis (after deg→rad)
      //   Measured accel bias (static mean):  ax=+0.22, ay=+0.07 m/s² — real offset
      //   Measured gyro  bias (static mean):  ~0.001 rad/s — near-zero, small
      //
      //   tc_accel_noise_sigma  0.04 m/s²   matches measured floor
      //   tc_gyro_noise_sigma   0.003 rad/s  slightly above measured 0.0025
      //                                      to give the graph some flexibility
      //   tc_accel_bias_rw_sigma 0.01        loose: real 0.22 m/s² accel offset
      //                                      must converge from zero
      //   tc_gyro_bias_rw_sigma  0.0003      gyro bias is small and stable
      // ---------------------------------------------------------------------------
      get("tc_accel_noise_sigma", tc_accel_sigma_, 0.04);             // measured ~0.04 m/s²
      get("tc_gyro_noise_sigma", tc_gyro_sigma_, 0.003);              // measured ~0.0025 rad/s
      get("tc_accel_bias_rw_sigma", tc_accel_bias_rw_sigma_, 0.01);   // real accel bias ~0.22 m/s²
      get("tc_gyro_bias_rw_sigma", tc_gyro_bias_rw_sigma_, 0.0003);   // gyro bias stable
      get("tc_integration_noise_sigma", tc_integration_sigma_, 1e-4); // raised: ~974 Hz IMU, many integration steps

      // ---------------------------------------------------------------------------
      // UWB / TC measurement noise
      //
      // tdoa_sigma = 0.3 m is the balanced operating point:
      //   • Too tight (0.15 m): any multipath/NLOS spike pulls the trajectory
      //     hard, causing the large error jumps seen in the logs.
      //   • Too loose (0.5 m): TDOA contributes almost nothing; TC degrades to
      //     IMU-only dead reckoning.
      //   • 0.3 m + Huber kernel: nominal measurements carry full weight;
      //     outliers beyond huber_k are automatically down-weighted.
      //
      // tc_wls_xy_sigma / z_sigma: WLS is proven accurate in XY.
      //   XY = 0.25 m (confident), Z = 0.35 m (WLS Z noisier due to poor
      //   vertical anchor geometry → slightly looser to avoid vertical lock).
      // ---------------------------------------------------------------------------
      get("tdoa_sigma", tdoa_sigma_, 0.3); // balanced: was 0.15 (too tight → NLOS jumps)
      get("use_robust_noise", use_robust_noise_, true);
      get("huber_k", huber_k_, 1.345);
      get("cycle_timeout", cycle_timeout_, 0.1);
      int mpc = 8, min_imu = 2;
      get("measurements_per_cycle", mpc, 8);
      get("min_imu_per_cycle", min_imu, 2);
      measurements_per_cycle_ = mpc;
      min_imu_per_cycle_ = min_imu;

      // ---------------------------------------------------------------------------
      // TC priors
      //
      // tc_prior_pose_sigmas  [rot_x, rot_y, rot_z, tx, ty, tz]
      // tc_prior_vel_sigma    startup velocity uncertainty [m/s]
      //
      // tc_prior_bias_sigma   FIX: was 0.001 — this pinned bias to zero.
      //   Real MEMS accel biases are 0.05–0.3 m/s².  Setting 0.001 makes it
      //   mathematically impossible for the graph to move bias far from zero,
      //   causing un-compensated offset to integrate quadratically into position.
      //   Set to 0.1 m/s² so the first cycle's TDOA residuals can drive bias
      //   to a realistic value.
      //
      // tc_update_vel_sigma   FIX: was 2.0 — too loose, allows velocity to
      //   drift far from the IMU prediction between cycles.  Tighten to 0.5 m/s
      //   so velocity stays anchored to the integrated IMU trajectory.
      //
      // tc_update_bias_sigma  Moderate: bias can evolve slowly between updates.
      // ---------------------------------------------------------------------------
      std::vector<double> tc_pps{0.5, 0.5, 0.5, 0.05, 0.05, 0.05};
      if (!nh_.getParam("/uwb_imu_fusion/tc_prior_pose_sigmas", tc_pps))
        pnh_.getParam("tc_prior_pose_sigmas", tc_pps);
      tc_prior_pose_sigmas_ = gtsam::Vector6::Map(tc_pps.data());
      get("tc_prior_vel_sigma", tc_prior_vel_sigma_, 0.1);
      get("tc_prior_bias_sigma", tc_prior_bias_sigma_, 0.1); // FIX: was 0.001 — too tight, froze bias at zero
      get("tc_update_vel_sigma", tc_update_vel_sigma_, 0.5); // FIX: was 2.0 — too loose, velocity drifts
      get("tc_update_bias_sigma", tc_update_bias_sigma_, 0.1);
      get("tc_use_vertical_prior", tc_use_vertical_prior_, true);
      get("tc_vertical_prior_sigma", tc_vertical_prior_sigma_, 0.2);

      // WLS position prior sigmas
      //   XY: 0.25 m — WLS proven accurate in horizontal plane
      //   Z:  0.35 m — WLS Z is noisier (poor vertical anchor geometry);
      //               looser Z avoids vertical locking to a biased WLS height
      get("tc_use_wls_position_prior", tc_use_wls_position_prior_, true);
      get("tc_wls_xy_sigma", tc_wls_xy_sigma_, 0.25); // balanced: was 0.15 (too tight → NLOS jumps)
      get("tc_wls_z_sigma", tc_wls_z_sigma_, 0.35);   // looser Z: WLS Z less reliable
      get("tc_wls_rescue_gate", tc_wls_rescue_gate_, 0.25);
      get("tc_wls_rescue_xy_sigma", tc_wls_rescue_xy_sigma_, 0.08);
      get("tc_wls_rescue_z_sigma", tc_wls_rescue_z_sigma_, 0.25);

      // ---------------------------------------------------------------------------
      // TC attitude stabilization
      //
      // TDOA is translation-only, so roll/pitch can become weakly observable in the
      // graph.  When roll/pitch tilt incorrectly, gravity leaks into horizontal
      // acceleration and creates large velocity errors.  The attitude prior keeps
      // roll/pitch near the leveled initial attitude while leaving yaw effectively
      // free, since yaw is not observable from UWB TDOA alone.
      // ---------------------------------------------------------------------------
      get("tc_use_attitude_prior", tc_use_attitude_prior_, true);
      get("tc_attitude_roll_pitch_sigma", tc_attitude_roll_pitch_sigma_, 0.05);
      get("tc_attitude_yaw_sigma", tc_attitude_yaw_sigma_, 1e6);

      // ---------------------------------------------------------------------------
      // IMU dropout handling
      //
      // When imu_n < min_imu_per_cycle the current code returns early — no TDOA
      // factors, no WLS prior, velocity frozen.  During a 0.6 s dropout at 1 m/s
      // this produces ~0.6 m phantom drift.  Data showed 100% of dropout cycles
      // had valid WLS (err<0.35 m).
      //
      // tc_wls_dropout_xy_sigma  looser WLS prior used when IMU is absent
      //   (no ImuFactor to constrain orientation, so position prior must be softer)
      // tc_dropout_vel_sigma     velocity prior applied during dropout to stop
      //   the stale velocity from integrating phantom position
      // wls_innovation_gate      max distance [m] between WLS and predicted TC
      //   position before the WLS prior is rejected as an outlier.  Prevents a
      //   drifted TC state from being anchored to a WLS point that is far from
      //   the linearisation point, which destabilises the graph.
      // ---------------------------------------------------------------------------
      get("tc_wls_dropout_xy_sigma", tc_wls_dropout_xy_sigma_, 0.40); // softer: no IMU orientation
      get("tc_dropout_vel_sigma", tc_dropout_vel_sigma_, 0.1);        // tight: stop phantom drift
      get("wls_innovation_gate", wls_innovation_gate_, 0.8);          // reject WLS >0.8m from pred

      // Initial pose
      std::vector<double> ip{0, 0, 1};
      if (!nh_.getParam("/uwb_imu_fusion/initial_position", ip))
        pnh_.getParam("initial_position", ip);
      std::vector<double> iq{0, 0, 0, 1};
      if (!nh_.getParam("/uwb_imu_fusion/initial_orientation", iq))
        pnh_.getParam("initial_orientation", iq);
      Eigen::Quaterniond q0(iq[3], iq[0], iq[1], iq[2]);
      q0.normalize();
      initial_pose_ = gtsam::Pose3(gtsam::Rot3(q0.toRotationMatrix()),
                                   gtsam::Point3(ip[0], ip[1], ip[2]));

      get("odom_frame", odom_frame_, std::string("map"));
      get("base_frame", base_frame_, std::string("base_link"));
      get("trajectory_log_path", traj_log_path_,
          std::string("/tmp/uwb_imu_trajectory.csv"));
      get("imu_log_path", imu_log_path_,
          std::string("/tmp/uwb_imu_raw_imu.csv"));
    }

    // ===========================================================================
    // GTSAM SETUP
    // ===========================================================================
    void setupGtsam()
    {
      // TC preintegration params
      // MakeSharedU sets the gravity vector along +Z with magnitude gravity_mag_.
      // Covariances are in (m/s²)² and (rad/s)² — consistent with accel_scale_ applied
      // in imuCallback so all buffered samples are already in m/s².
      auto tc_p = gtsam::PreintegrationParams::MakeSharedU(gravity_mag_);
      tc_p->setAccelerometerCovariance(
          gtsam::Matrix33::Identity() * std::pow(tc_accel_sigma_, 2));
      tc_p->setGyroscopeCovariance(
          gtsam::Matrix33::Identity() * std::pow(tc_gyro_sigma_, 2));
      tc_p->setIntegrationCovariance(
          gtsam::Matrix33::Identity() * std::pow(tc_integration_sigma_, 2));
      tc_imu_params_ = tc_p;

      tc_.bias = gtsam::imuBias::ConstantBias();
      tc_preint_ = boost::make_shared<gtsam::PreintegratedImuMeasurements>(
          tc_imu_params_, tc_.bias);

      // TC: TDOA noise
      // Declare as SharedNoiseModel so both if/else branches share the same type.
      gtsam::SharedNoiseModel base_tdoa =
          gtsam::noiseModel::Isotropic::Sigma(1, tdoa_sigma_);
      if (use_robust_noise_)
        tdoa_noise_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(huber_k_), base_tdoa);
      else
        tdoa_noise_ = base_tdoa;

      gtsam::ISAM2Params isam_p;
      isam_p.relinearizeThreshold = 0.1;
      isam_p.relinearizeSkip = 1;
      isam_p.factorization = gtsam::ISAM2Params::CHOLESKY;
      tc_isam_ = boost::make_shared<gtsam::ISAM2>(isam_p);
    }

    // ===========================================================================
    // LOG
    // ===========================================================================
    void openLog()
    {
      // --- trajectory log (one row per UWB cycle) ----------------------------
      traj_log_.open(traj_log_path_, std::ios::out | std::ios::trunc);
      if (traj_log_.is_open())
      {
        traj_log_ << "timestamp,"
                  << "gt_x,gt_y,gt_z,"
                  << "wls_x,wls_y,wls_z,"
                  << "tc_x,tc_y,tc_z,"
                  << "tc_qx,tc_qy,tc_qz,tc_qw,"
                  << "tc_vx,tc_vy,tc_vz,"
                  << "tc_bias_ax,tc_bias_ay,tc_bias_az,"
                  << "tc_bias_gx,tc_bias_gy,tc_bias_gz,"
                  << "imu_n,imu_mean_ax,imu_mean_ay,imu_mean_az,"
                  << "imu_mean_gx,imu_mean_gy,imu_mean_gz,"
                  << "imu_accel_norm_mean,imu_gyro_norm_mean,"
                  << "wls_gt_err,tc_gt_err\n";
      }
      else
      {
        ROS_WARN("[uwb_imu_fusion] Cannot open trajectory log: %s", traj_log_path_.c_str());
      }

      // --- raw IMU log (one row per IMU sample) --------------------------------
      // Columns:
      //   timestamp         — ROS time of the IMU sample [s]
      //   raw_ax/ay/az      — accelerometer as received from the sensor (original units: g)
      //   raw_gx/gy/gz      — gyroscope as received (original units: depends on gyro_is_degrees_)
      //   acc_x/y/z         — accelerometer after unit conversion [m/s²]
      //   gyr_x/y/z         — gyroscope after unit conversion [rad/s]
      //   accel_norm        — ||acc|| [m/s²]  (should be ~9.807 when static)
      //   gyro_norm         — ||gyr|| [rad/s] (should be ~0 when static)
      imu_log_.open(imu_log_path_, std::ios::out | std::ios::trunc);
      if (imu_log_.is_open())
      {
        imu_log_ << "timestamp,"
                 << "raw_ax,raw_ay,raw_az,"   // original sensor units (g if imu_accel_in_g_)
                 << "raw_gx,raw_gy,raw_gz,"   // original sensor units (deg/s or rad/s)
                 << "acc_x,acc_y,acc_z,"      // [m/s²] after conversion
                 << "gyr_x,gyr_y,gyr_z,"      // [rad/s] after conversion
                 << "accel_norm,gyro_norm\n"; // magnitudes for quick sanity check
      }
      else
      {
        ROS_WARN("[uwb_imu_fusion] Cannot open IMU log: %s", imu_log_path_.c_str());
      }
    }

    // ---------------------------------------------------------------------------
    // logRow — one row per UWB cycle in the trajectory CSV.
    //
    // imu_stats: pre-computed summary of the IMU samples that fell within this
    //   cycle window [last_cycle_t, t_mid].  Computed in processCycle() before
    //   calling logRow so we don't re-scan imu_buf_ twice.
    // ---------------------------------------------------------------------------
    struct CycleImuStats
    {
      int n{0};
      double mean_ax{0}, mean_ay{0}, mean_az{0};
      double mean_gx{0}, mean_gy{0}, mean_gz{0};
      double mean_accel_norm{0}, mean_gyro_norm{0};
    };

    CycleImuStats computeCycleImuStats(double from_t, double to_t) const
    {
      CycleImuStats s;
      for (const auto &imu : imu_buf_)
      {
        if (imu.t <= from_t)
          continue;
        if (imu.t > to_t)
          break;
        s.mean_ax += imu.acc.x();
        s.mean_ay += imu.acc.y();
        s.mean_az += imu.acc.z();
        s.mean_gx += imu.gyr.x();
        s.mean_gy += imu.gyr.y();
        s.mean_gz += imu.gyr.z();
        s.mean_accel_norm += imu.acc.norm();
        s.mean_gyro_norm += imu.gyr.norm();
        ++s.n;
      }
      if (s.n > 0)
      {
        const double inv = 1.0 / s.n;
        s.mean_ax *= inv;
        s.mean_ay *= inv;
        s.mean_az *= inv;
        s.mean_gx *= inv;
        s.mean_gy *= inv;
        s.mean_gz *= inv;
        s.mean_accel_norm *= inv;
        s.mean_gyro_norm *= inv;
      }
      return s;
    }

    void logRow(double t, bool wls_ok, const Eigen::Vector3d &wls_p,
                const CycleImuStats &imu_stats)
    {
      if (!traj_log_.is_open())
        return;
      const auto nan = std::numeric_limits<double>::quiet_NaN();
      auto v3 = [](const gtsam::Point3 &p)
      {
        return Eigen::Vector3d(p.x(), p.y(), p.z());
      };
      Eigen::Vector3d tp = tc_.initialized ? v3(tc_.pose.translation())
                                           : Eigen::Vector3d(nan, nan, nan);
      Eigen::Quaterniond tq = tc_.initialized
                                  ? Eigen::Quaterniond(tc_.pose.rotation().matrix())
                                  : Eigen::Quaterniond(nan, nan, nan, nan);
      Eigen::Vector3d gt_e(gt_pos_.x(), gt_pos_.y(), gt_pos_.z());

      traj_log_ << std::fixed << std::setprecision(6)
                << t << ","
                // ground truth
                << (gt_have_ ? gt_pos_.x() : nan) << ","
                << (gt_have_ ? gt_pos_.y() : nan) << ","
                << (gt_have_ ? gt_pos_.z() : nan) << ","
                // WLS position
                << (wls_ok ? wls_p.x() : nan) << ","
                << (wls_ok ? wls_p.y() : nan) << ","
                << (wls_ok ? wls_p.z() : nan) << ","
                // TC position + orientation
                << tp.x() << "," << tp.y() << "," << tp.z() << ","
                << tq.x() << "," << tq.y() << "," << tq.z() << "," << tq.w() << ","
                // TC velocity
                << tc_.vel.x() << "," << tc_.vel.y() << "," << tc_.vel.z() << ","
                // TC accel bias [m/s²]
                << tc_.bias.accelerometer().x() << ","
                << tc_.bias.accelerometer().y() << ","
                << tc_.bias.accelerometer().z() << ","
                // TC gyro bias [rad/s]
                << tc_.bias.gyroscope().x() << ","
                << tc_.bias.gyroscope().y() << ","
                << tc_.bias.gyroscope().z() << ","
                // IMU cycle summary (converted units: m/s² and rad/s)
                << imu_stats.n << ","
                << imu_stats.mean_ax << "," << imu_stats.mean_ay << "," << imu_stats.mean_az << ","
                << imu_stats.mean_gx << "," << imu_stats.mean_gy << "," << imu_stats.mean_gz << ","
                << imu_stats.mean_accel_norm << ","
                << imu_stats.mean_gyro_norm << ","
                // errors
                << (wls_ok && gt_have_ ? (wls_p - gt_e).norm() : nan) << ","
                << (tc_.initialized && gt_have_ ? (tp - gt_e).norm() : nan) << "\n";
      traj_log_.flush();
    }

    // ===========================================================================
    // IMU CALLBACK — convert units, buffer
    //
    // accel_scale_ is applied AFTER all hardware params are loaded.
    // With imu_accel_in_g_=true, accel_scale_ = gravity_mag_ = 9.80665.
    //
    // GYRO UNIT GUARD: if gyro_is_degrees_=false but the sensor actually
    // publishes deg/s, the raw norm will be ~10–130 instead of ~0.001–2.3.
    // We check the first 200 samples and warn loudly if the median gyro norm
    // exceeds 5 rad/s (= 286 deg/s), which is physically impossible for a
    // typical UWB-tracked robot/drone at rest or slow motion.
    // ===========================================================================
    void imuCallback(const sensor_msgs::Imu::ConstPtr &msg)
    {
      std::lock_guard<std::mutex> lk(mutex_);
      const double t = msg->header.stamp.toSec();
      if (last_imu_t_ >= 0.0 && t <= last_imu_t_)
      {
        ROS_WARN_THROTTLE(1.0, "[uwb_imu_fusion] IMU out-of-order %.6f", t);
        return;
      }

      // Convert raw sensor values to SI units:
      //   accel: raw_g × gravity_mag_ → m/s²
      //   gyro:  raw_deg × π/180     → rad/s  (if gyro_is_degrees_ is set)
      double ax = msg->linear_acceleration.x * accel_scale_;
      double ay = msg->linear_acceleration.y * accel_scale_;
      double az = msg->linear_acceleration.z * accel_scale_;
      double gx = msg->angular_velocity.x;
      double gy = msg->angular_velocity.y;
      double gz = msg->angular_velocity.z;
      if (gyro_is_degrees_)
      {
        const double d2r = M_PI / 180.0;
        gx *= d2r;
        gy *= d2r;
        gz *= d2r;
      }

      // --- Gyro unit sanity check (first 200 samples only) -------------------
      // Accumulate raw gyro norms before conversion to catch the deg/s mistake.
      if (gyro_sanity_count_ < 200)
      {
        gyro_sanity_raw_norm_sum_ += std::sqrt(
            msg->angular_velocity.x * msg->angular_velocity.x +
            msg->angular_velocity.y * msg->angular_velocity.y +
            msg->angular_velocity.z * msg->angular_velocity.z);
        ++gyro_sanity_count_;
        if (gyro_sanity_count_ == 200)
        {
          const double mean_raw = gyro_sanity_raw_norm_sum_ / 200.0;
          // After conversion the norm should be well under 5 rad/s for a
          // near-static or slowly moving platform.
          const double converted_mean = gyro_is_degrees_ ? mean_raw * (M_PI / 180.0) : mean_raw;
          if (!gyro_is_degrees_ && mean_raw > 5.0)
          {
            ROS_ERROR(
                "[uwb_imu_fusion] *** GYRO UNIT MISMATCH DETECTED ***\n"
                "  gyro_is_degrees_=false but mean raw gyro norm=%.2f\n"
                "  Values >5 rad/s are physically impossible for this platform.\n"
                "  The sensor almost certainly publishes deg/s.\n"
                "  FIX: set gyro_is_degrees:=true in your launch file.\n"
                "  TC fusion WILL diverge without this fix.",
                mean_raw);
          }
          else
          {
            ROS_INFO("[uwb_imu_fusion] Gyro unit check OK: mean converted norm=%.4f rad/s "
                     "(gyro_is_degrees=%s, raw_mean=%.4f)",
                     converted_mean,
                     gyro_is_degrees_ ? "true" : "false",
                     mean_raw);
          }
        }
      }
      // -----------------------------------------------------------------------
      ImuMeas s;
      s.t = t;
      s.acc = gtsam::Vector3(ax, ay, az); // [m/s²]
      s.gyr = gtsam::Vector3(gx, gy, gz); // [rad/s]
      imu_buf_.push_back(s);
      last_imu_t_ = t;

      // Write one row to the raw IMU log.
      // raw_* columns preserve the original sensor values before conversion so
      // you can verify the unit scale independently of the fusion results.
      if (imu_log_.is_open())
      {
        // Recover original sensor values: divide back out the scale factors.
        // accel_scale_ = gravity_mag_ when imu_accel_in_g_=true, else 1.0.
        // gyro was multiplied by d2r only if gyro_is_degrees_=true.
        const double raw_ax = msg->linear_acceleration.x;
        const double raw_ay = msg->linear_acceleration.y;
        const double raw_az = msg->linear_acceleration.z;
        const double raw_gx = msg->angular_velocity.x;
        const double raw_gy = msg->angular_velocity.y;
        const double raw_gz = msg->angular_velocity.z;
        imu_log_ << std::fixed << std::setprecision(9)
                 << t << ","
                 << raw_ax << "," << raw_ay << "," << raw_az << ","
                 << raw_gx << "," << raw_gy << "," << raw_gz << ","
                 << ax << "," << ay << "," << az << ","
                 << gx << "," << gy << "," << gz << ","
                 << s.acc.norm() << ","
                 << s.gyr.norm() << "\n";
        // Flush lazily (every ~100 samples) to avoid per-sample syscall overhead.
        if (++imu_log_flush_counter_ >= 100)
        {
          imu_log_.flush();
          imu_log_flush_counter_ = 0;
        }
      }

      // Gravity alignment uses the already-converted m/s² values
      updateInitialOrientationFromImu(s.acc);

      const double fence = tc_.initialized ? tc_.last_cycle_t - 0.5 : t - 5.0;
      while (!imu_buf_.empty() && imu_buf_.front().t < fence)
        imu_buf_.pop_front();
    }

    // ===========================================================================
    // UWB CALLBACK — cycle accumulator (validated, unchanged)
    // ===========================================================================
    void uwbCallback(const cf_msgs::Tdoa::ConstPtr &msg)
    {
      std::lock_guard<std::mutex> lk(mutex_);
      if (msg->idA < 0 || msg->idB < 0 ||
          msg->idA >= (int)anchors_.size() ||
          msg->idB >= (int)anchors_.size())
      {
        ROS_WARN_THROTTLE(2.0, "[uwb_imu_fusion] Anchor OOB (%d,%d)",
                          msg->idA, msg->idB);
        return;
      }
      TdoaMeas ts{msg->header.stamp.toSec(), msg->idA, msg->idB, msg->data};
      if (!current_cycle_.empty() &&
          (ts.t - current_cycle_.front().t) > cycle_timeout_)
      {
        current_cycle_.clear();
        cycle_pairs_.clear();
      }
      auto pr = std::make_pair(std::min(ts.idA, ts.idB), std::max(ts.idA, ts.idB));
      if (cycle_pairs_.count(pr))
      {
        if ((int)current_cycle_.size() >= measurements_per_cycle_)
          processCycle(current_cycle_);
        current_cycle_.clear();
        cycle_pairs_.clear();
      }
      current_cycle_.push_back(ts);
      cycle_pairs_.insert(pr);
      if ((int)current_cycle_.size() >= measurements_per_cycle_)
      {
        processCycle(current_cycle_);
        current_cycle_.clear();
        cycle_pairs_.clear();
      }
    }

    // ===========================================================================
    // GT CALLBACK
    // ===========================================================================
    void gtCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &msg)
    {
      std::lock_guard<std::mutex> lk(mutex_);
      gt_pos_ = gtsam::Point3(msg->pose.pose.position.x,
                              msg->pose.pose.position.y,
                              msg->pose.pose.position.z);
      gt_have_ = true;
      geometry_msgs::PoseStamped ps;
      ps.header = msg->header;
      ps.header.frame_id = odom_frame_;
      ps.pose = msg->pose.pose;
      gt_path_.header.stamp = msg->header.stamp;
      gt_path_.header.frame_id = odom_frame_;
      gt_path_.poses.push_back(ps);
      gt_path_pub_.publish(gt_path_);
    }

    // ===========================================================================
    // PROCESS CYCLE
    // ===========================================================================
    void processCycle(const std::vector<TdoaMeas> &meas)
    {
      double t_mid = 0.0;
      for (const auto &m : meas)
        t_mid += m.t;
      t_mid /= (double)meas.size();
      const ros::Time stamp(t_mid);

      // Compute IMU summary for this cycle window before any state update so
      // the window [last_cycle_t, t_mid] is still well-defined.
      const double imu_window_start = tc_.initialized ? tc_.last_cycle_t : (t_mid - cycle_timeout_);
      const CycleImuStats imu_stats = computeCycleImuStats(imu_window_start, t_mid);

      // WLS
      Eigen::Vector3d wls_p = wls_guess_;
      bool wls_ok = wls_solver_.solve(meas, anchors_, wls_p);
      if (wls_ok)
        wls_guess_ = wls_p;
      ROS_WARN("[WLS Solution] t=%.3f  x=%.3f  y=%.3f  z=%.3f",
               t_mid, wls_p.x(), wls_p.y(), wls_p.z());

      // ── TC: raw TDOA + IMU ─────────────────────────────────────────────────
      if (!tc_.initialized)
      {
        tcInitGraph(t_mid, meas, wls_ok, wls_p);
      }
      else
      {
        tcUpdateGraph(t_mid, meas, wls_ok, wls_p);
      }

      // Console
      printCycleInfo(t_mid, wls_ok, wls_p);

      // Publish WLS
      if (wls_ok)
      {
        publishOdom(wls_odom_pub_, stamp,
                    wls_p.x(), wls_p.y(), wls_p.z(),
                    gtsam::Rot3::identity(), gtsam::Vector3::Zero());
        appendPath(wls_path_, wls_path_pub_, stamp,
                   wls_p.x(), wls_p.y(), wls_p.z(), gtsam::Rot3::identity());
      }

      // Publish TC
      if (tc_.initialized)
      {
        const auto &tp = tc_.pose.translation();
        publishOdom(tc_odom_pub_, stamp,
                    tp.x(), tp.y(), tp.z(), tc_.pose.rotation(), tc_.vel);
        appendPath(tc_path_, tc_path_pub_, stamp,
                   tp.x(), tp.y(), tp.z(), tc_.pose.rotation());
        broadcastTf(stamp, tp, tc_.pose.rotation(), base_frame_ + "_tc");
      }

      logRow(t_mid, wls_ok, wls_p, imu_stats);
    }

    // ===========================================================================
    // INITIAL ATTITUDE ALIGNMENT FROM IMU GRAVITY
    //
    // NOTE: acc here is already in m/s² (converted in imuCallback).
    // The atan2 ratio is dimensionless so it works correctly whether we pass
    // raw g values or m/s² values.  The norm sanity check (< 1e-6) is now
    // consistent: a flat-lying sensor will give ~9.807 m/s², far from zero.
    // ===========================================================================
    void updateInitialOrientationFromImu(const gtsam::Vector3 &acc)
    {
      if (!estimate_initial_orientation_from_imu_ ||
          initial_orientation_aligned_ ||
          tc_.initialized)
      {
        return;
      }

      alignment_acc_sum_ += acc;
      ++alignment_acc_count_;
      if (alignment_acc_count_ < initial_alignment_imu_samples_)
      {
        return;
      }

      const gtsam::Vector3 mean_acc = alignment_acc_sum_ / alignment_acc_count_;
      const double norm = mean_acc.norm();

      // Sanity: in m/s² a static sensor should read ≈ 9.807 m/s².
      // Threshold of 1.0 m/s² (≈ 0.1 g) guards against near-zero bad data.
      if (norm < 1.0)
      {
        ROS_WARN("[uwb_imu_fusion] IMU leveling skipped: mean accel norm=%.3f m/s2 too small (expected ~%.1f)",
                 norm, gravity_mag_);
        initial_orientation_aligned_ = true;
        return;
      }

      const double roll = std::atan2(mean_acc.y(), mean_acc.z());
      const double pitch = std::atan2(-mean_acc.x(),
                                      std::sqrt(mean_acc.y() * mean_acc.y() +
                                                mean_acc.z() * mean_acc.z()));
      const auto yaw = initial_pose_.rotation().rpy().z();
      initial_pose_ = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw),
                                   initial_pose_.translation());
      initial_orientation_aligned_ = true;

      ROS_INFO("[uwb_imu_fusion] IMU leveling complete: roll=%.2f deg pitch=%.2f deg yaw=%.2f deg  (mean_acc_norm=%.3f m/s2)",
               roll * 180.0 / M_PI,
               pitch * 180.0 / M_PI,
               yaw * 180.0 / M_PI,
               norm);
    }

    // ===========================================================================
    // SHARED: IMU replay into a preintegrator over [from_t, to_t]
    // ===========================================================================
    int replayImu(double from_t, double to_t,
                  gtsam::PreintegratedImuMeasurements &preint)
    {
      int n = 0;
      double prev_t = from_t;
      for (const auto &imu : imu_buf_)
      {
        if (imu.t <= from_t)
          continue;
        if (imu.t > to_t)
          break;
        const double dt = imu.t - prev_t;
        if (dt <= 0.0 || dt > 1.0)
        {
          prev_t = imu.t;
          continue;
        }
        preint.integrateMeasurement(imu.acc, imu.gyr, dt);
        prev_t = imu.t;
        ++n;
      }
      return n;
    }

    // ===========================================================================
    // TC: bias random-walk BetweenFactor — uses TC-specific rw sigmas
    // ===========================================================================
    gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>
    tcBiasBetween(size_t k_pre, size_t k, double dt)
    {
      const double tau = std::max(dt, 1e-3);
      gtsam::Vector6 bs;
      bs << gtsam::Vector3::Constant(tc_accel_bias_rw_sigma_ * std::sqrt(tau)),
          gtsam::Vector3::Constant(tc_gyro_bias_rw_sigma_ * std::sqrt(tau));
      return gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
          B(k_pre), B(k),
          gtsam::imuBias::ConstantBias(),
          gtsam::noiseModel::Diagonal::Sigmas(bs));
    }

    void addTcVerticalPrior(gtsam::NonlinearFactorGraph &graph,
                            size_t k,
                            const gtsam::Point3 &pos_hint,
                            double z_ref)
    {
      if (!tc_use_vertical_prior_)
        return;
      gtsam::Vector6 sig;
      sig << 1e6, 1e6, 1e6,
          1e6, 1e6, tc_vertical_prior_sigma_;
      graph.addPrior(X(k),
                     gtsam::Pose3(gtsam::Rot3::identity(),
                                  gtsam::Point3(pos_hint.x(), pos_hint.y(), z_ref)),
                     gtsam::noiseModel::Diagonal::Sigmas(sig));
    }

    void addTcAttitudePrior(gtsam::NonlinearFactorGraph &graph,
                            size_t k,
                            const gtsam::Point3 &pos_hint)
    {
      if (!tc_use_attitude_prior_)
        return;

      // Rotation is constrained to the leveled startup attitude.  Translation is
      // unconstrained here; position is handled by TDOA, WLS, and vertical priors.
      gtsam::Vector6 sig;
      sig << tc_attitude_roll_pitch_sigma_,
          tc_attitude_roll_pitch_sigma_,
          tc_attitude_yaw_sigma_,
          1e6, 1e6, 1e6;
      graph.addPrior(X(k),
                     gtsam::Pose3(initial_pose_.rotation(), pos_hint),
                     gtsam::noiseModel::Diagonal::Sigmas(sig));
    }

    void addTcWlsPositionPrior(gtsam::NonlinearFactorGraph &graph,
                               size_t k,
                               const Eigen::Vector3d &wls_p,
                               double xy_sigma,
                               double z_sigma)
    {
      if (!tc_use_wls_position_prior_)
        return;
      // Rotation components are unconstrained here.  Roll/pitch are handled by
      // addTcAttitudePrior(), while translation uses the requested WLS sigmas.
      gtsam::Vector6 sig;
      sig << 1e6, 1e6, 1e6,
          xy_sigma, xy_sigma, z_sigma;
      graph.addPrior(X(k),
                     gtsam::Pose3(gtsam::Rot3::identity(),
                                  gtsam::Point3(wls_p.x(), wls_p.y(), wls_p.z())),
                     gtsam::noiseModel::Diagonal::Sigmas(sig));
    }

    void addTcWlsPositionPrior(gtsam::NonlinearFactorGraph &graph,
                               size_t k,
                               const Eigen::Vector3d &wls_p)
    {
      addTcWlsPositionPrior(graph, k, wls_p, tc_wls_xy_sigma_, tc_wls_z_sigma_);
    }

    // addTcWlsPositionPriorDropout: softer WLS prior used when there is no
    // ImuFactor (dropout cycle).  Without IMU the orientation is unconstrained
    // so the XY sigma is widened slightly; Z uses the same dropout sigma.
    void addTcWlsPositionPriorDropout(gtsam::NonlinearFactorGraph &graph,
                                      size_t k,
                                      const Eigen::Vector3d &wls_p)
    {
      if (!tc_use_wls_position_prior_)
        return;
      gtsam::Vector6 sig;
      sig << 1e6, 1e6, 1e6,
          tc_wls_dropout_xy_sigma_, tc_wls_dropout_xy_sigma_, tc_wls_z_sigma_;
      graph.addPrior(X(k),
                     gtsam::Pose3(gtsam::Rot3::identity(),
                                  gtsam::Point3(wls_p.x(), wls_p.y(), wls_p.z())),
                     gtsam::noiseModel::Diagonal::Sigmas(sig));
    }

    double wlsInnovationDistance(const Eigen::Vector3d &wls_p) const
    {
      const auto &tp = tc_.pose.translation();
      const double dx = wls_p.x() - tp.x();
      const double dy = wls_p.y() - tp.y();
      return std::sqrt(dx * dx + dy * dy); // XY only — Z less reliable
    }

    bool wlsNeedsRescuePrior(const Eigen::Vector3d &wls_p) const
    {
      return tc_wls_rescue_gate_ > 0.0 &&
             wlsInnovationDistance(wls_p) > tc_wls_rescue_gate_;
    }

    // wlsPassesInnovationGate: returns true if the WLS solution is consistent
    // with the current TC position estimate within wls_innovation_gate_ metres.
    //
    // Purpose: prevent a stale/drifted TC position from being forcibly anchored
    // to a WLS point that is far from the linearisation point.  When TC has
    // drifted 0.8 m from WLS, adding a tight WLS prior creates a large residual
    // that destabilises the ISAM2 Cholesky factorisation and can cause the bias
    // absorb the position error.  The gate rejects the prior when the gap is too
    // large, allowing TDOA factors alone to pull TC back gradually.
    bool wlsPassesInnovationGate(const Eigen::Vector3d &wls_p) const
    {
      if (wls_innovation_gate_ <= 0.0)
        return true; // gate disabled
      const double dist = wlsInnovationDistance(wls_p);
      if (dist > wls_innovation_gate_)
      {
        ROS_WARN_THROTTLE(0.5,
                          "[TC-FGO] WLS innovation gate rejected: WLS↔TC_XY=%.3f m > gate=%.3f m",
                          dist, wls_innovation_gate_);
        return false;
      }
      return true;
    }

    // ===========================================================================
    // TC-FGO DROPOUT UPDATE
    //
    // Called when n_imu < min_imu_per_cycle but WLS is available.
    // No ImuFactor is added (no IMU data), but we still:
    //   1. Add a new keyframe with a PriorFactor on position from WLS
    //   2. Add a tight velocity prior to zero (or near-zero) to kill phantom drift
    //   3. Add TDOA factors to constrain position from raw measurements
    //   4. Carry bias forward via BetweenFactor (no IMU but bias shouldn't jump)
    //
    // This prevents the >0.8 m phantom drift observed during 0.6 s IMU dropouts
    // while the vehicle is moving at ~1 m/s.
    // ===========================================================================
    void tcDropoutUpdate(double t_mid, const std::vector<TdoaMeas> &meas,
                         bool wls_ok, const Eigen::Vector3d &wls_p)
    {
      if (!wls_ok)
      {
        // No IMU and no WLS — nothing to anchor the state.  Update timestamp only.
        ROS_WARN_THROTTLE(1.0, "[TC-FGO] Dropout: no IMU, no WLS — skipping keyframe.");
        tc_.last_cycle_t = t_mid;
        return;
      }

      // Apply WLS innovation gate — reject if WLS is too far from current TC
      const bool wls_gated = wlsPassesInnovationGate(wls_p);

      ++tc_.cycle_idx;
      const size_t k = tc_.cycle_idx;
      const size_t k_pre = tc_.cycle_idx - 1;

      gtsam::NonlinearFactorGraph graph;
      gtsam::Values values;

      // ── Position: WLS prior (softer than normal — no IMU orientation) ────────
      if (wls_gated)
        addTcWlsPositionPriorDropout(graph, k, wls_p);
      addTcAttitudePrior(graph, k, tc_.pose.translation());

      // ── Velocity: tight prior pulling toward current estimate ─────────────────
      // During a dropout the last velocity is stale.  We damp it toward the
      // current value with a tight sigma to stop phantom integration.
      // Use the current tc_.vel as the prior mean (not zero) so we don't
      // violently brake a fast-moving vehicle — just prevent further growth.
      graph.addPrior(V(k), tc_.vel,
                     gtsam::noiseModel::Isotropic::Sigma(3, tc_dropout_vel_sigma_));

      // ── Bias: carry forward — no IMU means bias cannot update, just hold ─────
      graph.add(tcBiasBetween(k_pre, k, t_mid - tc_.last_cycle_t));
      graph.addPrior(B(k), tc_.bias,
                     gtsam::noiseModel::Isotropic::Sigma(6, tc_update_bias_sigma_));

      // ── Vertical prior ────────────────────────────────────────────────────────
      const gtsam::Point3 &pos_hint = tc_.pose.translation();
      if (wls_ok)
        addTcVerticalPrior(graph, k, pos_hint, wls_p.z());
      else
        addTcVerticalPrior(graph, k, pos_hint, pos_hint.z());

      // ── TDOA factors ──────────────────────────────────────────────────────────
      for (const auto &m : meas)
        graph.add(TdoaFactor(X(k), anchors_[m.idA], anchors_[m.idB],
                             -m.tdoa, tdoa_noise_));

      // Initial value: hold current pose (no IMU prediction)
      values.insert(X(k), tc_.pose);
      values.insert(V(k), tc_.vel);
      values.insert(B(k), tc_.bias);

      try
      {
        tc_isam_->update(graph, values);
        tc_isam_->update();

        const gtsam::Values est = tc_isam_->calculateEstimate();
        tc_.pose = est.at<gtsam::Pose3>(X(k));
        tc_.vel = est.at<gtsam::Vector3>(V(k));
        tc_.bias = est.at<gtsam::imuBias::ConstantBias>(B(k));
      }
      catch (const std::exception &e)
      {
        ROS_ERROR("[TC-FGO] Dropout update failed at k=%zu t=%.3f: %s",
                  k, t_mid, e.what());
        tc_.cycle_idx = k_pre;
      }

      tc_preint_->resetIntegrationAndSetBias(tc_.bias);
      tc_.last_cycle_t = t_mid;

      ROS_WARN_THROTTLE(0.5, "[TC-FGO] Dropout update k=%zu t=%.3f  wls_gated=%s  "
                             "pos=[%.3f,%.3f,%.3f]",
                        k, t_mid, wls_gated ? "OK" : "REJECTED",
                        tc_.pose.x(), tc_.pose.y(), tc_.pose.z());
    }

    void tcInitGraph(double t_mid, const std::vector<TdoaMeas> &meas,
                     bool wls_ok, const Eigen::Vector3d &wls_p)
    {
      tc_.cycle_idx = 0;
      Eigen::Vector3d seed_p(initial_pose_.x(), initial_pose_.y(), initial_pose_.z());
      gtsam::Rot3 seed_r = initial_pose_.rotation();
      gtsam::Vector3 seed_v = gtsam::Vector3::Zero();
      gtsam::imuBias::ConstantBias seed_b;

      if (wls_ok)
      {
        seed_p = wls_p;
      }

      tc_.pose = gtsam::Pose3(seed_r, gtsam::Point3(seed_p.x(), seed_p.y(), seed_p.z()));
      tc_.vel = seed_v;
      tc_.bias = seed_b;

      gtsam::NonlinearFactorGraph graph;
      gtsam::Values values;

      graph.addPrior(X(0), tc_.pose,
                     gtsam::noiseModel::Diagonal::Sigmas(tc_prior_pose_sigmas_));
      graph.addPrior(V(0), tc_.vel,
                     gtsam::noiseModel::Isotropic::Sigma(3, tc_prior_vel_sigma_));
      graph.addPrior(B(0), tc_.bias,
                     gtsam::noiseModel::Isotropic::Sigma(6, tc_prior_bias_sigma_));

      if (wls_ok)
        addTcWlsPositionPrior(graph, 0, wls_p);
      addTcAttitudePrior(graph, 0, tc_.pose.translation());
      if (wls_ok)
        addTcVerticalPrior(graph, 0, tc_.pose.translation(), wls_p.z());

      for (const auto &m : meas)
        graph.add(TdoaFactor(X(0), anchors_[m.idA], anchors_[m.idB],
                             -m.tdoa, tdoa_noise_));

      values.insert(X(0), tc_.pose);
      values.insert(V(0), tc_.vel);
      values.insert(B(0), tc_.bias);

      tc_isam_->update(graph, values);
      tc_isam_->update();

      gtsam::Values est = tc_isam_->calculateEstimate();
      tc_.pose = est.at<gtsam::Pose3>(X(0));
      tc_.vel = est.at<gtsam::Vector3>(V(0));
      tc_.bias = est.at<gtsam::imuBias::ConstantBias>(B(0));

      tc_preint_->resetIntegrationAndSetBias(tc_.bias);
      tc_.initialized = true;
      tc_.last_cycle_t = t_mid;

      ROS_INFO("[TC-FGO] Init  t=%.3f  seed=[%.3f,%.3f,%.3f]  pos=[%.3f,%.3f,%.3f]",
               t_mid, seed_p.x(), seed_p.y(), seed_p.z(),
               tc_.pose.x(), tc_.pose.y(), tc_.pose.z());
    }

    // ===========================================================================
    // TC-FGO UPDATE
    // Per keyframe k: ImuFactor + BiasBetween + TdoaFactor × N
    // ===========================================================================
    void tcUpdateGraph(double t_mid, const std::vector<TdoaMeas> &meas,
                       bool wls_ok, const Eigen::Vector3d &wls_p)
    {
      tc_preint_->resetIntegrationAndSetBias(tc_.bias);
      int n_imu = replayImu(tc_.last_cycle_t, t_mid, *tc_preint_);

      if (n_imu < min_imu_per_cycle_)
      {
        // IMU dropout — route to dedicated dropout handler instead of skipping.
        // Skipping leaves velocity frozen and position unanchored, which causes
        // phantom drift of speed×gap_duration (observed: up to 0.83 m per gap).
        tcDropoutUpdate(t_mid, meas, wls_ok, wls_p);
        return;
      }

      ++tc_.cycle_idx;
      const size_t k = tc_.cycle_idx;
      const size_t k_pre = tc_.cycle_idx - 1;

      const gtsam::NavState pred_nav = tc_preint_->predict(
          gtsam::NavState(tc_.pose, tc_.vel), tc_.bias);

      gtsam::NonlinearFactorGraph graph;
      gtsam::Values values;

      graph.add(gtsam::ImuFactor(
          X(k_pre), V(k_pre), X(k), V(k), B(k_pre), *tc_preint_));
      graph.add(tcBiasBetween(k_pre, k, t_mid - tc_.last_cycle_t));

      graph.addPrior(V(k), pred_nav.velocity(),
                     gtsam::noiseModel::Isotropic::Sigma(3, tc_update_vel_sigma_));
      graph.addPrior(B(k), tc_.bias,
                     gtsam::noiseModel::Isotropic::Sigma(6, tc_update_bias_sigma_));

      // Apply WLS innovation gate before adding position prior.
      // When TC has drifted far from WLS (e.g. after a dropout), adding a tight
      // WLS prior against a badly linearised point amplifies the error rather
      // than correcting it.  The gate rejects the prior when the XY gap exceeds
      // wls_innovation_gate_, letting TDOA factors pull TC back gradually.
      const bool wls_gated = wls_ok && wlsPassesInnovationGate(wls_p);
      if (wls_gated)
      {
        if (wlsNeedsRescuePrior(wls_p))
        {
          addTcWlsPositionPrior(graph, k, wls_p,
                                tc_wls_rescue_xy_sigma_,
                                tc_wls_rescue_z_sigma_);
          ROS_WARN_THROTTLE(0.5,
                            "[TC-FGO] WLS rescue prior active: WLS↔TC_XY=%.3f m "
                            "(xy_sigma=%.3f z_sigma=%.3f)",
                            wlsInnovationDistance(wls_p),
                            tc_wls_rescue_xy_sigma_,
                            tc_wls_rescue_z_sigma_);
        }
        else
        {
          addTcWlsPositionPrior(graph, k, wls_p);
        }
        addTcVerticalPrior(graph, k, pred_nav.pose().translation(), wls_p.z());
      }
      else
      {
        addTcVerticalPrior(graph, k, pred_nav.pose().translation(), pred_nav.pose().z());
      }
      addTcAttitudePrior(graph, k, pred_nav.pose().translation());

      for (const auto &m : meas)
        graph.add(TdoaFactor(X(k), anchors_[m.idA], anchors_[m.idB],
                             -m.tdoa, tdoa_noise_));

      values.insert(X(k), pred_nav.pose());
      values.insert(V(k), pred_nav.velocity());
      values.insert(B(k), tc_.bias);

      try
      {
        tc_isam_->update(graph, values);
        tc_isam_->update();

        const gtsam::Values est = tc_isam_->calculateEstimate();
        tc_.pose = est.at<gtsam::Pose3>(X(k));
        tc_.vel = est.at<gtsam::Vector3>(V(k));
        tc_.bias = est.at<gtsam::imuBias::ConstantBias>(B(k));
      }
      catch (const std::exception &e)
      {
        ROS_ERROR("[TC-FGO] ISAM update failed at k=%zu t=%.3f: %s",
                  k, t_mid, e.what());
        tc_.cycle_idx = k_pre;
        tc_.last_cycle_t = t_mid;
        tc_preint_->resetIntegrationAndSetBias(tc_.bias);
        return;
      }

      tc_preint_->resetIntegrationAndSetBias(tc_.bias);
      tc_.last_cycle_t = t_mid;
    }

    // ===========================================================================
    // PUBLISH HELPERS
    // ===========================================================================
    void publishOdom(ros::Publisher &pub, const ros::Time &stamp,
                     double x, double y, double z,
                     const gtsam::Rot3 &rot, const gtsam::Vector3 &vel)
    {
      Eigen::Quaterniond q(rot.matrix());
      nav_msgs::Odometry odom;
      odom.header.stamp = stamp;
      odom.header.frame_id = odom_frame_;
      odom.child_frame_id = base_frame_;
      odom.pose.pose.position.x = x;
      odom.pose.pose.position.y = y;
      odom.pose.pose.position.z = z;
      odom.pose.pose.orientation.x = q.x();
      odom.pose.pose.orientation.y = q.y();
      odom.pose.pose.orientation.z = q.z();
      odom.pose.pose.orientation.w = q.w();
      odom.twist.twist.linear.x = vel.x();
      odom.twist.twist.linear.y = vel.y();
      odom.twist.twist.linear.z = vel.z();
      pub.publish(odom);
    }

    void appendPath(nav_msgs::Path &path, ros::Publisher &pub,
                    const ros::Time &stamp,
                    double x, double y, double z, const gtsam::Rot3 &rot)
    {
      Eigen::Quaterniond q(rot.matrix());
      geometry_msgs::PoseStamped ps;
      ps.header.stamp = stamp;
      ps.header.frame_id = odom_frame_;
      ps.pose.position.x = x;
      ps.pose.position.y = y;
      ps.pose.position.z = z;
      ps.pose.orientation.x = q.x();
      ps.pose.orientation.y = q.y();
      ps.pose.orientation.z = q.z();
      ps.pose.orientation.w = q.w();
      path.header.stamp = stamp;
      path.header.frame_id = odom_frame_;
      path.poses.push_back(ps);
      pub.publish(path);
    }

    void broadcastTf(const ros::Time &stamp, const gtsam::Point3 &pos,
                     const gtsam::Rot3 &rot, const std::string &child)
    {
      Eigen::Quaterniond q(rot.matrix());
      geometry_msgs::TransformStamped tf;
      tf.header.stamp = stamp;
      tf.header.frame_id = odom_frame_;
      tf.child_frame_id = child;
      tf.transform.translation.x = pos.x();
      tf.transform.translation.y = pos.y();
      tf.transform.translation.z = pos.z();
      tf.transform.rotation.x = q.x();
      tf.transform.rotation.y = q.y();
      tf.transform.rotation.z = q.z();
      tf.transform.rotation.w = q.w();
      tf_broadcaster_.sendTransform(tf);
    }
    // ===========================================================================
    // CONSOLE
    // ===========================================================================

    void logConfiguration() const
    {
      ROS_INFO("[uwb_imu_fusion] %zu anchors  meas/cycle=%d",
               anchors_.size(), measurements_per_cycle_);
      ROS_INFO("[uwb_imu_fusion] gravity=%.3f  tdoa_sigma=%.3f",
               gravity_mag_, tdoa_sigma_);
      ROS_INFO("[uwb_imu_fusion] accel units=%s  accel_scale=%.6f  gyro_units=%s",
               imu_accel_in_g_ ? "g" : "m/s^2",
               accel_scale_,
               gyro_is_degrees_ ? "deg/s" : "rad/s");
      ROS_INFO("[uwb_imu_fusion] TC IMU noise: accel_sigma=%.4f [m/s^2] gyro_sigma=%.5f [rad/s]",
               tc_accel_sigma_, tc_gyro_sigma_);
      ROS_INFO("[uwb_imu_fusion] TC IMU bias rw: accel=%.5f gyro=%.6f",
               tc_accel_bias_rw_sigma_, tc_gyro_bias_rw_sigma_);
      ROS_INFO("[uwb_imu_fusion] WLS prior: xy_sigma=%.3f z_sigma=%.3f",
               tc_wls_xy_sigma_, tc_wls_z_sigma_);
      ROS_INFO("[uwb_imu_fusion] update regularization: tc_vel=%.3f tc_bias=%.3f",
               tc_update_vel_sigma_, tc_update_bias_sigma_);
      ROS_INFO("[uwb_imu_fusion] imu leveling: %s  samples=%d",
               estimate_initial_orientation_from_imu_ ? "enabled" : "disabled",
               initial_alignment_imu_samples_);
      ROS_INFO("[uwb_imu_fusion] tc vertical prior: %s  sigma=%.3f",
               tc_use_vertical_prior_ ? "enabled" : "disabled",
               tc_vertical_prior_sigma_);
      ROS_INFO("[uwb_imu_fusion] tc WLS position prior: %s  xy_sigma=%.3f z_sigma=%.3f",
               tc_use_wls_position_prior_ ? "enabled" : "disabled",
               tc_wls_xy_sigma_, tc_wls_z_sigma_);
      ROS_INFO("[uwb_imu_fusion] tc WLS rescue: gate=%.3f xy_sigma=%.3f z_sigma=%.3f",
               tc_wls_rescue_gate_, tc_wls_rescue_xy_sigma_, tc_wls_rescue_z_sigma_);
      ROS_INFO("[uwb_imu_fusion] tc attitude prior: %s  roll_pitch_sigma=%.3f yaw_sigma=%.1e",
               tc_use_attitude_prior_ ? "enabled" : "disabled",
               tc_attitude_roll_pitch_sigma_, tc_attitude_yaw_sigma_);
    }

    void printCycleInfo(double t_mid, bool wls_ok,
                        const Eigen::Vector3d &wls_p) const
    {
      auto err3 = [&](const gtsam::Point3 &p) -> double
      {
        return gt_have_ ? std::sqrt(std::pow(p.x() - gt_pos_.x(), 2) +
                                    std::pow(p.y() - gt_pos_.y(), 2) +
                                    std::pow(p.z() - gt_pos_.z(), 2))
                        : -1.0;
      };
      ROS_INFO("=================================================");
      ROS_INFO("[TC=%4zu | t=%.3f s]", tc_.cycle_idx, t_mid);
      if (wls_ok)
        ROS_INFO("  WLS    : [%7.3f,%7.3f,%7.3f]", wls_p.x(), wls_p.y(), wls_p.z());
      else
        ROS_INFO("  WLS    : DID NOT CONVERGE");
      if (tc_.initialized)
      {
        const auto &tp = tc_.pose.translation();
        const auto rpy = tc_.pose.rotation().rpy();
        ROS_INFO("  TC-FGO : [%7.3f,%7.3f,%7.3f]  rpy=[%.1f,%.1f,%.1f]deg",
                 tp.x(), tp.y(), tp.z(), rpy.x() * 180 / M_PI, rpy.y() * 180 / M_PI, rpy.z() * 180 / M_PI);
        ROS_INFO("           vel=[%.3f,%.3f,%.3f]  ba=[%.4f,%.4f,%.4f] m/s2",
                 tc_.vel.x(), tc_.vel.y(), tc_.vel.z(),
                 tc_.bias.accelerometer().x(), tc_.bias.accelerometer().y(),
                 tc_.bias.accelerometer().z());
      }
      if (gt_have_)
      {
        ROS_INFO("  GT     : [%7.3f,%7.3f,%7.3f]", gt_pos_.x(), gt_pos_.y(), gt_pos_.z());
        if (wls_ok)
          ROS_INFO("  |WLS-GT| = %.4f m",
                   (wls_p - Eigen::Vector3d(gt_pos_.x(), gt_pos_.y(), gt_pos_.z())).norm());
        if (tc_.initialized)
          ROS_INFO("  |TC -GT| = %.4f m", err3(tc_.pose.translation()));
      }
      ROS_INFO("=================================================");
    }

    // ===========================================================================
    // MEMBERS
    // ===========================================================================
    ros::NodeHandle nh_, pnh_;
    ros::Subscriber imu_sub_, uwb_sub_, gt_sub_;

    ros::Publisher wls_odom_pub_, tc_odom_pub_;
    ros::Publisher wls_path_pub_, tc_path_pub_, gt_path_pub_;

    tf2_ros::TransformBroadcaster tf_broadcaster_;
    std::mutex mutex_;

    std::vector<gtsam::Point3> anchors_;
    std::string odom_frame_{"map"}, base_frame_{"base_link"};

    std::vector<TdoaMeas> current_cycle_;
    std::set<std::pair<int, int>> cycle_pairs_;
    int measurements_per_cycle_{8}, min_imu_per_cycle_{2};
    double cycle_timeout_{0.1};

    TdoaWlsSolver wls_solver_;
    Eigen::Vector3d wls_guess_{0, 0, 1};

    std::deque<ImuMeas> imu_buf_;
    double last_imu_t_{-1.0};

    // Gyro unit sanity check — evaluated once over first 200 samples
    int gyro_sanity_count_{0};
    double gyro_sanity_raw_norm_sum_{0.0};

    // IMU unit conversion — loaded in strict order in loadParams()
    double accel_scale_{9.80665}; // default: assume g input → convert to m/s²
    bool imu_accel_in_g_{true};
    bool gyro_is_degrees_{true}; // FIX: was false — sensor publishes deg/s
    double gravity_mag_{9.80665};
    bool estimate_initial_orientation_from_imu_{true};
    int initial_alignment_imu_samples_{100};
    bool initial_orientation_aligned_{false};
    gtsam::Vector3 alignment_acc_sum_{gtsam::Vector3::Zero()};
    int alignment_acc_count_{0};

    // TC IMU noise — data-measured values from static-period analysis
    double tc_accel_sigma_{0.04};          // [m/s²]  measured ~0.04 m/s² per axis
    double tc_gyro_sigma_{0.003};          // [rad/s]  measured ~0.0025 rad/s per axis
    double tc_accel_bias_rw_sigma_{0.01};  // [m/s²/√s] real accel bias ~0.22 m/s²
    double tc_gyro_bias_rw_sigma_{0.0003}; // [rad/s/√s] gyro bias small and stable
    double tc_integration_sigma_{1e-4};    // raised: ~974 Hz IMU rate, many steps

    // UWB noise — balanced for NLOS robustness + Huber kernel
    double tdoa_sigma_{0.3}; // [m] — balanced: 0.15 too tight, 0.5 too loose
    bool use_robust_noise_{true};
    double huber_k_{1.345};
    gtsam::SharedNoiseModel tdoa_noise_;

    // TC priors
    gtsam::Vector6 tc_prior_pose_sigmas_;
    double tc_prior_vel_sigma_{0.1};
    double tc_prior_bias_sigma_{0.1}; // FIX: was 0.001 — too tight, pinned bias at zero
    double tc_update_vel_sigma_{0.5}; // FIX: was 2.0 — too loose, velocity drifts
    double tc_update_bias_sigma_{0.1};
    bool tc_use_vertical_prior_{true};
    double tc_vertical_prior_sigma_{0.2};
    bool tc_use_wls_position_prior_{true};
    double tc_wls_xy_sigma_{0.25}; // [m] balanced XY trust
    double tc_wls_z_sigma_{0.35};  // [m] looser Z: WLS Z less reliable
    double tc_wls_rescue_gate_{0.25};     // [m] strengthen WLS prior above this XY gap
    double tc_wls_rescue_xy_sigma_{0.08}; // [m] rescue-mode horizontal WLS prior
    double tc_wls_rescue_z_sigma_{0.25};  // [m] rescue-mode vertical WLS prior
    bool tc_use_attitude_prior_{true};
    double tc_attitude_roll_pitch_sigma_{0.05}; // [rad] strong roll/pitch leveling
    double tc_attitude_yaw_sigma_{1e6};          // [rad] yaw effectively free
    // IMU-dropout fallback parameters
    double tc_wls_dropout_xy_sigma_{0.40}; // [m] softer WLS prior when no IMU
    double tc_dropout_vel_sigma_{0.1};     // [m/s] tight vel prior to stop phantom drift
    double wls_innovation_gate_{0.8};      // [m] max WLS↔TC XY gap before rejecting prior

    gtsam::Pose3 initial_pose_;

    boost::shared_ptr<gtsam::PreintegrationParams> tc_imu_params_;

    // TC branch
    FgoState tc_;
    boost::shared_ptr<gtsam::PreintegratedImuMeasurements> tc_preint_;
    boost::shared_ptr<gtsam::ISAM2> tc_isam_;

    bool gt_have_{false};
    gtsam::Point3 gt_pos_{gtsam::Point3::Zero()};

    nav_msgs::Path wls_path_, tc_path_, gt_path_;

    std::string traj_log_path_{"/tmp/uwb_imu_trajectory.csv"};
    std::ofstream traj_log_;
    std::string imu_log_path_{"/tmp/uwb_imu_raw_imu.csv"};
    std::ofstream imu_log_;
    int imu_log_flush_counter_{0};
  };

} // namespace uwb_imu_fusion

int main(int argc, char **argv)
{
  ros::init(argc, argv, "uwb_imu_fusion_node");
  ros::NodeHandle nh, pnh("~");
  uwb_imu_fusion::UwbImuFusionNode node(nh, pnh);
  ros::AsyncSpinner spinner(2);
  spinner.start();
  ros::waitForShutdown();
  return 0;
}
