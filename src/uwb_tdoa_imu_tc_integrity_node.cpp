// =============================================================================
// uwb_tdoa_imu_tc_integrity_node.cpp  —  Dataset-specialized tightly coupled
//                                        UWB TDOA + IMU fusion with integrity
//
// PIPELINE
// ─────────────────────────────────────────────────────────────────────────────
//  UWB stream → cycle accumulator (validated, unchanged)
//                    │
//                    ├─► WLS solver  ──────────────────────────────────────────►  uwb_wls / uwb_wls_path
//                    │
//                    └─► TC-FGO  (IntegrityTdoaFactor × N
//                                  + ImuFactor + BiasBetween)  ───────────────► tc_fusion / tc_fusion_path
//
//  IMU stream → imu_buf_
//
// TIGHTLY COUPLED (TC)
//   Runs a RAIM-style integrity check at each UWB cycle, then adds one
//   IntegrityTdoaFactor per accepted TDOA measurement. The factor residual is
//   still raw TDOA geometry, but its noise model is shaped by integrity:
//     • FDE-selected measurements are excluded before graph insertion.
//     • Large standardized residuals inflate sigma instead of pulling the graph.
//     • Healthy measurements keep the nominal/dynamic sigma.
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
//   WLS prior → trusted for XY, moderately trusted for Z.  Z is weaker than XY,
//              but flight data changes altitude, so the TC vertical reference
//              follows WLS unless tc_vertical_prior_source is explicitly changed.
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
#include <map>
#include <set>
#include <sstream>
#include <utility>
#include <vector>
#include <cmath>
#include <string>
#include <numeric>

#include "tdoa_factor.h"
#include "wlssolver.h"
#include "integrity_monitor.h"
#include "integrity_tdoa_factor.h"

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

  struct GtMeas
  {
    double t;
    gtsam::Point3 p;
  };

  using TdoaMeasurements = std::vector<TdoaMeas>;

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
      anchor_extrinsic_translation_ = Eigen::Vector3d(ext_t[0], ext_t[1], ext_t[2]);
      Eigen::Quaterniond qe(ext_q[3], ext_q[0], ext_q[1], ext_q[2]);
      qe.normalize();
      Eigen::Vector3d te = anchor_extrinsic_translation_;
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
      get("tc_accel_noise_sigma", tc_accel_sigma_, 0.05);             // const4 CSV static accel floor
      get("tc_gyro_noise_sigma", tc_gyro_sigma_, 0.01);               // const4 CSV gyro is noisy during motion
      get("tc_accel_bias_rw_sigma", tc_accel_bias_rw_sigma_, 0.02);   // allow accel bias to absorb IMU mismatch
      get("tc_gyro_bias_rw_sigma", tc_gyro_bias_rw_sigma_, 0.0006);   // allow gyro bias to adapt
      get("tc_integration_noise_sigma", tc_integration_sigma_, 1e-2); // CSV-tuned: avoid overconfident preintegration

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
      get("tdoa_sigma", tdoa_sigma_, 0.35); // const4 CSV: keep UWB strong, leave room for multipath
      get("use_robust_noise", use_robust_noise_, true);
      get("huber_k", huber_k_, 1.345);
      get("use_dynamic_tdoa_sigma", use_dynamic_tdoa_sigma_, true);
      get("tdoa_sigma_min", tdoa_sigma_min_, 0.05);
      get("tdoa_sigma_max", tdoa_sigma_max_, 1.2);
      get("tdoa_residual_sigma_scale", tdoa_residual_sigma_scale_, 0.75);
      get("wls_refine_with_dynamic_sigma", wls_refine_with_dynamic_sigma_, true);
      get("cycle_timeout", cycle_timeout_, 0.06);
      int mpc = 8, min_cycle = 6, min_imu = 1;
      get("measurements_per_cycle", mpc, 8);
      get("min_cycle_measurements", min_cycle, 6);
      get("min_imu_per_cycle", min_imu, 1);
      get("close_epoch_on_count", close_epoch_on_count_, false);
      get("max_epoch_measurements", max_epoch_measurements_, 24);
      get("compress_epoch_pairs", compress_epoch_pairs_, false);
      measurements_per_cycle_ = mpc;
      min_cycle_measurements_ = std::min(std::max(min_cycle, 3), measurements_per_cycle_);
      min_imu_per_cycle_ = min_imu;
      max_epoch_measurements_ =
          std::max(max_epoch_measurements_, min_cycle_measurements_);
      get("tc_use_tdoa_residual_gate", tc_use_tdoa_residual_gate_, true);
      get("tc_tdoa_residual_gate", tc_tdoa_residual_gate_, 1.2);
      get("tc_min_tdoa_factors", tc_min_tdoa_factors_, 4);
      tc_min_tdoa_factors_ = std::min(std::max(tc_min_tdoa_factors_, 3), measurements_per_cycle_);

      get("tc_integrity_enabled", tc_integrity_enabled_, true);
      get("tc_integrity_tdoa_noise_sigma", tc_integrity_monitor_.tdoa_noise_sigma, 0.08);
      get("tc_integrity_p_fa", tc_integrity_monitor_.p_fa, 0.001);
      get("tc_integrity_hal", tc_integrity_monitor_.hal, 1.0);
      get("tc_integrity_val", tc_integrity_monitor_.val, 2.0);
      get("tc_integrity_ring_threshold", tc_integrity_monitor_.ring_threshold, 0.10);
      get("tc_integrity_enable_fde", tc_integrity_monitor_.enable_fde, true);
      get("tc_integrity_exclude_fault", tc_integrity_exclude_fault_, true);
      get("tc_integrity_sigma_inflation", tc_integrity_sigma_inflation_, 4.0);
      get("tc_integrity_std_resid_soft_gate", tc_integrity_std_resid_soft_gate_, 2.5);
      get("tc_integrity_std_resid_hard_gate", tc_integrity_std_resid_hard_gate_, 5.0);
      get("tc_integrity_gate_wls_on_unavailable",
          tc_integrity_gate_wls_on_unavailable_, true);
      get("tc_integrity_unavailable_sigma_scale",
          tc_integrity_unavailable_sigma_scale_, 3.0);
      get("tc_integrity_min_factors", tc_integrity_min_factors_, 4);
      tc_integrity_min_factors_ =
          std::min(std::max(tc_integrity_min_factors_, 3), measurements_per_cycle_);
      get("tc_nlos_anchor_rejection_enabled",
          tc_nlos_anchor_rejection_enabled_, true);
      get("tc_nlos_anchor_min_incident_measurements",
          tc_nlos_anchor_min_incident_measurements_, 2);
      get("tc_nlos_anchor_min_remaining_measurements",
          tc_nlos_anchor_min_remaining_measurements_, 5);
      tc_nlos_anchor_min_remaining_measurements_ =
          std::min(std::max(tc_nlos_anchor_min_remaining_measurements_, 4),
                   measurements_per_cycle_);
      get("tc_nlos_anchor_hpl_improvement_ratio",
          tc_nlos_anchor_hpl_improvement_ratio_, 0.85);
      get("tc_nlos_anchor_chi2_improvement_ratio",
          tc_nlos_anchor_chi2_improvement_ratio_, 0.80);
      get("tc_nlos_anchor_max_clean_wls_shift",
          tc_nlos_anchor_max_clean_wls_shift_, 2.0);

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
      std::vector<double> tc_pps{0.5, 0.5, 0.5, 0.05, 0.05, 0.20};
      if (!nh_.getParam("/uwb_imu_fusion/tc_prior_pose_sigmas", tc_pps))
        pnh_.getParam("tc_prior_pose_sigmas", tc_pps);
      tc_prior_pose_sigmas_ = gtsam::Vector6::Map(tc_pps.data());
      get("tc_prior_vel_sigma", tc_prior_vel_sigma_, 0.1);
      get("tc_prior_bias_sigma", tc_prior_bias_sigma_, 0.1); // FIX: was 0.001 — too tight, froze bias at zero
      get("tc_update_vel_sigma", tc_update_vel_sigma_, 0.8); // looser: reduce IMU-predicted velocity authority
      get("tc_update_bias_sigma", tc_update_bias_sigma_, 0.2);
      get("tc_use_wls_velocity_prior", tc_use_wls_velocity_prior_, true);
      get("tc_wls_velocity_sigma", tc_wls_velocity_sigma_, 0.15);
      get("tc_wls_velocity_max_speed", tc_wls_velocity_max_speed_, 1.2);
      get("tc_wls_velocity_max_dt", tc_wls_velocity_max_dt_, 0.25);
      get("tc_use_wls_z_velocity_prior", tc_use_wls_z_velocity_prior_, true);
      get("tc_wls_z_velocity_sigma", tc_wls_z_velocity_sigma_, 0.25);
      get("tc_wls_z_velocity_max_speed", tc_wls_z_velocity_max_speed_, 0.8);
      get("tc_use_vertical_prior", tc_use_vertical_prior_, true);
      get("tc_vertical_prior_sigma", tc_vertical_prior_sigma_, 0.15);
      get("tc_vertical_prior_source", tc_vertical_prior_source_, std::string("wls"));
      get("tc_vertical_prior_z_ref", tc_vertical_prior_z_ref_,
          std::numeric_limits<double>::quiet_NaN());
      get("tc_wls_z_bias_correction", tc_wls_z_bias_correction_, 0.0);
      get("tc_wls_z_filter_enabled", tc_wls_z_filter_enabled_, false);
      get("tc_wls_z_filter_alpha", tc_wls_z_filter_alpha_, 0.65);
      get("tc_wls_z_filter_outlier_gate", tc_wls_z_filter_outlier_gate_, 0.45);
      get("tc_wls_z_filter_outlier_alpha", tc_wls_z_filter_outlier_alpha_, 0.20);

      // WLS position prior sigmas
      //   XY: 0.25 m — WLS proven accurate in horizontal plane
      //   Z:  0.35 m — WLS Z is noisier (poor vertical anchor geometry);
      //               looser Z avoids vertical locking to a biased WLS height
      get("tc_use_wls_position_prior", tc_use_wls_position_prior_, true);
      get("tc_wls_xy_sigma", tc_wls_xy_sigma_, 0.10); // WLS target error is <0.1 m
      get("tc_wls_z_sigma", tc_wls_z_sigma_, 0.18);   // still looser than XY due to vertical geometry
      get("tc_wls_rescue_gate", tc_wls_rescue_gate_, 0.20);
      get("tc_wls_rescue_xy_sigma", tc_wls_rescue_xy_sigma_, 0.04);
      get("tc_wls_rescue_z_sigma", tc_wls_rescue_z_sigma_, 0.12);

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
      get("tc_wls_dropout_xy_sigma", tc_wls_dropout_xy_sigma_, 0.25); // more UWB pull during IMU gaps
      get("tc_dropout_vel_sigma", tc_dropout_vel_sigma_, 0.2);        // lower stale-IMU velocity authority
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
      if (!std::isfinite(tc_vertical_prior_z_ref_))
        tc_vertical_prior_z_ref_ = initial_pose_.z();

      get("odom_frame", odom_frame_, std::string("map"));
      get("base_frame", base_frame_, std::string("base_link"));
      get("gt_match_max_dt", gt_match_max_dt_, 0.02);
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
                  << "wls_clean_x,wls_clean_y,wls_clean_z,"
                  << "tc_x,tc_y,tc_z,"
                  << "tc_qx,tc_qy,tc_qz,tc_qw,"
                  << "tc_vx,tc_vy,tc_vz,"
                  << "tc_bias_ax,tc_bias_ay,tc_bias_az,"
                  << "tc_bias_gx,tc_bias_gy,tc_bias_gz,"
                  << "imu_n,imu_mean_ax,imu_mean_ay,imu_mean_az,"
                  << "imu_mean_gx,imu_mean_gy,imu_mean_gz,"
                  << "imu_accel_norm_mean,imu_gyro_norm_mean,"
                  << "wls_gt_err,wls_clean_gt_err,tc_gt_err,"
                  << "nlos_anchor_id,nlos_anchor_removed,clean_meas_count,"
                  << "clean_excluded_anchors,"
                  << "integ_used,integ_input_count,integ_factor_count,integ_excluded_count,"
                  << "integ_chi2,integ_threshold,integ_dof,integ_fault,"
                  << "integ_ring_sum,integ_ring_ok,integ_ring_closed_loops,"
                  << "integ_hpl,integ_vpl,integ_available,"
                  << "integ_excluded_idx,integ_chi2_after_fde,"
                  << "integ_hal,integ_val,integ_p_fa,integ_tdoa_noise_sigma,"
                  << "integ_soft_gate,integ_hard_gate,integ_sigma_inflation,"
                  << "integ_input_pairs,integ_factor_pairs,integ_excluded_pairs,"
                  << "integ_residuals,integ_std_resid,integ_input_sigmas,"
                  << "integ_factor_sigmas,integ_factor_std_resid,integ_factor_labels\n";
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

    struct TcIntegritySelection
    {
      TdoaMeasurements meas;
      Eigen::VectorXd sigmas;
      std::vector<double> std_resid;
      std::vector<std::string> labels;
      IntegrityResult result;
      std::vector<std::string> input_pairs;
      std::vector<std::string> factor_pairs;
      std::vector<std::string> excluded_pairs;
      std::vector<double> factor_sigmas;
      int input_count{0};
      int excluded_count{0};
      bool used_monitor{false};
    };

    struct CleanLosSelection
    {
      TdoaMeasurements meas;
      Eigen::VectorXd sigmas;
      Eigen::Vector3d wls_p{0, 0, 0};
      bool wls_ok{false};
      bool used_clean{false};
      int nlos_anchor_id{-1};
      std::vector<std::string> excluded_anchors;
      IntegrityResult all_result;
      IntegrityResult clean_result;
    };

    static std::string csvList(const std::vector<std::string> &items)
    {
      if (items.empty())
        return "none";
      std::ostringstream oss;
      for (size_t i = 0; i < items.size(); ++i)
      {
        if (i > 0) oss << "|";
        oss << items[i];
      }
      return oss.str();
    }

    static std::string csvList(const std::vector<double> &items)
    {
      if (items.empty())
        return "none";
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6);
      for (size_t i = 0; i < items.size(); ++i)
      {
        if (i > 0) oss << "|";
        oss << items[i];
      }
      return oss.str();
    }

    static std::string csvList(const Eigen::VectorXd &items)
    {
      if (items.size() == 0)
        return "none";
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6);
      for (int i = 0; i < items.size(); ++i)
      {
        if (i > 0) oss << "|";
        oss << items(i);
      }
      return oss.str();
    }

    static std::string formatRingClosureLoops(const std::vector<RingClosureLoop> &loops)
    {
      if (loops.empty())
        return "none";

      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6);
      for (size_t i = 0; i < loops.size(); ++i)
      {
        const auto &loop = loops[i];
        if (i > 0) oss << "; ";
        oss << loop.a << "-" << loop.b << "-" << loop.c
            << ": ab=" << loop.ab
            << " bc=" << loop.bc
            << " ca=" << loop.ca
            << " err=" << loop.error;
      }
      return oss.str();
    }

    void logTcRingClosureCheck(double t_mid,
                               const std::string &check_name,
                               const IntegrityResult &result) const
    {
      const std::string details = formatRingClosureLoops(result.ring_loops);
      ROS_INFO(
          "[TC-RingClose] t=%.3f check=%s loops=%d worst=%.6f m threshold=%.6f m ok=%s details=%s",
          t_mid, check_name.c_str(), result.ring_closed_loops,
          result.ring_sum, tc_integrity_monitor_.ring_threshold,
          result.ring_ok ? "true" : "false", details.c_str());
    }

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

    void logRow(double t,
                bool wls_ok, const Eigen::Vector3d &wls_p,
                bool clean_wls_ok, const Eigen::Vector3d &clean_wls_p,
                const CycleImuStats &imu_stats,
                const CleanLosSelection &clean,
                const TcIntegritySelection &integrity)
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
      gtsam::Point3 gt_at_t;
      const bool gt_at_t_ok = getGroundTruthAt(t, gt_at_t);
      Eigen::Vector3d gt_e(gt_at_t.x(), gt_at_t.y(), gt_at_t.z());

      traj_log_ << std::fixed << std::setprecision(6)
                << t << ","
                // ground truth
                << (gt_at_t_ok ? gt_at_t.x() : nan) << ","
                << (gt_at_t_ok ? gt_at_t.y() : nan) << ","
                << (gt_at_t_ok ? gt_at_t.z() : nan) << ","
                // WLS position
                << (wls_ok ? wls_p.x() : nan) << ","
                << (wls_ok ? wls_p.y() : nan) << ","
                << (wls_ok ? wls_p.z() : nan) << ","
                // Clean LOS WLS position
                << (clean_wls_ok ? clean_wls_p.x() : nan) << ","
                << (clean_wls_ok ? clean_wls_p.y() : nan) << ","
                << (clean_wls_ok ? clean_wls_p.z() : nan) << ","
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
                << (wls_ok && gt_at_t_ok ? (wls_p - gt_e).norm() : nan) << ","
                << (clean_wls_ok && gt_at_t_ok ? (clean_wls_p - gt_e).norm() : nan) << ","
                << (tc_.initialized && gt_at_t_ok ? (tp - gt_e).norm() : nan) << ","
                << clean.nlos_anchor_id << ","
                << (clean.used_clean ? 1 : 0) << ","
                << clean.meas.size() << ","
                << csvList(clean.excluded_anchors) << ","
                // integrity monitor summary
                << (integrity.used_monitor ? 1 : 0) << ","
                << integrity.input_count << ","
                << integrity.meas.size() << ","
                << integrity.excluded_count << ","
                << integrity.result.chi2_stat << ","
                << integrity.result.chi2_threshold << ","
                << integrity.result.dof << ","
                << (integrity.result.fault_detected ? 1 : 0) << ","
                << integrity.result.ring_sum << ","
                << (integrity.result.ring_ok ? 1 : 0) << ","
                << integrity.result.ring_closed_loops << ","
                << integrity.result.hpl << ","
                << integrity.result.vpl << ","
                << (integrity.result.available ? 1 : 0) << ","
                << integrity.result.excluded_idx << ","
                << integrity.result.chi2_after_fde << ","
                << tc_integrity_monitor_.hal << ","
                << tc_integrity_monitor_.val << ","
                << tc_integrity_monitor_.p_fa << ","
                << tc_integrity_monitor_.tdoa_noise_sigma << ","
                << tc_integrity_std_resid_soft_gate_ << ","
                << tc_integrity_std_resid_hard_gate_ << ","
                << tc_integrity_sigma_inflation_ << ","
                << csvList(integrity.input_pairs) << ","
                << csvList(integrity.factor_pairs) << ","
                << csvList(integrity.excluded_pairs) << ","
                << csvList(integrity.result.residuals) << ","
                << csvList(integrity.result.std_resid) << ","
                << csvList(integrity.result.sigmas) << ","
                << csvList(integrity.factor_sigmas) << ","
                << csvList(integrity.std_resid) << ","
                << csvList(integrity.labels) << "\n";
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
    // UWB CALLBACK — const4 CSV/rosbag-specialized cycle accumulator
    //
    // The const4 CSV stream can contain reciprocal or repeated unordered pairs
    // within the same physical epoch, e.g. 4-0 followed by 0-4.  The generic
    // node treats an unordered-pair repeat as a cycle boundary; that is useful
    // for some live UWB schedules, but it prematurely splits this rosbag.  This
    // specialized node forms epochs by timestamp window by default.  A count
    // close can be re-enabled by parameter for old structured sequences.
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
        processCurrentCycleIfReady();
        current_cycle_.clear();
              }
      current_cycle_.push_back(ts);
      if ((close_epoch_on_count_ &&
           (int)current_cycle_.size() >= measurements_per_cycle_) ||
          (int)current_cycle_.size() >= max_epoch_measurements_)
      {
        processUnstructuredEpoch(current_cycle_);
        current_cycle_.clear();
              }
    }

    void processCurrentCycleIfReady()
    {
      if ((int)current_cycle_.size() >= min_cycle_measurements_)
      {
        processUnstructuredEpoch(current_cycle_);
      }
      else if (!current_cycle_.empty())
      {
        ROS_WARN_THROTTLE(1.0,
                          "[uwb_imu_fusion] Dropping short UWB cycle: %zu measurements < min=%d",
                          current_cycle_.size(), min_cycle_measurements_);
      }
    }

    TdoaMeasurements compressEpochPairs(
        const TdoaMeasurements &meas) const
    {
      struct Accum
      {
        double t_sum{0.0};
        double tdoa_sum{0.0};
        int count{0};
        int idA{-1};
        int idB{-1};
      };

      std::map<std::pair<int, int>, Accum> acc;
      for (const auto &m : meas)
      {
        const int a = std::min(m.idA, m.idB);
        const int b = std::max(m.idA, m.idB);
        const double canonical_tdoa = (m.idA == a && m.idB == b)
                                          ? m.tdoa
                                          : -m.tdoa;
        auto &slot = acc[std::make_pair(a, b)];
        slot.t_sum += m.t;
        slot.tdoa_sum += canonical_tdoa;
        ++slot.count;
        slot.idA = a;
        slot.idB = b;
      }

      TdoaMeasurements out;
      out.reserve(acc.size());
      for (const auto &kv : acc)
      {
        const auto &slot = kv.second;
        if (slot.count <= 0)
          continue;
        out.push_back(TdoaMeas{
            slot.t_sum / static_cast<double>(slot.count),
            slot.idA,
            slot.idB,
            slot.tdoa_sum / static_cast<double>(slot.count)});
      }
      return out;
    }

    void processUnstructuredEpoch(const TdoaMeasurements &raw_meas)
    {
      if (!compress_epoch_pairs_)
      {
        processCycle(raw_meas);
        return;
      }

      const TdoaMeasurements compressed = compressEpochPairs(raw_meas);
      if (static_cast<int>(compressed.size()) < min_cycle_measurements_)
      {
        ROS_WARN_THROTTLE(
            1.0,
            "[uwb_tdoa_imu_tc_integrity] Dropping compressed UWB epoch: raw=%zu unique_pairs=%zu < min=%d",
            raw_meas.size(), compressed.size(), min_cycle_measurements_);
        return;
      }

      ROS_DEBUG("[uwb_tdoa_imu_tc_integrity] compressed epoch raw=%zu unique_pairs=%zu",
                raw_meas.size(), compressed.size());
      processCycle(compressed);
    }

    // ===========================================================================
    // GT CALLBACK
    // ===========================================================================
    void gtCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &msg)
    {
      std::lock_guard<std::mutex> lk(mutex_);
      const double t = msg->header.stamp.toSec();
      gt_pos_ = gtsam::Point3(msg->pose.pose.position.x,
                              msg->pose.pose.position.y,
                              msg->pose.pose.position.z);
      gt_have_ = true;
      if (gt_buf_.empty() || t > gt_buf_.back().t)
      {
        gt_buf_.push_back(GtMeas{t, gt_pos_});
      }
      else
      {
        ROS_WARN_THROTTLE(1.0, "[uwb_imu_fusion] GT out-of-order %.6f", t);
      }
      const double keep_after = (tc_.initialized ? tc_.last_cycle_t : t) - 5.0;
      while (gt_buf_.size() > 2 && gt_buf_.front().t < keep_after)
        gt_buf_.pop_front();

      geometry_msgs::PoseStamped ps;
      ps.header = msg->header;
      ps.header.frame_id = odom_frame_;
      ps.pose = msg->pose.pose;
      gt_path_.header.stamp = msg->header.stamp;
      gt_path_.header.frame_id = odom_frame_;
      gt_path_.poses.push_back(ps);
      gt_path_pub_.publish(gt_path_);
    }

    bool getGroundTruthAt(double t, gtsam::Point3 &gt) const
    {
      if (gt_buf_.empty())
        return false;

      if (gt_buf_.size() == 1)
      {
        if (std::fabs(gt_buf_.front().t - t) <= gt_match_max_dt_)
        {
          gt = gt_buf_.front().p;
          return true;
        }
        return false;
      }

      for (size_t i = 1; i < gt_buf_.size(); ++i)
      {
        const auto &a = gt_buf_[i - 1];
        const auto &b = gt_buf_[i];
        if (t < a.t)
          break;
        if (t <= b.t)
        {
          const double dt = b.t - a.t;
          if (dt <= 1e-9)
          {
            gt = b.p;
            return true;
          }
          const double u = (t - a.t) / dt;
          gt = gtsam::Point3(a.p.x() + u * (b.p.x() - a.p.x()),
                             a.p.y() + u * (b.p.y() - a.p.y()),
                             a.p.z() + u * (b.p.z() - a.p.z()));
          return true;
        }
      }

      const auto &front = gt_buf_.front();
      const auto &back = gt_buf_.back();
      const double front_dt = std::fabs(t - front.t);
      const double back_dt = std::fabs(t - back.t);
      if (std::min(front_dt, back_dt) <= gt_match_max_dt_)
      {
        gt = (front_dt <= back_dt) ? front.p : back.p;
        return true;
      }
      return false;
    }

    TcIntegritySelection applyTcIntegrityMonitoring(
        double t_mid,
        const TdoaMeasurements &meas,
        const Eigen::VectorXd &base_sigmas,
        bool wls_ok,
        const Eigen::Vector3d &wls_p) const
    {
      TcIntegritySelection out;
      out.meas = meas;
      out.sigmas = base_sigmas;
      out.std_resid.assign(meas.size(), 0.0);
      out.labels.assign(meas.size(), "nominal");
      out.input_count = static_cast<int>(meas.size());
      out.input_pairs.reserve(meas.size());
      out.factor_pairs.reserve(meas.size());
      out.factor_sigmas.reserve(meas.size());
      for (size_t i = 0; i < meas.size(); ++i)
      {
        const std::string pair_name = toPairString(meas[i].idA, meas[i].idB);
        out.input_pairs.push_back(pair_name);
        out.factor_pairs.push_back(pair_name);
        out.factor_sigmas.push_back(
            base_sigmas.size() == static_cast<int>(meas.size())
                ? base_sigmas(static_cast<int>(i))
                : 0.0);
      }

      if (!tc_integrity_enabled_ || !wls_ok ||
          static_cast<int>(meas.size()) < tc_integrity_min_factors_)
        return out;

      out.result = tc_integrity_monitor_.check(meas, anchors_, wls_p, base_sigmas);
      out.used_monitor = true;
      logTcRingClosureCheck(t_mid, "tc-factor-selection", out.result);

      if (out.result.std_resid.size() == meas.size())
      {
        for (size_t i = 0; i < meas.size(); ++i)
        {
          const double std_abs = std::fabs(out.result.std_resid[i]);
          out.std_resid[i] = out.result.std_resid[i];
          if (std_abs > tc_integrity_std_resid_soft_gate_)
          {
            const double scale =
                std::min(tc_integrity_sigma_inflation_,
                         std::max(1.0, std_abs / tc_integrity_std_resid_soft_gate_));
            if (out.sigmas.size() == static_cast<int>(meas.size()))
              out.sigmas(static_cast<int>(i)) *= scale;
            out.labels[i] = "inflated";
            if (i < out.factor_sigmas.size() &&
                out.sigmas.size() == static_cast<int>(meas.size()))
              out.factor_sigmas[i] = out.sigmas(static_cast<int>(i));
          }
        }
      }

      if (!out.result.available && out.sigmas.size() == static_cast<int>(meas.size()))
      {
        const double h_scale =
            std::isfinite(out.result.hpl) && tc_integrity_monitor_.hal > 1e-9
                ? out.result.hpl / tc_integrity_monitor_.hal
                : 1.0;
        const double v_scale =
            std::isfinite(out.result.vpl) && tc_integrity_monitor_.val > 1e-9
                ? out.result.vpl / tc_integrity_monitor_.val
                : 1.0;
        const double scale = std::min(
            tc_integrity_sigma_inflation_,
            std::max(tc_integrity_unavailable_sigma_scale_,
                     std::max(h_scale, v_scale)));

        for (size_t i = 0; i < meas.size(); ++i)
        {
          out.sigmas(static_cast<int>(i)) *= scale;
          out.labels[i] =
              (out.labels[i] == "nominal") ? "unavailable" : out.labels[i] + "+unavailable";
          if (i < out.factor_sigmas.size())
            out.factor_sigmas[i] = out.sigmas(static_cast<int>(i));
        }

        ROS_WARN_THROTTLE(
            0.5,
            "[TC-Integrity] t=%.3f unavailable: HPL/VPL=%.3f/%.3f limits=%.3f/%.3f, inflating TC TDOA sigmas x%.2f",
            t_mid, out.result.hpl, out.result.vpl,
            tc_integrity_monitor_.hal, tc_integrity_monitor_.val, scale);
      }

      std::set<int> drop;
      if (tc_integrity_exclude_fault_ &&
          out.result.excluded_idx >= 0 &&
          static_cast<int>(meas.size()) - 1 >= tc_integrity_min_factors_)
      {
        drop.insert(out.result.excluded_idx);
      }

      if (out.result.std_resid.size() == meas.size() &&
          tc_integrity_std_resid_hard_gate_ > 0.0)
      {
        for (int i = 0; i < static_cast<int>(meas.size()); ++i)
        {
          if (static_cast<int>(meas.size()) - static_cast<int>(drop.size()) <=
              tc_integrity_min_factors_)
            break;
          if (std::fabs(out.result.std_resid[i]) > tc_integrity_std_resid_hard_gate_)
            drop.insert(i);
        }
      }

      if (!drop.empty())
      {
        TdoaMeasurements filtered;
        filtered.reserve(meas.size() - drop.size());
        Eigen::VectorXd filtered_sigmas(static_cast<int>(meas.size() - drop.size()));
        std::vector<double> filtered_std;
        std::vector<std::string> filtered_labels;
        std::vector<std::string> filtered_pairs;
        std::vector<double> filtered_factor_sigmas;
        int row = 0;
        std::ostringstream dropped;
        for (int i = 0; i < static_cast<int>(meas.size()); ++i)
        {
          if (drop.count(i))
          {
            if (out.excluded_count++ > 0) dropped << ";";
            dropped << toPairString(meas[i].idA, meas[i].idB);
            out.excluded_pairs.push_back(toPairString(meas[i].idA, meas[i].idB));
            continue;
          }
          filtered.push_back(meas[i]);
          filtered_sigmas(row++) =
              (out.sigmas.size() == static_cast<int>(meas.size()))
                  ? out.sigmas(i)
                  : tdoa_sigma_;
          filtered_std.push_back(out.std_resid[i]);
          filtered_labels.push_back(out.labels[i]);
          filtered_pairs.push_back(toPairString(meas[i].idA, meas[i].idB));
          filtered_factor_sigmas.push_back(filtered_sigmas(row - 1));
        }
        out.meas = std::move(filtered);
        out.sigmas = filtered_sigmas;
        out.std_resid = std::move(filtered_std);
        out.labels = std::move(filtered_labels);
        out.factor_pairs = std::move(filtered_pairs);
        out.factor_sigmas = std::move(filtered_factor_sigmas);
        ROS_WARN_THROTTLE(0.5,
            "[TC-Integrity] t=%.3f excluded %d TDOA factors (%s), chi2=%.2f/%.2f",
            t_mid, out.excluded_count, dropped.str().c_str(),
            out.result.chi2_stat, out.result.chi2_threshold);
      }

      return out;
    }

    static std::string anchorName(int id)
    {
      return std::to_string(id);
    }

    static std::string toPairString(int id_a, int id_b)
    {
      return std::to_string(id_a) + "-" + std::to_string(id_b);
    }

    TdoaMeasurements dropAnchorMeasurements(
        const TdoaMeasurements &meas,
        int anchor_id) const
    {
      TdoaMeasurements kept;
      kept.reserve(meas.size());
      for (const auto &m : meas)
      {
        if (m.idA == anchor_id || m.idB == anchor_id)
          continue;
        kept.push_back(m);
      }
      return kept;
    }

    int incidentMeasurementCount(const TdoaMeasurements &meas,
                                 int anchor_id) const
    {
      int count = 0;
      for (const auto &m : meas)
      {
        if (m.idA == anchor_id || m.idB == anchor_id)
          ++count;
      }
      return count;
    }

    bool solveDynamicWls(const TdoaMeasurements &meas,
                         const Eigen::Vector3d &seed,
                         Eigen::Vector3d &wls_p,
                         Eigen::VectorXd &sigmas)
    {
      wls_p = seed;
      bool ok = wls_solver_.solve(meas, anchors_, wls_p);
      if (!ok)
      {
        sigmas = Eigen::VectorXd();
        return false;
      }

      sigmas = computeDynamicTdoaSigmas(meas, wls_p);
      if (wls_refine_with_dynamic_sigma_ &&
          sigmas.size() == static_cast<int>(meas.size()))
      {
        Eigen::Vector3d weighted_wls_p = wls_p;
        if (wls_solver_.solveWeighted(meas, anchors_, sigmas, weighted_wls_p))
        {
          wls_p = weighted_wls_p;
          sigmas = computeDynamicTdoaSigmas(meas, wls_p);
        }
      }
      return true;
    }

    double integrityScore(const IntegrityResult &r) const
    {
      const double h_ratio =
          std::isfinite(r.hpl) && tc_integrity_monitor_.hal > 1e-9
              ? r.hpl / tc_integrity_monitor_.hal
              : 10.0;
      const double v_ratio =
          std::isfinite(r.vpl) && tc_integrity_monitor_.val > 1e-9
              ? r.vpl / tc_integrity_monitor_.val
              : 10.0;
      const double chi_ratio =
          (r.chi2_threshold > 1e-9)
              ? r.chi2_stat / r.chi2_threshold
              : 10.0;
      return std::max(std::max(h_ratio, v_ratio), chi_ratio);
    }

    bool cleanCandidateImproves(const IntegrityResult &all_result,
                                const IntegrityResult &candidate_result) const
    {
      if (candidate_result.available && !all_result.available)
        return true;

      const double all_score = integrityScore(all_result);
      const double candidate_score = integrityScore(candidate_result);
      if (candidate_score < all_score * tc_nlos_anchor_hpl_improvement_ratio_)
        return true;

      if (all_result.chi2_stat > 1e-9 &&
          candidate_result.chi2_stat <
              all_result.chi2_stat * tc_nlos_anchor_chi2_improvement_ratio_)
        return true;

      return false;
    }

    CleanLosSelection selectCleanLosMeasurements(
        double t_mid,
        const TdoaMeasurements &meas,
        bool all_wls_ok,
        const Eigen::Vector3d &all_wls_p,
        const Eigen::VectorXd &all_sigmas)
    {
      CleanLosSelection out;
      out.meas = meas;
      out.sigmas = all_sigmas;
      out.wls_p = all_wls_p;
      out.wls_ok = all_wls_ok;

      if (!tc_nlos_anchor_rejection_enabled_ || !all_wls_ok ||
          static_cast<int>(meas.size()) < tc_nlos_anchor_min_remaining_measurements_)
      {
        return out;
      }

      out.all_result =
          tc_integrity_monitor_.check(meas, anchors_, all_wls_p, all_sigmas);
      logTcRingClosureCheck(t_mid, "nlos-all", out.all_result);
      if (out.all_result.available &&
          !out.all_result.fault_detected &&
          out.all_result.ring_ok)
      {
        return out;
      }

      double best_score = integrityScore(out.all_result);
      int best_anchor = -1;
      TdoaMeasurements best_meas;
      Eigen::VectorXd best_sigmas;
      Eigen::Vector3d best_wls_p = all_wls_p;
      IntegrityResult best_result;

      for (int anchor_id = 0; anchor_id < static_cast<int>(anchors_.size()); ++anchor_id)
      {
        const int incident = incidentMeasurementCount(meas, anchor_id);
        if (incident < tc_nlos_anchor_min_incident_measurements_)
          continue;

        TdoaMeasurements candidate_meas =
            dropAnchorMeasurements(meas, anchor_id);
        if (static_cast<int>(candidate_meas.size()) <
            tc_nlos_anchor_min_remaining_measurements_)
          continue;

        Eigen::Vector3d candidate_wls_p;
        Eigen::VectorXd candidate_sigmas;
        if (!solveDynamicWls(candidate_meas, all_wls_p,
                             candidate_wls_p, candidate_sigmas))
          continue;

        if ((candidate_wls_p - all_wls_p).norm() > tc_nlos_anchor_max_clean_wls_shift_)
          continue;

        IntegrityResult candidate_result =
            tc_integrity_monitor_.check(candidate_meas, anchors_,
                                        candidate_wls_p, candidate_sigmas);
        logTcRingClosureCheck(
            t_mid, std::string("nlos-drop-anchor-") + std::to_string(anchor_id),
            candidate_result);
        if (!cleanCandidateImproves(out.all_result, candidate_result))
          continue;

        const double candidate_score = integrityScore(candidate_result);
        if (candidate_score < best_score ||
            (candidate_result.available && !best_result.available))
        {
          best_score = candidate_score;
          best_anchor = anchor_id;
          best_meas = std::move(candidate_meas);
          best_sigmas = candidate_sigmas;
          best_wls_p = candidate_wls_p;
          best_result = candidate_result;
        }
      }

      if (best_anchor >= 0)
      {
        out.meas = std::move(best_meas);
        out.sigmas = best_sigmas;
        out.wls_p = best_wls_p;
        out.wls_ok = true;
        out.used_clean = true;
        out.nlos_anchor_id = best_anchor;
        out.excluded_anchors.push_back(anchorName(best_anchor));
        out.clean_result = best_result;
        ROS_WARN_THROTTLE(
            0.5,
            "[TC-NLOS] t=%.3f anchor %d removed: meas %zu→%zu, score %.2f→%.2f, HPL/VPL %.3f/%.3f→%.3f/%.3f",
            t_mid, best_anchor, meas.size(), out.meas.size(),
            integrityScore(out.all_result), best_score,
            out.all_result.hpl, out.all_result.vpl,
            out.clean_result.hpl, out.clean_result.vpl);
      }

      return out;
    }

    // ===========================================================================
    // PROCESS CYCLE
    // ===========================================================================
    void processCycle(const TdoaMeasurements &meas)
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
      Eigen::VectorXd tdoa_sigmas;
      bool wls_ok = solveDynamicWls(meas, wls_guess_, wls_p, tdoa_sigmas);
      if (wls_ok)
        wls_guess_ = wls_p;
      ROS_WARN("[WLS Solution] t=%.3f  x=%.3f  y=%.3f  z=%.3f",
               t_mid, wls_p.x(), wls_p.y(), wls_p.z());

      // ── Clean LOS subset: all-anchor WLS is logged, clean WLS drives TC ─────
      const CleanLosSelection clean =
          selectCleanLosMeasurements(t_mid, meas, wls_ok, wls_p, tdoa_sigmas);
      const bool clean_wls_ok = clean.wls_ok;
      const Eigen::Vector3d clean_wls_p = clean.wls_p;

      if (clean.used_clean)
      {
        wls_guess_ = clean_wls_p;
        ROS_WARN_THROTTLE(
            0.5,
            "[WLS Clean LOS] t=%.3f  all=[%.3f,%.3f,%.3f] clean=[%.3f,%.3f,%.3f] removed_anchor=%d",
            t_mid, wls_p.x(), wls_p.y(), wls_p.z(),
            clean_wls_p.x(), clean_wls_p.y(), clean_wls_p.z(),
            clean.nlos_anchor_id);
      }

      // ── TC: clean LOS TDOA + IMU ───────────────────────────────────────────
      const TdoaMeasurements tc_meas =
          selectTcTdoaMeasurements(clean.meas, clean_wls_ok, clean_wls_p);
      const Eigen::VectorXd tc_tdoa_sigmas =
          clean_wls_ok ? computeDynamicTdoaSigmas(tc_meas, clean_wls_p)
                       : Eigen::VectorXd();
      const TcIntegritySelection tc_integrity =
          applyTcIntegrityMonitoring(t_mid, tc_meas, tc_tdoa_sigmas,
                                     clean_wls_ok, clean_wls_p);
      latest_tc_integrity_std_resid_ = tc_integrity.std_resid;
      latest_tc_integrity_labels_ = tc_integrity.labels;
      const Eigen::Vector3d tc_wls_p =
          filterTcWlsPositionForTc(t_mid, tcCorrectedWlsPosition(clean_wls_p));
      const bool tc_wls_ok =
          clean_wls_ok &&
          (!tc_integrity_gate_wls_on_unavailable_ ||
           !tc_integrity.used_monitor ||
           tc_integrity.result.available);

      if (clean_wls_ok && !tc_wls_ok)
      {
        ROS_WARN_THROTTLE(
            0.5,
            "[TC-Integrity] gating WLS-derived TC priors: available=%s HPL=%.3f VPL=%.3f input=%d factors=%zu",
            tc_integrity.result.available ? "true" : "false",
            tc_integrity.result.hpl, tc_integrity.result.vpl,
            tc_integrity.input_count, tc_integrity.meas.size());
      }

      if (!tc_.initialized)
      {
        tcInitGraph(t_mid, tc_integrity.meas, tc_integrity.sigmas, tc_wls_ok, tc_wls_p);
      }
      else
      {
        tcUpdateGraph(t_mid, tc_integrity.meas, tc_integrity.sigmas, tc_wls_ok, tc_wls_p);
      }
      rememberWlsSample(t_mid, wls_ok, wls_p);

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

      logRow(t_mid, wls_ok, wls_p,
             clean_wls_ok, clean_wls_p,
             imu_stats, clean, tc_integrity);
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

    double tcVerticalReference(double wls_z, double fallback_z) const
    {
      if (tc_vertical_prior_source_ == "wls")
        return wls_z;
      if (tc_vertical_prior_source_ == "fallback" ||
          tc_vertical_prior_source_ == "predicted")
        return fallback_z;
      return tc_vertical_prior_z_ref_;
    }

    Eigen::Vector3d tcCorrectedWlsPosition(const Eigen::Vector3d &wls_p) const
    {
      Eigen::Vector3d corrected = wls_p;
      corrected.z() += tc_wls_z_bias_correction_;
      return corrected;
    }

    Eigen::Vector3d filterTcWlsPositionForTc(double t,
                                             const Eigen::Vector3d &wls_p)
    {
      if (!tc_wls_z_filter_enabled_ || !std::isfinite(wls_p.z()))
        return wls_p;

      Eigen::Vector3d filtered = wls_p;
      if (!have_tc_wls_z_filter_)
      {
        tc_wls_z_filter_z_ = wls_p.z();
        tc_wls_z_filter_t_ = t;
        have_tc_wls_z_filter_ = true;
        return filtered;
      }

      const double dz = wls_p.z() - tc_wls_z_filter_z_;
      const bool outlier = std::fabs(dz) > tc_wls_z_filter_outlier_gate_;
      const double alpha = std::min(
          1.0, std::max(0.0, outlier ? tc_wls_z_filter_outlier_alpha_
                                      : tc_wls_z_filter_alpha_));
      tc_wls_z_filter_z_ += alpha * dz;
      tc_wls_z_filter_t_ = t;
      filtered.z() = tc_wls_z_filter_z_;

      if (outlier)
      {
        ROS_WARN_THROTTLE(
            0.5,
            "[TC-WLS-Z] smoothing vertical WLS prior: raw_z=%.3f filtered_z=%.3f dz=%.3f alpha=%.2f",
            wls_p.z(), filtered.z(), dz, alpha);
      }
      return filtered;
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

    void rememberWlsSample(double t_mid, bool wls_ok, const Eigen::Vector3d &wls_p)
    {
      if (!wls_ok)
        return;
      last_wls_p_ = wls_p;
      last_wls_t_ = t_mid;
      have_last_wls_ = true;
    }

    bool computeWlsVelocityReference(double t_mid,
                                     const Eigen::Vector3d &wls_p,
                                     gtsam::Vector3 &vel_ref) const
    {
      if (!tc_use_wls_velocity_prior_ || !have_last_wls_)
        return false;

      const double dt = t_mid - last_wls_t_;
      if (dt <= 1e-3 || dt > tc_wls_velocity_max_dt_)
        return false;

      Eigen::Vector3d v = (wls_p - last_wls_p_) / dt;
      if (tc_use_wls_z_velocity_prior_)
      {
        v.z() = std::min(std::max(v.z(), -tc_wls_z_velocity_max_speed_),
                         tc_wls_z_velocity_max_speed_);
      }
      else
      {
        v.z() = 0.0;
      }
      const double speed = v.norm();
      if (speed > tc_wls_velocity_max_speed_)
        v *= tc_wls_velocity_max_speed_ / speed;

      vel_ref = gtsam::Vector3(v.x(), v.y(), v.z());
      return true;
    }

    double tdoaResidualAt(const TdoaMeas &m, const Eigen::Vector3d &p) const
    {
      const Eigen::Vector3d aA(anchors_[m.idA].x(),
                               anchors_[m.idA].y(),
                               anchors_[m.idA].z());
      const Eigen::Vector3d aB(anchors_[m.idB].x(),
                               anchors_[m.idB].y(),
                               anchors_[m.idB].z());
      const double dA = (p - aA).norm();
      const double dB = (p - aB).norm();
      return (dB - dA) - m.tdoa;
    }

    double dynamicTdoaSigmaFromResidual(double abs_residual) const
    {
      if (!use_dynamic_tdoa_sigma_)
        return tdoa_sigma_;
      const double sigma = tdoa_sigma_min_ + tdoa_residual_sigma_scale_ * abs_residual;
      return std::min(std::max(sigma, tdoa_sigma_min_), tdoa_sigma_max_);
    }

    Eigen::VectorXd computeDynamicTdoaSigmas(const TdoaMeasurements &meas,
                                             const Eigen::Vector3d &wls_p) const
    {
      Eigen::VectorXd sigmas(meas.size());
      for (size_t i = 0; i < meas.size(); ++i)
      {
        sigmas(static_cast<int>(i)) =
            dynamicTdoaSigmaFromResidual(std::fabs(tdoaResidualAt(meas[i], wls_p)));
      }
      return sigmas;
    }

    gtsam::SharedNoiseModel makeTdoaNoise(double sigma) const
    {
      const double bounded_sigma = std::min(std::max(sigma, tdoa_sigma_min_), tdoa_sigma_max_);
      gtsam::SharedNoiseModel base =
          gtsam::noiseModel::Isotropic::Sigma(1, bounded_sigma);
      if (use_robust_noise_)
      {
        return gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(huber_k_), base);
      }
      return base;
    }

    void addTcTdoaFactors(gtsam::NonlinearFactorGraph &graph,
                          size_t k,
                          const TdoaMeasurements &meas,
                          const Eigen::VectorXd &sigmas) const
    {
      for (size_t i = 0; i < meas.size(); ++i)
      {
        const auto &m = meas[i];
        const double sigma =
            (sigmas.size() == static_cast<int>(meas.size()))
                ? sigmas(static_cast<int>(i))
                : tdoa_sigma_;
        const double std_resid =
            (i < latest_tc_integrity_std_resid_.size())
                ? latest_tc_integrity_std_resid_[i]
                : 0.0;
        const std::string label =
            (i < latest_tc_integrity_labels_.size())
                ? latest_tc_integrity_labels_[i]
                : std::string("nominal");
        graph.add(IntegrityTdoaFactor(X(k), anchors_[m.idA], anchors_[m.idB],
                                     -m.tdoa, makeTdoaNoise(sigma), m.idA,
                                     m.idB, sigma, std_resid, label));
      }
    }

    TdoaMeasurements selectTcTdoaMeasurements(const TdoaMeasurements &meas,
                                             bool wls_ok,
                                             const Eigen::Vector3d &wls_p) const
    {
      if (!tc_use_tdoa_residual_gate_ || !wls_ok ||
          tc_tdoa_residual_gate_ <= 0.0 ||
          static_cast<int>(meas.size()) <= tc_min_tdoa_factors_)
      {
        return meas;
      }

      TdoaMeasurements kept;
      kept.reserve(meas.size());
      double max_abs_residual = 0.0;
      for (const auto &m : meas)
      {
        const double abs_r = std::fabs(tdoaResidualAt(m, wls_p));
        max_abs_residual = std::max(max_abs_residual, abs_r);
        if (abs_r <= tc_tdoa_residual_gate_)
          kept.push_back(m);
      }

      if (static_cast<int>(kept.size()) < tc_min_tdoa_factors_)
      {
        ROS_WARN_THROTTLE(0.5,
                          "[TC-FGO] TDOA residual gate skipped: would keep %zu/%zu (< min=%d), max|r|=%.3f m",
                          kept.size(), meas.size(), tc_min_tdoa_factors_, max_abs_residual);
        return meas;
      }

      if (kept.size() != meas.size())
      {
        ROS_WARN_THROTTLE(0.5,
                          "[TC-FGO] TDOA residual gate kept %zu/%zu measurements (gate=%.3f m, max|r|=%.3f m)",
                          kept.size(), meas.size(), tc_tdoa_residual_gate_, max_abs_residual);
      }
      return kept;
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
    void tcDropoutUpdate(double t_mid, const TdoaMeasurements &meas,
                         const Eigen::VectorXd &tdoa_sigmas,
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

      // ── Velocity: prefer WLS XY displacement over stale TC XY velocity.
      gtsam::Vector3 dropout_vel_ref = tc_.vel;
      gtsam::Vector3 dropout_vel_sig = gtsam::Vector3::Constant(tc_dropout_vel_sigma_);
      if (wls_gated)
      {
        gtsam::Vector3 wls_vel_ref;
        if (computeWlsVelocityReference(t_mid, wls_p, wls_vel_ref))
        {
          dropout_vel_ref.x() = wls_vel_ref.x();
          dropout_vel_ref.y() = wls_vel_ref.y();
          if (tc_use_wls_z_velocity_prior_)
          {
            dropout_vel_ref.z() = wls_vel_ref.z();
            dropout_vel_sig.z() = tc_wls_z_velocity_sigma_;
          }
          else
          {
            dropout_vel_sig.z() = tc_update_vel_sigma_;
          }
        }
      }
      graph.addPrior(V(k), dropout_vel_ref,
                     gtsam::noiseModel::Diagonal::Sigmas(dropout_vel_sig));

      // ── Bias: carry forward — no IMU means bias cannot update, just hold ─────
      graph.add(tcBiasBetween(k_pre, k, t_mid - tc_.last_cycle_t));
      graph.addPrior(B(k), tc_.bias,
                     gtsam::noiseModel::Isotropic::Sigma(6, tc_update_bias_sigma_));

      // ── Vertical prior ────────────────────────────────────────────────────────
      const gtsam::Point3 &pos_hint = tc_.pose.translation();
      if (wls_ok)
        addTcVerticalPrior(graph, k, pos_hint,
                           tcVerticalReference(wls_p.z(), pos_hint.z()));
      else
        addTcVerticalPrior(graph, k, pos_hint,
                           tcVerticalReference(pos_hint.z(), pos_hint.z()));

      // ── TDOA factors ──────────────────────────────────────────────────────────
      addTcTdoaFactors(graph, k, meas, tdoa_sigmas);

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

    void tcInitGraph(double t_mid, const TdoaMeasurements &meas,
                     const Eigen::VectorXd &tdoa_sigmas,
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
        addTcVerticalPrior(graph, 0, tc_.pose.translation(),
                           tcVerticalReference(wls_p.z(), tc_.pose.z()));

      addTcTdoaFactors(graph, 0, meas, tdoa_sigmas);

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
    void tcUpdateGraph(double t_mid, const TdoaMeasurements &meas,
                       const Eigen::VectorXd &tdoa_sigmas,
                       bool wls_ok, const Eigen::Vector3d &wls_p)
    {
      tc_preint_->resetIntegrationAndSetBias(tc_.bias);
      int n_imu = replayImu(tc_.last_cycle_t, t_mid, *tc_preint_);

      if (n_imu < min_imu_per_cycle_)
      {
        // IMU dropout — route to dedicated dropout handler instead of skipping.
        // Skipping leaves velocity frozen and position unanchored, which causes
        // phantom drift of speed×gap_duration (observed: up to 0.83 m per gap).
        tcDropoutUpdate(t_mid, meas, tdoa_sigmas, wls_ok, wls_p);
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

      // Apply WLS innovation gate before adding WLS-derived priors.
      // When TC has drifted far from WLS (e.g. after a dropout), adding tight
      // WLS constraints against a badly linearised point can amplify the error
      // rather than correcting it.
      const bool wls_gated = wls_ok && wlsPassesInnovationGate(wls_p);

      gtsam::Vector3 vel_prior_mean = pred_nav.velocity();
      double vel_prior_sigma = tc_update_vel_sigma_;
      bool using_wls_vel_prior = false;

      gtsam::Vector3 wls_vel_ref;
      if (wls_gated && computeWlsVelocityReference(t_mid, wls_p, wls_vel_ref))
      {
        vel_prior_mean.x() = wls_vel_ref.x();
        vel_prior_mean.y() = wls_vel_ref.y();
        if (tc_use_wls_z_velocity_prior_)
          vel_prior_mean.z() = wls_vel_ref.z();
        vel_prior_sigma = tc_wls_velocity_sigma_;
        using_wls_vel_prior = true;
      }

      gtsam::Vector3 vel_prior_sig;
      vel_prior_sig << vel_prior_sigma, vel_prior_sigma,
          tc_use_wls_z_velocity_prior_ ? tc_wls_z_velocity_sigma_
                                       : tc_update_vel_sigma_;
      graph.addPrior(V(k), vel_prior_mean,
                     gtsam::noiseModel::Diagonal::Sigmas(vel_prior_sig));
      if (using_wls_vel_prior)
      {
        ROS_WARN_THROTTLE(0.5,
                          "[TC-FGO] WLS velocity prior active: v=[%.3f,%.3f,%.3f] sigma_xy=%.3f sigma_z=%.3f",
                          vel_prior_mean.x(), vel_prior_mean.y(), vel_prior_mean.z(),
                          vel_prior_sigma, vel_prior_sig.z());
      }
      graph.addPrior(B(k), tc_.bias,
                     gtsam::noiseModel::Isotropic::Sigma(6, tc_update_bias_sigma_));

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
        addTcVerticalPrior(graph, k, pred_nav.pose().translation(),
                           tcVerticalReference(wls_p.z(), pred_nav.pose().z()));
      }
      else
      {
        addTcVerticalPrior(graph, k, pred_nav.pose().translation(),
                           tcVerticalReference(pred_nav.pose().z(), pred_nav.pose().z()));
      }
      addTcAttitudePrior(graph, k, pred_nav.pose().translation());

      addTcTdoaFactors(graph, k, meas, tdoa_sigmas);

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
      if (!anchors_.empty())
      {
        double min_z = anchors_.front().z();
        double max_z = anchors_.front().z();
        for (const auto &a : anchors_)
        {
          min_z = std::min(min_z, a.z());
          max_z = std::max(max_z, a.z());
        }
        ROS_INFO("[uwb_imu_fusion] anchor extrinsic t=[%.4f, %.4f, %.4f]  transformed anchor z range=[%.3f, %.3f]",
                 anchor_extrinsic_translation_.x(),
                 anchor_extrinsic_translation_.y(),
                 anchor_extrinsic_translation_.z(),
                 min_z, max_z);
      }
      ROS_INFO("[uwb_imu_fusion] UWB cycle: min_measurements=%d timeout=%.3f s",
               min_cycle_measurements_, cycle_timeout_);
      ROS_INFO("[uwb_imu_fusion] specialized epoching: close_on_count=%s count=%d max_epoch_measurements=%d",
               close_epoch_on_count_ ? "true" : "false",
               measurements_per_cycle_, max_epoch_measurements_);
      ROS_INFO("[uwb_imu_fusion] specialized epoch compression: compress_epoch_pairs=%s",
               compress_epoch_pairs_ ? "true" : "false");
      ROS_INFO("[uwb_imu_fusion] gravity=%.3f  tdoa_sigma=%.3f",
               gravity_mag_, tdoa_sigma_);
      ROS_INFO("[uwb_imu_fusion] dynamic TDOA sigma: %s  min=%.3f max=%.3f residual_scale=%.3f wls_refine=%s",
               use_dynamic_tdoa_sigma_ ? "enabled" : "disabled",
               tdoa_sigma_min_, tdoa_sigma_max_, tdoa_residual_sigma_scale_,
               wls_refine_with_dynamic_sigma_ ? "enabled" : "disabled");
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
      ROS_INFO("[uwb_imu_fusion] WLS velocity prior: %s  sigma=%.3f max_speed=%.3f max_dt=%.3f",
               tc_use_wls_velocity_prior_ ? "enabled" : "disabled",
               tc_wls_velocity_sigma_, tc_wls_velocity_max_speed_, tc_wls_velocity_max_dt_);
      ROS_INFO("[uwb_imu_fusion] WLS Z velocity prior: %s  sigma=%.3f max_z_speed=%.3f",
               tc_use_wls_z_velocity_prior_ ? "enabled" : "disabled",
               tc_wls_z_velocity_sigma_, tc_wls_z_velocity_max_speed_);
      ROS_INFO("[uwb_imu_fusion] imu leveling: %s  samples=%d",
               estimate_initial_orientation_from_imu_ ? "enabled" : "disabled",
               initial_alignment_imu_samples_);
      ROS_INFO("[uwb_imu_fusion] GT comparison: interpolated at output timestamp, nearest max_dt=%.3f s",
               gt_match_max_dt_);
      ROS_INFO("[uwb_imu_fusion] tc vertical prior: %s  sigma=%.3f source=%s z_ref=%.3f",
               tc_use_vertical_prior_ ? "enabled" : "disabled",
               tc_vertical_prior_sigma_,
               tc_vertical_prior_source_.c_str(),
               tc_vertical_prior_z_ref_);
      ROS_INFO("[uwb_imu_fusion] tc WLS position prior: %s  xy_sigma=%.3f z_sigma=%.3f",
               tc_use_wls_position_prior_ ? "enabled" : "disabled",
               tc_wls_xy_sigma_, tc_wls_z_sigma_);
      ROS_INFO("[uwb_imu_fusion] tc WLS Z bias correction: %.3f m",
               tc_wls_z_bias_correction_);
      ROS_INFO("[uwb_imu_fusion] tc WLS Z filter: %s alpha=%.2f outlier_gate=%.3f outlier_alpha=%.2f",
               tc_wls_z_filter_enabled_ ? "enabled" : "disabled",
               tc_wls_z_filter_alpha_, tc_wls_z_filter_outlier_gate_,
               tc_wls_z_filter_outlier_alpha_);
      ROS_INFO("[uwb_imu_fusion] tc WLS rescue: gate=%.3f xy_sigma=%.3f z_sigma=%.3f",
               tc_wls_rescue_gate_, tc_wls_rescue_xy_sigma_, tc_wls_rescue_z_sigma_);
      ROS_INFO("[uwb_imu_fusion] tc TDOA residual gate: %s  gate=%.3f min_factors=%d",
               tc_use_tdoa_residual_gate_ ? "enabled" : "disabled",
               tc_tdoa_residual_gate_, tc_min_tdoa_factors_);
      ROS_INFO("[uwb_imu_fusion] tc attitude prior: %s  roll_pitch_sigma=%.3f yaw_sigma=%.1e",
               tc_use_attitude_prior_ ? "enabled" : "disabled",
               tc_attitude_roll_pitch_sigma_, tc_attitude_yaw_sigma_);
    }

    void printCycleInfo(double t_mid, bool wls_ok,
                        const Eigen::Vector3d &wls_p) const
    {
      gtsam::Point3 gt_at_t;
      const bool gt_at_t_ok = getGroundTruthAt(t_mid, gt_at_t);
      auto err3 = [&](const gtsam::Point3 &p) -> double
      {
        return gt_at_t_ok ? std::sqrt(std::pow(p.x() - gt_at_t.x(), 2) +
                                      std::pow(p.y() - gt_at_t.y(), 2) +
                                      std::pow(p.z() - gt_at_t.z(), 2))
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
      if (gt_at_t_ok)
      {
        ROS_INFO("  GT@t   : [%7.3f,%7.3f,%7.3f]", gt_at_t.x(), gt_at_t.y(), gt_at_t.z());
        if (wls_ok)
          ROS_INFO("  |WLS-GT| = %.4f m",
                   (wls_p - Eigen::Vector3d(gt_at_t.x(), gt_at_t.y(), gt_at_t.z())).norm());
        if (tc_.initialized)
          ROS_INFO("  |TC -GT| = %.4f m", err3(tc_.pose.translation()));
      }
      else
      {
        ROS_INFO("  GT@t   : unavailable at t=%.3f (buffer=%zu, max_dt=%.3f s)",
                 t_mid, gt_buf_.size(), gt_match_max_dt_);
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

    TdoaMeasurements current_cycle_;
    int measurements_per_cycle_{8}, min_cycle_measurements_{6}, min_imu_per_cycle_{2};
    int max_epoch_measurements_{24};
    double cycle_timeout_{0.1};
    bool close_epoch_on_count_{false};
    bool compress_epoch_pairs_{false};

    TdoaWlsSolver wls_solver_;
    Eigen::Vector3d wls_guess_{0, 0, 1};
    Eigen::Vector3d anchor_extrinsic_translation_{0, 0, 0};
    Eigen::Vector3d last_wls_p_{0, 0, 1};
    double last_wls_t_{0.0};
    bool have_last_wls_{false};

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
    double tc_accel_sigma_{0.08};          // [m/s²] inflated to lower IMU weight
    double tc_gyro_sigma_{0.006};          // [rad/s] inflated to lower IMU weight
    double tc_accel_bias_rw_sigma_{0.02};  // [m/s²/√s] allow bias adaptation
    double tc_gyro_bias_rw_sigma_{0.0006}; // [rad/s/√s] allow bias adaptation
    double tc_integration_sigma_{5e-4};    // lower long-window IMU confidence

    // UWB noise — higher weight with residual gating + Huber kernel
    double tdoa_sigma_{0.2}; // [m] lower sigma gives UWB more pull
    bool use_robust_noise_{true};
    double huber_k_{1.345};
    gtsam::SharedNoiseModel tdoa_noise_;
    bool use_dynamic_tdoa_sigma_{true};
    double tdoa_sigma_min_{0.05};
    double tdoa_sigma_max_{0.8};
    double tdoa_residual_sigma_scale_{0.75};
    bool wls_refine_with_dynamic_sigma_{true};
    bool tc_use_tdoa_residual_gate_{true};
    double tc_tdoa_residual_gate_{0.9}; // [m] reject clear per-cycle NLOS spikes
    int tc_min_tdoa_factors_{4};
    bool tc_integrity_enabled_{true};
    bool tc_integrity_exclude_fault_{true};
    double tc_integrity_sigma_inflation_{4.0};
    double tc_integrity_std_resid_soft_gate_{2.5};
    double tc_integrity_std_resid_hard_gate_{5.0};
    bool tc_integrity_gate_wls_on_unavailable_{true};
    double tc_integrity_unavailable_sigma_scale_{3.0};
    int tc_integrity_min_factors_{4};
    bool tc_nlos_anchor_rejection_enabled_{true};
    int tc_nlos_anchor_min_incident_measurements_{2};
    int tc_nlos_anchor_min_remaining_measurements_{5};
    double tc_nlos_anchor_hpl_improvement_ratio_{0.85};
    double tc_nlos_anchor_chi2_improvement_ratio_{0.80};
    double tc_nlos_anchor_max_clean_wls_shift_{2.0};
    TdoaIntegrityMonitor tc_integrity_monitor_;
    std::vector<double> latest_tc_integrity_std_resid_;
    std::vector<std::string> latest_tc_integrity_labels_;

    // TC priors
    gtsam::Vector6 tc_prior_pose_sigmas_;
    double tc_prior_vel_sigma_{0.1};
    double tc_prior_bias_sigma_{0.1}; // FIX: was 0.001 — too tight, pinned bias at zero
    double tc_update_vel_sigma_{0.8}; // looser: reduce IMU prediction authority
    double tc_update_bias_sigma_{0.2};
    bool tc_use_wls_velocity_prior_{true};
    double tc_wls_velocity_sigma_{0.15};    // [m/s] stronger WLS-derived velocity regularizer
    double tc_wls_velocity_max_speed_{1.2}; // [m/s] cap noisy WLS finite differences
    double tc_wls_velocity_max_dt_{0.25};   // [s] reject stale WLS velocity pairs
    bool tc_use_wls_z_velocity_prior_{true};
    double tc_wls_z_velocity_sigma_{0.25};
    double tc_wls_z_velocity_max_speed_{0.8};
    bool tc_use_vertical_prior_{true};
    double tc_vertical_prior_sigma_{0.15};
    std::string tc_vertical_prior_source_{"wls"};
    double tc_vertical_prior_z_ref_{std::numeric_limits<double>::quiet_NaN()};
    bool tc_wls_z_filter_enabled_{false};
    double tc_wls_z_filter_alpha_{0.65};
    double tc_wls_z_filter_outlier_gate_{0.45};
    double tc_wls_z_filter_outlier_alpha_{0.20};
    bool have_tc_wls_z_filter_{false};
    double tc_wls_z_filter_z_{0.0};
    double tc_wls_z_filter_t_{0.0};
    bool tc_use_wls_position_prior_{true};
    double tc_wls_xy_sigma_{0.15}; // [m] stronger horizontal WLS trust
    double tc_wls_z_sigma_{0.25};  // [m] looser than XY: WLS Z less reliable
    double tc_wls_z_bias_correction_{0.0};
    double tc_wls_rescue_gate_{0.20};     // [m] strengthen WLS prior above this XY gap
    double tc_wls_rescue_xy_sigma_{0.06}; // [m] rescue-mode horizontal WLS prior
    double tc_wls_rescue_z_sigma_{0.18};  // [m] rescue-mode vertical WLS prior
    bool tc_use_attitude_prior_{true};
    double tc_attitude_roll_pitch_sigma_{0.05}; // [rad] strong roll/pitch leveling
    double tc_attitude_yaw_sigma_{1e6};          // [rad] yaw effectively free
    // IMU-dropout fallback parameters
    double tc_wls_dropout_xy_sigma_{0.25}; // [m] WLS prior when no IMU
    double tc_dropout_vel_sigma_{0.2};     // [m/s] lower stale-IMU velocity authority
    double wls_innovation_gate_{0.8};      // [m] max WLS↔TC XY gap before rejecting prior

    gtsam::Pose3 initial_pose_;

    boost::shared_ptr<gtsam::PreintegrationParams> tc_imu_params_;

    // TC branch
    FgoState tc_;
    boost::shared_ptr<gtsam::PreintegratedImuMeasurements> tc_preint_;
    boost::shared_ptr<gtsam::ISAM2> tc_isam_;

    bool gt_have_{false};
    gtsam::Point3 gt_pos_{gtsam::Point3::Zero()};
    std::deque<GtMeas> gt_buf_;
    double gt_match_max_dt_{0.02};

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
  ros::init(argc, argv, "uwb_tdoa_imu_tc_integrity_node");
  ros::NodeHandle nh, pnh("~");
  uwb_imu_fusion::UwbImuFusionNode node(nh, pnh);
  ros::AsyncSpinner spinner(2);
  spinner.start();
  ros::waitForShutdown();
  return 0;
}
