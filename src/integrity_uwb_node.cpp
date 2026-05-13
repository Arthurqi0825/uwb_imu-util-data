// =============================================================================
// integrity_uwb_node.cpp  —  WLS + Loosely Coupled IMU FGO
//
// PIPELINE
// ─────────────────────────────────────────────────────────────────────────────
//  UWB stream → cycle accumulator (validated, unchanged)
//                    │
//                    ├─► WLS solver  ───────────────────────────────────────►  uwb_wls / uwb_wls_path
//                    │
//                    └─► LC-FGO  (WLS prior + ImuFactor + BiasBetween)  ──►  lc_fusion / lc_fusion_path
//
//  IMU stream → imu_buf_ (replayed into one LC preintegrator)
//
// LOOSELY COUPLED (LC)
//   Treats the WLS solution as a 3-DOF position measurement with sigma=wls_sigma.
//   Each cycle: ImuFactor + BiasBetween + PriorFactor<Pose3> (tight on XYZ,
//   effectively unconstrained on rotation — rotation comes from IMU only).
//
// IMU UNITS
//   GTSAM preintegration expects specific force in m/s^2. If the incoming IMU
//   publishes acceleration in g, this node converts it before buffering.
//
// SIGN CONVENTION (DO NOT CHANGE — validated on real data)
//   WLS residual:  r(i) = (||p-aB|| - ||p-aA||) - tdoa_measured
//
// FIXES vs. previous version
// ─────────────────────────────────────────────────────────────────────────────
// FIX A – Dynamic sigma defaults tuned to match real UWB hardware
//   Previous defaults produced sigma_pair ≈ 1.55 m (observed in logs) due to
//   tdoa_range_sigma_per_meter=0.02 with ranges ≈ 7–8 m.  This inflated
//   P=(HᵀWH)⁻¹ and drove HPL ≈ 5 m >> HAL, making the system permanently
//   unavailable even with chi² T ≈ 0.004.
//
//   New defaults:
//     tdoa_range_sigma_base      0.10 m  → 0.05 m   (UWB hardware floor)
//     tdoa_range_sigma_per_meter 0.02    → 0.005     (gentle range dependence)
//     tdoa_pair_sigma_max        10.0 m  → 0.50 m    (cap prevents HPL blow-up)
//
//   With 8 m range: sigma_pair = sqrt(2*(0.05+0.005*8)²) ≈ 0.128 m
//   This gives HPL ≈ 0.3–0.5 m which is consistent with the observed
//   WLS-GT errors of 0.07–0.13 m.
//
//   All defaults are ROS-param overridable; set them in your launch file to
//   match your specific hardware characterisation.
//
// FIX B – HAL/VAL defaults raised to match indoor UWB performance
//   Previous defaults: HAL=0.50 m, VAL=1.00 m
//   New defaults:      HAL=1.00 m, VAL=2.00 m
//   Rationale: the observed WLS-GT errors are 0.07–0.13 m horizontal and
//   ~0.1 m vertical.  A protection level of 0.5 m would require HPL < 0.5 m
//   which demands sigma_pair < 0.08 m — only achievable in ideal LOS
//   conditions.  HAL=1.0 m is a pragmatic indoor standard.
//   Both values remain ROS-param overridable.
//
// FIX C – Integrity LC gate uses full availability flag (HPL + chi² combined)
//   Previous code gated only on chi² fault flag; HPL/VAL were checked but
//   did not block the LC update.  Now the gate checks ir.available which
//   combines chi² + HPL < HAL + VPL < VAL, matching the intent of RAIM.
//   A separate ROS param `integrity_gate_mode` allows reverting to chi2-only
//   gating if desired (e.g. during initial deployment when HAL is being tuned).
//
// FIX D – Low-DOF warning when M == 4
//   Added ROS_WARN_THROTTLE when DOF == 1 (M==4) to alert the operator that
//   chi-squared detection power is minimal.
// =============================================================================

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <visualization_msgs/MarkerArray.h>
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
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "integrity_monitor.h"
#include "wlssolver.h"

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

namespace uwb_imu_fusion
{

  struct ImuMeas
  {
    double t;
    gtsam::Vector3 acc;
    gtsam::Vector3 gyr;
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

  class UwbImuLcWlsFusionNode
  {
  public:
    UwbImuLcWlsFusionNode(ros::NodeHandle &nh, ros::NodeHandle &pnh)
        : nh_(nh), pnh_(pnh)
    {
      loadParams();
      setupGtsam();
      openLog();

      wls_odom_pub_   = nh_.advertise<nav_msgs::Odometry>("uwb_wls", 50);
      lc_odom_pub_    = nh_.advertise<nav_msgs::Odometry>("lc_fusion", 50);
      wls_path_pub_   = nh_.advertise<nav_msgs::Path>("uwb_wls_path", 10);
      lc_path_pub_    = nh_.advertise<nav_msgs::Path>("lc_fusion_path", 10);
      gt_path_pub_    = nh_.advertise<nav_msgs::Path>("uwb_imu_path_gt", 10);
      integ_odom_pub_ = nh_.advertise<nav_msgs::Odometry>("uwb_integrity", 50);
      anchor_marker_pub_ =
          nh_.advertise<visualization_msgs::MarkerArray>("uwb_anchors", 1, true);
      publishAnchorMarkers();

      std::string imu_topic, uwb_topic, gt_topic;
      pnh_.param<std::string>("imu_topic", imu_topic, "/imu_data");
      pnh_.param<std::string>("uwb_topic", uwb_topic, "/tdoa_data");
      pnh_.param<std::string>("gt_topic",  gt_topic,  "/pose_data");
      imu_sub_ = nh_.subscribe(imu_topic, 1000, &UwbImuLcWlsFusionNode::imuCallback, this);
      uwb_sub_ = nh_.subscribe(uwb_topic,  500, &UwbImuLcWlsFusionNode::uwbCallback, this);
      gt_sub_  = nh_.subscribe(gt_topic,    50, &UwbImuLcWlsFusionNode::gtCallback,  this);

      wls_guess_ = Eigen::Vector3d(initial_pose_.translation().x(),
                                   initial_pose_.translation().y(),
                                   initial_pose_.translation().z());

      ROS_INFO("[uwb_imu_fusion] %zu anchors  meas/cycle=%d",
               anchors_.size(), measurements_per_cycle_);
      ROS_INFO("[uwb_imu_fusion] gravity=%.3f  wls_sigma=%.3f",
               gravity_mag_, wls_sigma_);
      ROS_INFO("[uwb_imu_fusion] accel units=%s  accel_scale=%.6f  gyro_units=%s",
               imu_accel_in_g_ ? "g" : "m/s^2",
               accel_scale_,
               gyro_is_degrees_ ? "deg/s" : "rad/s");
      ROS_INFO("[uwb_imu_fusion] lc update regularization: vel_sigma=%.3f  bias_sigma=%.3f",
               lc_update_vel_sigma_, lc_update_bias_sigma_);
      ROS_INFO("[uwb_imu_fusion] dynamic TDOA sigma: %s  base=%.3f  slope=%.4f  clamp=[%.3f, %.3f]",
               dynamic_tdoa_sigma_enabled_ ? "ON" : "OFF",
               tdoa_range_sigma_base_, tdoa_range_sigma_per_meter_,
               tdoa_pair_sigma_min_, tdoa_pair_sigma_max_);
      ROS_INFO("[uwb_imu_fusion] integrity gate mode: %s",
               integrity_gate_full_avail_ ? "FULL (chi2+HPL+VPL)" : "CHI2-ONLY");
    }

    ~UwbImuLcWlsFusionNode()
    {
      if (traj_log_.is_open())
        traj_log_.close();
    }

  private:
    // =========================================================================
    void loadParams()
    // =========================================================================
    {
      auto get = [&](const std::string &k, auto &v, auto def)
      {
        if (!nh_.getParam("/uwb_imu_fusion/" + k, v) && !pnh_.getParam(k, v))
          v = def;
      };
      auto get_private_first = [&](const std::string &k, auto &v, auto def)
      {
        if (!pnh_.getParam(k, v) && !nh_.getParam("/uwb_imu_fusion/" + k, v))
          v = def;
      };

      // ── Anchors ─────────────────────────────────────────────────────────────
      XmlRpc::XmlRpcValue axv;
      const bool ok = nh_.getParam("/uwb_imu_fusion/anchors", axv) ||
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
        ROS_ERROR("[uwb_imu_fusion] No anchors configured — shutting down.");
        ros::shutdown();
        return;
      }

      // ── Anchor extrinsic transform ───────────────────────────────────────────
      std::vector<double> ext_t{0, 0, 0}, ext_q{0, 0, 0, 1};
      if (!nh_.getParam("/uwb_imu_fusion/anchor_extrinsic_translation", ext_t))
        pnh_.getParam("anchor_extrinsic_translation", ext_t);
      if (!nh_.getParam("/uwb_imu_fusion/anchor_extrinsic_rotation", ext_q))
        pnh_.getParam("anchor_extrinsic_rotation", ext_q);
      Eigen::Quaterniond qe(ext_q[3], ext_q[0], ext_q[1], ext_q[2]);
      qe.normalize();
      const Eigen::Vector3d te(ext_t[0], ext_t[1], ext_t[2]);
      for (auto &a : anchors_)
      {
        const Eigen::Vector3d pw =
            qe.toRotationMatrix() * Eigen::Vector3d(a.x(), a.y(), a.z()) + te;
        a = gtsam::Point3(pw.x(), pw.y(), pw.z());
      }

      // ── IMU parameters ──────────────────────────────────────────────────────
      get("gyro_is_degrees",      gyro_is_degrees_, false);
      get("gravity_magnitude",    gravity_mag_,     9.80665);
      get("imu_accel_in_g",       imu_accel_in_g_,  true);
      const double default_accel_scale = imu_accel_in_g_ ? gravity_mag_ : 1.0;
      get("accel_scale",          accel_scale_,     default_accel_scale);

      get("accel_noise_sigma",    accel_sigma_,           0.003);
      get("gyro_noise_sigma",     gyro_sigma_,            0.0003);
      get("accel_bias_rw_sigma",  accel_bias_rw_sigma_,   0.0003);
      get("gyro_bias_rw_sigma",   gyro_bias_rw_sigma_,    0.00003);
      get("integration_noise_sigma", integration_sigma_,  1e-6);

      // ── Cycle parameters ─────────────────────────────────────────────────────
      get("cycle_timeout", cycle_timeout_, 0.1);
      int mpc = 8, min_imu = 2;
      get("measurements_per_cycle", mpc,     8);
      get("min_imu_per_cycle",       min_imu, 2);
      measurements_per_cycle_ = mpc;
      min_imu_per_cycle_      = min_imu;

      // ── FGO parameters ───────────────────────────────────────────────────────
      get("wls_position_sigma", wls_sigma_, 0.15);

      std::vector<double> pps{0.5, 0.5, 0.5, 0.1, 0.1, 0.1};
      if (!nh_.getParam("/uwb_imu_fusion/prior_pose_sigmas", pps))
        pnh_.getParam("prior_pose_sigmas", pps);
      prior_pose_sigmas_ = gtsam::Vector6::Map(pps.data());
      get("prior_vel_sigma",        prior_vel_sigma_,        0.1);
      get("prior_bias_sigma",       prior_bias_sigma_,       0.01);
      get("lc_update_vel_sigma",    lc_update_vel_sigma_,    2.0);
      get("lc_update_bias_sigma",   lc_update_bias_sigma_,   0.5);
      get("lc_wls_reset_threshold", lc_wls_reset_threshold_, 1.0);
      get("wls_consistency_gate_enabled", wls_consistency_gate_enabled_, true);
      get("wls_reference_gate",      wls_reference_gate_,      0.75);
      get("wls_vertical_gate",       wls_vertical_gate_,       0.55);
      get("wls_max_step",           wls_max_step_,            0.85);
      get("wls_hold_last_on_reject", wls_hold_last_on_reject_, true);

      // ── Initial pose ─────────────────────────────────────────────────────────
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

      get("odom_frame",          odom_frame_,      std::string("map"));
      get("base_frame",          base_frame_,      std::string("base_link"));
      get("trajectory_log_path", traj_log_path_,
          std::string("/tmp/uwb_imu_trajectory.csv"));

      // ── Integrity monitoring parameters ──────────────────────────────────────
      //
      // FIX A: sigma defaults tuned to real UWB hardware (see header comment).
      //   With range ≈ 8 m: sigma_pair = sqrt(2*(0.05 + 0.005*8)²) ≈ 0.128 m
      //   This keeps HPL in the 0.3–0.5 m range and allows availability.
      //
      // FIX B: HAL/VAL defaults raised to practical indoor UWB limits.
      //   HAL 0.50→1.00 m,  VAL 1.00→2.00 m.  Override in your launch file.
      // ────────────────────────────────────────────────────────────────────────
      double integ_noise_sigma  = 0.05;   // FIX A: was 0.10
      double integ_p_fa         = 1e-3;
      double integ_hal          = 1.00;   // FIX B: was 0.50
      double integ_val          = 2.00;   // FIX B: was 1.00
      double integ_ring_thresh  = 0.10;
      bool   integ_fde          = true;

      get_private_first("integrity_tdoa_noise_sigma", integ_noise_sigma,  0.05);
      get_private_first("integrity_p_fa",             integ_p_fa,         1e-3);
      get_private_first("integrity_hal",              integ_hal,          1.00);
      get_private_first("integrity_val",              integ_val,          2.00);
      get_private_first("integrity_ring_threshold",   integ_ring_thresh,  0.10);
      get_private_first("integrity_enable_fde",       integ_fde,          true);

      // FIX C: gate mode — default uses full availability (chi2 + HPL + VPL)
      get_private_first("integrity_gate_full_avail",  integrity_gate_full_avail_, true);

      // Dynamic sigma model
      get_private_first("dynamic_tdoa_sigma_enabled",    dynamic_tdoa_sigma_enabled_,  true);
      get_private_first("tdoa_range_sigma_base",         tdoa_range_sigma_base_,       0.05);  // FIX A
      get_private_first("tdoa_range_sigma_per_meter",    tdoa_range_sigma_per_meter_,  0.005); // FIX A
      get_private_first("tdoa_pair_sigma_min",           tdoa_pair_sigma_min_,         1e-3);
      get_private_first("tdoa_pair_sigma_max",           tdoa_pair_sigma_max_,         0.50);  // FIX A

      get_private_first("nlos_rejection_enabled",        nlos_rejection_enabled_,      true);
      get_private_first("nlos_robust_weighting_enabled", nlos_robust_weighting_enabled_, true);
      get_private_first("nlos_huber_k",                  nlos_huber_k_,                1.5);
      get_private_first("nlos_sigma_inflation_max",      nlos_sigma_inflation_max_,    8.0);
      get_private_first("nlos_robust_outer_iterations",  nlos_robust_outer_iterations_, 4);
#ifdef UWB_SUBSET_CONSENSUS_INTEGRITY
      subset_consensus_enabled_ = true;
#endif
      get_private_first("subset_consensus_enabled",      subset_consensus_enabled_,
                        subset_consensus_enabled_);
      get_private_first("subset_consensus_trials",       subset_consensus_trials_,       80);
      get_private_first("subset_consensus_anchor_count", subset_consensus_anchor_count_, 0);
      get_private_first("subset_consensus_min_measurements",
                        subset_consensus_min_measurements_, 3);
      get_private_first("subset_consensus_cluster_gate", subset_consensus_cluster_gate_, 0.25);
      get_private_first("subset_consensus_reject_score", subset_consensus_reject_score_, 0.55);
      get_private_first("subset_consensus_min_support",  subset_consensus_min_support_, 3);
      get_private_first("nlos_anchor_rejection_enabled", nlos_anchor_rejection_enabled_, true);
      get_private_first("nlos_anchor_min_incident_measurements",
                        nlos_anchor_min_incident_measurements_, 2);
      get_private_first("nlos_anchor_min_remaining_measurements",
                        nlos_anchor_min_remaining_measurements_, 6);
      get_private_first("nlos_anchor_solution_shift_gate",
                        nlos_anchor_solution_shift_gate_, 0.25);
      get_private_first("nlos_anchor_prior_improvement_gate",
                        nlos_anchor_prior_improvement_gate_, 0.15);
      get_private_first("nlos_anchor_chi2_ratio",        nlos_anchor_chi2_ratio_,      0.85);
      get_private_first("nlos_temporal_filter_enabled",  nlos_temporal_filter_enabled_, true);
      get_private_first("nlos_score_decay",              nlos_score_decay_,            0.85);
      get_private_first("nlos_score_gate_sigma",         nlos_score_gate_sigma_,       2.5);
      get_private_first("nlos_score_increment",          nlos_score_increment_,        1.0);
      get_private_first("nlos_score_threshold",          nlos_score_threshold_,        1.0);
      get_private_first("nlos_score_sigma_scale_max",    nlos_score_sigma_scale_max_,  5.0);
      get_private_first("nlos_residual_gate_sigma",      nlos_residual_gate_sigma_,    2.8);
      get_private_first("nlos_residual_gate_abs",        nlos_residual_gate_abs_,      0.35);
      get_private_first("nlos_max_rejections",           nlos_max_rejections_,         1);
      get_private_first("nlos_min_measurements",         nlos_min_measurements_,       6);

      if (tdoa_pair_sigma_max_ < tdoa_pair_sigma_min_)
        std::swap(tdoa_pair_sigma_min_, tdoa_pair_sigma_max_);
      nlos_robust_outer_iterations_ = std::max(1, nlos_robust_outer_iterations_);
      nlos_sigma_inflation_max_ = std::max(1.0, nlos_sigma_inflation_max_);
      subset_consensus_trials_ = std::max(1, subset_consensus_trials_);
      subset_consensus_min_measurements_ = std::max(3, subset_consensus_min_measurements_);
      subset_consensus_cluster_gate_ = std::max(0.01, subset_consensus_cluster_gate_);
      subset_consensus_reject_score_ =
          std::min(1.0, std::max(0.0, subset_consensus_reject_score_));
      subset_consensus_min_support_ = std::max(1, subset_consensus_min_support_);
      nlos_anchor_min_incident_measurements_ =
          std::max(1, nlos_anchor_min_incident_measurements_);
      nlos_anchor_min_remaining_measurements_ =
          std::max(3, nlos_anchor_min_remaining_measurements_);
      nlos_anchor_chi2_ratio_ =
          std::min(1.0, std::max(0.0, nlos_anchor_chi2_ratio_));
      nlos_score_decay_ = std::min(0.999, std::max(0.0, nlos_score_decay_));
      nlos_score_gate_sigma_ = std::max(0.1, nlos_score_gate_sigma_);
      nlos_score_increment_ = std::max(0.0, nlos_score_increment_);
      nlos_score_sigma_scale_max_ = std::max(1.0, nlos_score_sigma_scale_max_);

      // Push parameters into the monitor
      integ_monitor_.tdoa_noise_sigma = integ_noise_sigma;
      integ_monitor_.p_fa             = integ_p_fa;
      integ_monitor_.hal              = integ_hal;
      integ_monitor_.val              = integ_val;
      integ_monitor_.ring_threshold   = integ_ring_thresh;
      integ_monitor_.enable_fde       = integ_fde;
    }

    // =========================================================================
    void setupGtsam()
    // =========================================================================
    {
      auto p = gtsam::PreintegrationParams::MakeSharedU(gravity_mag_);
      p->setAccelerometerCovariance(
          gtsam::Matrix33::Identity() * std::pow(accel_sigma_, 2));
      p->setGyroscopeCovariance(
          gtsam::Matrix33::Identity() * std::pow(gyro_sigma_, 2));
      p->setIntegrationCovariance(
          gtsam::Matrix33::Identity() * std::pow(integration_sigma_, 2));
      imu_params_ = p;

      lc_.bias    = gtsam::imuBias::ConstantBias();
      lc_preint_  = boost::make_shared<gtsam::PreintegratedImuMeasurements>(
                        imu_params_, lc_.bias);

      lc_isam_ = makeIsam2();
    }

    // =========================================================================
    void openLog()
    // =========================================================================
    {
      traj_log_.open(traj_log_path_, std::ios::out | std::ios::trunc);
      if (traj_log_.is_open())
      {
        traj_log_ << "timestamp,"
                  << "gt_x,gt_y,gt_z,"
                  << "wls_x,wls_y,wls_z,"
                  << "lc_x,lc_y,lc_z,"
                  << "lc_qx,lc_qy,lc_qz,lc_qw,"
                  << "lc_vx,lc_vy,lc_vz,"
                  << "lc_bias_ax,lc_bias_ay,lc_bias_az,"
                  << "wls_gt_err,lc_gt_err,"
                  << "integ_chi2,integ_threshold,integ_dof,"
                  << "integ_fault,integ_ring_sum,integ_ring_ok,"
                  << "integ_hpl,integ_vpl,integ_available,"
                  << "integ_excluded_idx,"
                  << "integ_ring_loops,integ_chi2_after_fde,"
                  << "raw_meas_count,used_meas_count,"
                  << "integrity_gate,wls_accepted,consistency_reject,wls_held_last,"
                  << "last_good_wls_age,lc_wls_gap,"
                  << "residual_abs_max,residual_rms,std_resid_abs_max,"
                  << "sigma_min,sigma_mean,sigma_max,"
                  << "nlos_rejected_count,nlos_rejected_pairs\n";
      }
      else
      {
        ROS_WARN("[uwb_imu_fusion] Cannot open log: %s", traj_log_path_.c_str());
      }
    }

    // =========================================================================
    void logRow(double t, bool wls_ok, const Eigen::Vector3d &wls_p,
                const IntegrityResult &ir,
                size_t raw_meas_count, size_t used_meas_count,
                bool integrity_gate, bool wls_accepted,
                bool consistency_reject, bool wls_held_last)
    // =========================================================================
    {
      if (!traj_log_.is_open()) return;
      const auto nan = std::numeric_limits<double>::quiet_NaN();
      const auto v3  = [](const gtsam::Point3 &p) {
        return Eigen::Vector3d(p.x(), p.y(), p.z());
      };
      const Eigen::Vector3d lp = lc_.initialized
                                     ? v3(lc_.pose.translation())
                                     : Eigen::Vector3d(nan, nan, nan);
      const Eigen::Quaterniond lq = lc_.initialized
                                        ? Eigen::Quaterniond(lc_.pose.rotation().matrix())
                                        : Eigen::Quaterniond(nan, nan, nan, nan);
      const Eigen::Vector3d gt_e(gt_pos_.x(), gt_pos_.y(), gt_pos_.z());
      const double last_good_age =
          last_good_wls_have_ ? std::max(0.0, t - last_good_wls_t_) : nan;
      const double lc_wls_gap =
          (wls_ok && lc_.initialized) ? (lp - wls_p).norm() : nan;

      double residual_abs_max = nan;
      double residual_rms = nan;
      if (!ir.residuals.empty())
      {
        double sum_sq = 0.0;
        residual_abs_max = 0.0;
        for (const double r : ir.residuals)
        {
          residual_abs_max = std::max(residual_abs_max, std::fabs(r));
          sum_sq += r * r;
        }
        residual_rms = std::sqrt(sum_sq / static_cast<double>(ir.residuals.size()));
      }

      double std_resid_abs_max = nan;
      if (!ir.std_resid.empty())
      {
        std_resid_abs_max = 0.0;
        for (const double sr : ir.std_resid)
          std_resid_abs_max = std::max(std_resid_abs_max, std::fabs(sr));
      }

      double sigma_min = nan;
      double sigma_mean = nan;
      double sigma_max = nan;
      if (!ir.sigmas.empty())
      {
        sigma_min = std::numeric_limits<double>::infinity();
        sigma_max = 0.0;
        double sigma_sum = 0.0;
        for (const double s : ir.sigmas)
        {
          sigma_min = std::min(sigma_min, s);
          sigma_max = std::max(sigma_max, s);
          sigma_sum += s;
        }
        sigma_mean = sigma_sum / static_cast<double>(ir.sigmas.size());
      }

      traj_log_ << std::fixed << std::setprecision(6)
                << t << ","
                << (gt_have_ ? gt_pos_.x() : nan) << ","
                << (gt_have_ ? gt_pos_.y() : nan) << ","
                << (gt_have_ ? gt_pos_.z() : nan) << ","
                << (wls_ok ? wls_p.x() : nan) << ","
                << (wls_ok ? wls_p.y() : nan) << ","
                << (wls_ok ? wls_p.z() : nan) << ","
                << lp.x() << "," << lp.y() << "," << lp.z() << ","
                << lq.x() << "," << lq.y() << "," << lq.z() << "," << lq.w() << ","
                << lc_.vel.x() << "," << lc_.vel.y() << "," << lc_.vel.z() << ","
                << lc_.bias.accelerometer().x() << ","
                << lc_.bias.accelerometer().y() << ","
                << lc_.bias.accelerometer().z() << ","
                << (wls_ok && gt_have_ ? (wls_p - gt_e).norm() : nan) << ","
                << (lc_.initialized && gt_have_ ? (lp - gt_e).norm() : nan) << ","
                << ir.chi2_stat       << ","
                << ir.chi2_threshold  << ","
                << ir.dof             << ","
                << (ir.fault_detected ? 1 : 0) << ","
                << ir.ring_sum        << ","
                << (ir.ring_ok ? 1 : 0) << ","
                << ir.hpl             << ","
                << ir.vpl             << ","
                << (ir.available ? 1 : 0) << ","
                << ir.excluded_idx    << ","
                << ir.ring_closed_loops << ","
                << ir.chi2_after_fde  << ","
                << raw_meas_count     << ","
                << used_meas_count    << ","
                << (integrity_gate ? 1 : 0) << ","
                << (wls_accepted ? 1 : 0) << ","
                << (consistency_reject ? 1 : 0) << ","
                << (wls_held_last ? 1 : 0) << ","
                << last_good_age      << ","
                << lc_wls_gap         << ","
                << residual_abs_max   << ","
                << residual_rms       << ","
                << std_resid_abs_max  << ","
                << sigma_min          << ","
                << sigma_mean         << ","
                << sigma_max          << ","
                << last_nlos_rejected_count_ << ","
                << csvEscape(last_nlos_rejected_pairs_) << "\n";
      traj_log_.flush();
    }

    // =========================================================================
    void imuCallback(const sensor_msgs::Imu::ConstPtr &msg)
    // =========================================================================
    {
      std::lock_guard<std::mutex> lk(mutex_);
      const double t = msg->header.stamp.toSec();
      if (last_imu_t_ >= 0.0 && t <= last_imu_t_)
      {
        ROS_WARN_THROTTLE(1.0, "[uwb_imu_fusion] IMU out-of-order %.6f", t);
        return;
      }

      double gx = msg->angular_velocity.x;
      double gy = msg->angular_velocity.y;
      double gz = msg->angular_velocity.z;
      if (gyro_is_degrees_)
      {
        const double d2r = M_PI / 180.0;
        gx *= d2r; gy *= d2r; gz *= d2r;
      }

      ImuMeas s;
      s.t   = t;
      s.acc = gtsam::Vector3(msg->linear_acceleration.x * accel_scale_,
                             msg->linear_acceleration.y * accel_scale_,
                             msg->linear_acceleration.z * accel_scale_);
      s.gyr = gtsam::Vector3(gx, gy, gz);
      imu_buf_.push_back(s);
      last_imu_t_ = t;

      const double fence = lc_.initialized ? lc_.last_cycle_t - 0.5 : t - 5.0;
      while (!imu_buf_.empty() && imu_buf_.front().t < fence)
        imu_buf_.pop_front();
    }

    // =========================================================================
    void uwbCallback(const cf_msgs::Tdoa::ConstPtr &msg)
    // =========================================================================
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

      const auto pr = std::make_pair(std::min(ts.idA, ts.idB),
                                     std::max(ts.idA, ts.idB));
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

    // =========================================================================
    // computeAnchorSigmas
    //
    // Per-anchor 1-sigma range noise based on the linear model
    //   sigma_anchor_k = base + slope * range_k
    // Returns a vector of size anchors_.size().  Used by the integrity monitor
    // to build the full correlated TDOA covariance C = D · diag(σ²) · D^T.
    // =========================================================================
    Eigen::VectorXd computeAnchorSigmas(const Eigen::Vector3d &p_wls) const
    {
      Eigen::VectorXd sa(static_cast<int>(anchors_.size()));
      for (int k = 0; k < static_cast<int>(anchors_.size()); ++k)
      {
        const Eigen::Vector3d ak(anchors_[k].x(), anchors_[k].y(), anchors_[k].z());
        const double range_k = (p_wls - ak).norm();
        sa(k) = std::max(1e-9,
            tdoa_range_sigma_base_ + tdoa_range_sigma_per_meter_ * range_k);
      }
      return sa;
    }

    // =========================================================================
    // computeDynamicTdoaSigmas
    //
    // Per-measurement TDOA sigma based on range to each anchor:
    //   sigma_i = base + slope * range_i
    //   sigma_pair = sqrt(sigma_A^2 + sigma_B^2)   clamped to [min, max]
    //
    // FIX A: defaults changed so that typical indoor ranges produce
    //        sigma_pair ≈ 0.10–0.15 m (was ≈ 1.5 m with old defaults).
    // =========================================================================
    Eigen::VectorXd computeDynamicTdoaSigmas(const std::vector<TdoaMeas> &meas,
                                             const Eigen::Vector3d       &p_wls) const
    {
      Eigen::VectorXd sigmas(static_cast<int>(meas.size()));
      for (int i = 0; i < static_cast<int>(meas.size()); ++i)
      {
        const auto &m = meas[i];
        const Eigen::Vector3d aA(anchors_[m.idA].x(),
                                 anchors_[m.idA].y(),
                                 anchors_[m.idA].z());
        const Eigen::Vector3d aB(anchors_[m.idB].x(),
                                 anchors_[m.idB].y(),
                                 anchors_[m.idB].z());
        const double range_a = (p_wls - aA).norm();
        const double range_b = (p_wls - aB).norm();
        const double sigma_a = std::max(1e-9,
            tdoa_range_sigma_base_ + tdoa_range_sigma_per_meter_ * range_a);
        const double sigma_b = std::max(1e-9,
            tdoa_range_sigma_base_ + tdoa_range_sigma_per_meter_ * range_b);
        const double pair_sigma = std::sqrt(sigma_a * sigma_a + sigma_b * sigma_b);
        sigmas(i) = std::min(tdoa_pair_sigma_max_,
                             std::max(tdoa_pair_sigma_min_, pair_sigma));
      }
      return sigmas;
    }

    // =========================================================================
    double tdoaResidualAt(const TdoaMeas &m, const Eigen::Vector3d &p) const
    // =========================================================================
    {
      const Eigen::Vector3d aA(anchors_[m.idA].x(),
                               anchors_[m.idA].y(),
                               anchors_[m.idA].z());
      const Eigen::Vector3d aB(anchors_[m.idB].x(),
                               anchors_[m.idB].y(),
                               anchors_[m.idB].z());
      const double rA = std::max((p - aA).norm(), 1e-9);
      const double rB = std::max((p - aB).norm(), 1e-9);
      return (rB - rA) - m.tdoa;
    }

    // =========================================================================
    double weightedResidualChi2(const std::vector<TdoaMeas> &meas,
                                const Eigen::VectorXd       &sigmas,
                                const Eigen::Vector3d       &p) const
    // =========================================================================
    {
      if (sigmas.size() != static_cast<int>(meas.size()))
        return std::numeric_limits<double>::quiet_NaN();

      double chi2 = 0.0;
      for (int i = 0; i < static_cast<int>(meas.size()); ++i)
      {
        const double sigma = std::max(sigmas(i), 1e-9);
        const double std_r = tdoaResidualAt(meas[i], p) / sigma;
        chi2 += std_r * std_r;
      }
      return chi2;
    }

    // =========================================================================
    double temporalNlosScore(const TdoaMeas &m) const
    // =========================================================================
    {
      const auto key = std::make_pair(std::min(m.idA, m.idB),
                                      std::max(m.idA, m.idB));
      double score = 0.0;
      const auto pit = nlos_pair_scores_.find(key);
      if (pit != nlos_pair_scores_.end())
        score = std::max(score, pit->second);

      const auto ait = nlos_anchor_scores_.find(m.idA);
      if (ait != nlos_anchor_scores_.end())
        score = std::max(score, ait->second);
      const auto bit = nlos_anchor_scores_.find(m.idB);
      if (bit != nlos_anchor_scores_.end())
        score = std::max(score, bit->second);
      return score;
    }

    // =========================================================================
    // rejectNlosBySubsetConsensus
    //
    // RANSAC-style anchor integrity check:
    //   1. Build many anchor subsets from the anchors visible in this cycle.
    //   2. Solve WLS using only TDOAs whose two endpoints are inside each subset.
    //   3. Find the largest cluster of subset solutions in position space.
    //   4. Vote against anchors that appear mostly in non-consensus subsets.
    //
    // If one anchor is NLOS, subsets containing that anchor tend to solve to a
    // different position, while LOS-only subsets cluster together.
    // =========================================================================
    bool rejectNlosBySubsetConsensus(double t_mid,
                                     std::vector<TdoaMeas> &meas,
                                     Eigen::VectorXd       &sigmas,
                                     Eigen::Vector3d       &wls_p)
    {
      if (!nlos_rejection_enabled_ ||
          !subset_consensus_enabled_ ||
          sigmas.size() != static_cast<int>(meas.size()) ||
          static_cast<int>(meas.size()) <= nlos_min_measurements_)
        return false;

      std::vector<int> active;
      for (const auto &m : meas)
      {
        if (std::find(active.begin(), active.end(), m.idA) == active.end())
          active.push_back(m.idA);
        if (std::find(active.begin(), active.end(), m.idB) == active.end())
          active.push_back(m.idB);
      }
      std::sort(active.begin(), active.end());

      if (static_cast<int>(active.size()) < 5)
        return false;

      int subset_size = subset_consensus_anchor_count_ > 0
                            ? subset_consensus_anchor_count_
                            : static_cast<int>(active.size()) - 1;
      subset_size = std::min(subset_size, static_cast<int>(active.size()) - 1);
      subset_size = std::max(4, subset_size);
      if (subset_size >= static_cast<int>(active.size()))
        return false;

      struct Trial
      {
        std::vector<int> anchors;
        Eigen::Vector3d p{Eigen::Vector3d::Zero()};
        double rms{std::numeric_limits<double>::infinity()};
        bool in_cluster{false};
      };

      auto in_subset = [](const std::vector<int> &subset, int id)
      {
        return std::binary_search(subset.begin(), subset.end(), id);
      };

      auto run_trial = [&](const std::vector<int> &subset, Trial &trial) -> bool
      {
        std::vector<TdoaMeas> kept;
        kept.reserve(meas.size());
        Eigen::VectorXd kept_sigmas(static_cast<int>(meas.size()));
        int row = 0;
        for (int i = 0; i < static_cast<int>(meas.size()); ++i)
        {
          if (!in_subset(subset, meas[i].idA) || !in_subset(subset, meas[i].idB))
            continue;
          kept.push_back(meas[i]);
          kept_sigmas(row++) = sigmas(i);
        }
        if (static_cast<int>(kept.size()) < subset_consensus_min_measurements_)
          return false;
        kept_sigmas.conservativeResize(static_cast<int>(kept.size()));

        Eigen::Vector3d p = lc_.initialized
                                ? Eigen::Vector3d(lc_.pose.x(), lc_.pose.y(), lc_.pose.z())
                                : wls_guess_;
        if (!wls_solver_.solveWeighted(kept, anchors_, kept_sigmas, p))
          return false;

        const double chi2 = weightedResidualChi2(kept, kept_sigmas, p);
        if (!std::isfinite(chi2))
          return false;

        trial.anchors = subset;
        trial.p = p;
        trial.rms = std::sqrt(chi2 / static_cast<double>(kept.size()));
        return true;
      };

      std::vector<std::vector<int>> subsets;
      std::vector<int> mask(active.size(), 0);
      std::fill(mask.begin(), mask.begin() + subset_size, 1);
      do
      {
        std::vector<int> subset;
        subset.reserve(subset_size);
        for (size_t i = 0; i < active.size(); ++i)
          if (mask[i]) subset.push_back(active[i]);
        subsets.push_back(std::move(subset));
      } while (std::prev_permutation(mask.begin(), mask.end()));

      if (static_cast<int>(subsets.size()) > subset_consensus_trials_)
      {
        std::mt19937 rng(static_cast<uint32_t>(std::llround(t_mid * 1000.0)));
        std::shuffle(subsets.begin(), subsets.end(), rng);
        subsets.resize(subset_consensus_trials_);
      }

      std::vector<Trial> trials;
      trials.reserve(subsets.size());
      for (const auto &subset : subsets)
      {
        Trial trial;
        if (run_trial(subset, trial))
          trials.push_back(std::move(trial));
      }

      if (static_cast<int>(trials.size()) < subset_consensus_min_support_)
        return false;

      int best_center = -1;
      int best_support = 0;
      double best_rms_sum = std::numeric_limits<double>::infinity();
      for (int i = 0; i < static_cast<int>(trials.size()); ++i)
      {
        int support = 0;
        double rms_sum = 0.0;
        for (const auto &trial : trials)
        {
          if ((trial.p - trials[i].p).norm() <= subset_consensus_cluster_gate_)
          {
            ++support;
            rms_sum += trial.rms;
          }
        }
        if (support > best_support ||
            (support == best_support && rms_sum < best_rms_sum))
        {
          best_center = i;
          best_support = support;
          best_rms_sum = rms_sum;
        }
      }

      if (best_center < 0 || best_support < subset_consensus_min_support_)
        return false;

      Eigen::Vector3d consensus = Eigen::Vector3d::Zero();
      for (auto &trial : trials)
      {
        trial.in_cluster =
            (trial.p - trials[best_center].p).norm() <= subset_consensus_cluster_gate_;
        if (trial.in_cluster)
          consensus += trial.p;
      }
      consensus /= static_cast<double>(best_support);

      int best_anchor = -1;
      double best_score = -1.0;
      int best_seen = 0;
      int best_bad = 0;
      for (const int anchor_id : active)
      {
        int seen = 0;
        int bad = 0;
        for (const auto &trial : trials)
        {
          if (!in_subset(trial.anchors, anchor_id))
            continue;
          ++seen;
          if (!trial.in_cluster)
            ++bad;
        }
        if (seen < subset_consensus_min_support_)
          continue;

        const double score = static_cast<double>(bad) / static_cast<double>(seen);
        if (score > best_score)
        {
          best_anchor = anchor_id;
          best_score = score;
          best_seen = seen;
          best_bad = bad;
        }
      }

      if (best_anchor < 0 || best_score < subset_consensus_reject_score_)
        return false;

      std::vector<TdoaMeas> filtered;
      filtered.reserve(meas.size());
      Eigen::VectorXd filtered_sigmas(static_cast<int>(meas.size()));
      std::ostringstream rejected_pairs;
      int row = 0;
      int rejected = 0;
      for (int i = 0; i < static_cast<int>(meas.size()); ++i)
      {
        if (meas[i].idA == best_anchor || meas[i].idB == best_anchor)
        {
          if (rejected > 0) rejected_pairs << ";";
          rejected_pairs << meas[i].idA << "-" << meas[i].idB;
          ++rejected;
          continue;
        }
        filtered.push_back(meas[i]);
        filtered_sigmas(row++) = sigmas(i);
      }
      filtered_sigmas.conservativeResize(static_cast<int>(filtered.size()));

      if (static_cast<int>(filtered.size()) < nlos_min_measurements_)
        return false;

      Eigen::Vector3d filtered_p = consensus;
      if (!wls_solver_.solveWeighted(filtered, anchors_, filtered_sigmas, filtered_p))
        return false;

      for (const auto &m : meas)
      {
        if (m.idA != best_anchor && m.idB != best_anchor)
          continue;
        const auto key = std::make_pair(std::min(m.idA, m.idB),
                                        std::max(m.idA, m.idB));
        ++nlos_pair_counts_[key];
      }
      const int anchor_count = ++nlos_anchor_counts_[best_anchor];

      ROS_WARN("[SubsetIntegrity] t=%.3f rejecting anchor %d (%d TDOAs: %s). "
               "bad_subsets=%d/%d score=%.2f support=%d/%zu consensus=[%.3f,%.3f,%.3f] "
               "anchor_count=%d",
               t_mid, best_anchor, rejected, rejected_pairs.str().c_str(),
               best_bad, best_seen, best_score, best_support, trials.size(),
               consensus.x(), consensus.y(), consensus.z(), anchor_count);

      last_nlos_rejected_count_ = rejected;
      last_nlos_rejected_pairs_ = std::string("subset_anchor") +
                                  std::to_string(best_anchor) + ":" +
                                  rejected_pairs.str();
      meas = std::move(filtered);
      wls_p = filtered_p;
      const Eigen::VectorXd dynamic_sigmas = computeDynamicTdoaSigmas(meas, wls_p);
      sigmas = filtered_sigmas.size() == dynamic_sigmas.size()
                   ? filtered_sigmas.cwiseMax(dynamic_sigmas)
                   : dynamic_sigmas;
      return true;
    }

    // =========================================================================
    void applyTemporalNlosSigmas(const std::vector<TdoaMeas> &meas,
                                 Eigen::VectorXd             &sigmas) const
    // =========================================================================
    {
      if (!nlos_temporal_filter_enabled_ ||
          sigmas.size() != static_cast<int>(meas.size()))
        return;

      for (int i = 0; i < static_cast<int>(meas.size()); ++i)
      {
        const double score = temporalNlosScore(meas[i]);
        if (score <= nlos_score_threshold_)
          continue;

        const double scale =
            std::min(nlos_score_sigma_scale_max_,
                     1.0 + (score - nlos_score_threshold_));
        sigmas(i) *= scale;
      }
    }

    // =========================================================================
    void updateTemporalNlosScores(double t_mid,
                                  const std::vector<TdoaMeas> &meas,
                                  const Eigen::VectorXd       &base_sigmas,
                                  const Eigen::Vector3d       &p)
    // =========================================================================
    {
      if (!nlos_temporal_filter_enabled_ ||
          base_sigmas.size() != static_cast<int>(meas.size()))
        return;

      auto decay_map = [&](auto &scores)
      {
        for (auto it = scores.begin(); it != scores.end(); )
        {
          it->second *= nlos_score_decay_;
          if (it->second < 0.05)
            it = scores.erase(it);
          else
            ++it;
        }
      };
      decay_map(nlos_pair_scores_);
      decay_map(nlos_anchor_scores_);

      for (int i = 0; i < static_cast<int>(meas.size()); ++i)
      {
        const auto &m = meas[i];
        const double sigma = std::max(base_sigmas(i), 1e-9);
        const double std_abs = std::fabs(tdoaResidualAt(m, p)) / sigma;
        if (std_abs <= nlos_score_gate_sigma_)
          continue;

        const double inc =
            nlos_score_increment_ *
            std::min(3.0, std_abs - nlos_score_gate_sigma_);
        const auto key = std::make_pair(std::min(m.idA, m.idB),
                                        std::max(m.idA, m.idB));
        nlos_pair_scores_[key] += inc;
        nlos_anchor_scores_[m.idA] += 0.5 * inc;
        nlos_anchor_scores_[m.idB] += 0.5 * inc;

        ROS_WARN_THROTTLE(1.0,
            "[NLOS] temporal score update at t=%.3f: pair %d-%d std_res=%.2f "
            "pair_score=%.2f anchor_scores=(%d:%.2f,%d:%.2f)",
            t_mid, m.idA, m.idB, std_abs,
            nlos_pair_scores_[key],
            m.idA, nlos_anchor_scores_[m.idA],
            m.idB, nlos_anchor_scores_[m.idB]);
      }
    }

    // =========================================================================
    bool rejectNlosAnchor(double t_mid,
                          std::vector<TdoaMeas> &meas,
                          Eigen::VectorXd       &sigmas,
                          Eigen::Vector3d       &wls_p)
    // =========================================================================
    {
      if (!nlos_rejection_enabled_ ||
          !nlos_anchor_rejection_enabled_ ||
          sigmas.size() != static_cast<int>(meas.size()) ||
          static_cast<int>(meas.size()) <= nlos_anchor_min_remaining_measurements_)
        return false;

      const Eigen::Vector3d prior_p =
          lc_.initialized
              ? Eigen::Vector3d(lc_.pose.x(), lc_.pose.y(), lc_.pose.z())
              : wls_guess_;
      const double full_chi2 = weightedResidualChi2(meas, sigmas, wls_p);
      const double full_prior_dist = (wls_p - prior_p).norm();

      struct Candidate
      {
        int anchor_id{-1};
        std::vector<int> drop;
        std::vector<TdoaMeas> kept;
        Eigen::VectorXd kept_sigmas;
        Eigen::Vector3d p{Eigen::Vector3d::Zero()};
        double reduced_chi2{0.0};
        double shift{0.0};
        double prior_improvement{0.0};
        double score{0.0};
      };

      Candidate best;
      best.score = -std::numeric_limits<double>::infinity();

      for (int anchor_id = 0; anchor_id < static_cast<int>(anchors_.size()); ++anchor_id)
      {
        std::vector<int> drop;
        std::vector<TdoaMeas> kept;
        kept.reserve(meas.size());
        Eigen::VectorXd kept_sigmas(static_cast<int>(meas.size()));

        int row = 0;
        for (int i = 0; i < static_cast<int>(meas.size()); ++i)
        {
          if (meas[i].idA == anchor_id || meas[i].idB == anchor_id)
          {
            drop.push_back(i);
            continue;
          }
          kept.push_back(meas[i]);
          kept_sigmas(row++) = sigmas(i);
        }

        if (static_cast<int>(drop.size()) < nlos_anchor_min_incident_measurements_ ||
            static_cast<int>(kept.size()) < nlos_anchor_min_remaining_measurements_)
          continue;
        kept_sigmas.conservativeResize(static_cast<int>(kept.size()));

        Eigen::Vector3d reduced_p = wls_p;
        if (!wls_solver_.solveWeighted(kept, anchors_, kept_sigmas, reduced_p))
          continue;

        const double reduced_chi2 = weightedResidualChi2(kept, kept_sigmas, reduced_p);
        const double shift = (reduced_p - wls_p).norm();
        const double prior_improvement =
            full_prior_dist - (reduced_p - prior_p).norm();
        const bool chi2_better =
            std::isfinite(full_chi2) &&
            reduced_chi2 < full_chi2 * nlos_anchor_chi2_ratio_;
        const bool prior_better =
            prior_improvement > nlos_anchor_prior_improvement_gate_;

        if (shift < nlos_anchor_solution_shift_gate_ ||
            (!chi2_better && !prior_better))
          continue;

        const double score =
            prior_improvement + shift + std::max(0.0, full_chi2 - reduced_chi2) * 0.01;
        if (score > best.score)
        {
          best.anchor_id = anchor_id;
          best.drop = std::move(drop);
          best.kept = std::move(kept);
          best.kept_sigmas = kept_sigmas;
          best.p = reduced_p;
          best.reduced_chi2 = reduced_chi2;
          best.shift = shift;
          best.prior_improvement = prior_improvement;
          best.score = score;
        }
      }

      if (best.anchor_id < 0)
        return false;

      for (const int idx : best.drop)
      {
        const auto &m = meas[idx];
        const auto key = std::make_pair(std::min(m.idA, m.idB),
                                        std::max(m.idA, m.idB));
        ++nlos_pair_counts_[key];
      }
      const int anchor_count = ++nlos_anchor_counts_[best.anchor_id];

      std::ostringstream pairs;
      pairs << "anchor" << best.anchor_id << ":";
      for (int j = 0; j < static_cast<int>(best.drop.size()); ++j)
      {
        const auto &m = meas[best.drop[j]];
        if (j > 0) pairs << ";";
        pairs << m.idA << "-" << m.idB;
      }

      ROS_WARN("[NLOS] t=%.3f rejecting anchor %d (%zu incident TDOAs: %s). "
               "solution_shift=%.3f m prior_improvement=%.3f m chi2 %.2f->%.2f "
               "anchor_count=%d",
               t_mid, best.anchor_id, best.drop.size(), pairs.str().c_str(),
               best.shift, best.prior_improvement, full_chi2, best.reduced_chi2,
               anchor_count);

      last_nlos_rejected_count_ = static_cast<int>(best.drop.size());
      last_nlos_rejected_pairs_ = pairs.str();

      meas = std::move(best.kept);
      wls_p = best.p;
      const Eigen::VectorXd dynamic_sigmas = computeDynamicTdoaSigmas(meas, wls_p);
      sigmas = best.kept_sigmas.size() == dynamic_sigmas.size()
                   ? best.kept_sigmas.cwiseMax(dynamic_sigmas)
                   : dynamic_sigmas;
      return true;
    }

    // =========================================================================
    bool rejectNlosMeasurements(double t_mid,
                                std::vector<TdoaMeas> &meas,
                                Eigen::VectorXd       &sigmas,
                                Eigen::Vector3d       &wls_p)
    // =========================================================================
    {
      if (!nlos_rejection_enabled_ ||
          sigmas.size() != static_cast<int>(meas.size()) ||
          static_cast<int>(meas.size()) <= nlos_min_measurements_)
        return false;

      struct Candidate
      {
        int idx;
        double abs_res;
        double std_res;
      };
      std::vector<Candidate> candidates;
      candidates.reserve(meas.size());

      for (int i = 0; i < static_cast<int>(meas.size()); ++i)
      {
        const double sigma = std::max(sigmas(i), 1e-9);
        const double abs_r = std::fabs(tdoaResidualAt(meas[i], wls_p));
        const double std_r = abs_r / sigma;
        if (abs_r > nlos_residual_gate_abs_ &&
            std_r > nlos_residual_gate_sigma_)
          candidates.push_back({i, abs_r, std_r});
      }

      if (candidates.empty()) return false;

      std::sort(candidates.begin(), candidates.end(),
                [](const Candidate &a, const Candidate &b) {
                  if (a.std_res == b.std_res) return a.abs_res > b.abs_res;
                  return a.std_res > b.std_res;
                });

      const int max_drop = std::min(
          nlos_max_rejections_,
          static_cast<int>(meas.size()) - nlos_min_measurements_);
      if (max_drop <= 0) return false;

      std::set<int> drop;
      for (const auto &c : candidates)
      {
        if (static_cast<int>(drop.size()) >= max_drop) break;
        drop.insert(c.idx);
      }

      std::vector<TdoaMeas> filtered;
      filtered.reserve(meas.size() - drop.size());
      Eigen::VectorXd filtered_sigmas(static_cast<int>(meas.size() - drop.size()));
      int row = 0;
      for (int i = 0; i < static_cast<int>(meas.size()); ++i)
      {
        if (drop.count(i)) continue;
        filtered.push_back(meas[i]);
        filtered_sigmas(row++) = sigmas(i);
      }

      Eigen::Vector3d filtered_p = wls_p;
      if (!wls_solver_.solveWeighted(filtered, anchors_, filtered_sigmas, filtered_p))
      {
        ROS_WARN_THROTTLE(1.0,
            "[NLOS] Candidate rejection found %zu bad TDOAs, but reduced WLS failed; keeping full set.",
            drop.size());
        return false;
      }

      for (const int idx : drop)
      {
        const auto &m = meas[idx];
        const double sigma = std::max(sigmas(idx), 1e-9);
        const double res = tdoaResidualAt(m, wls_p);
        const auto key = std::make_pair(std::min(m.idA, m.idB),
                                        std::max(m.idA, m.idB));
        const int pair_count = ++nlos_pair_counts_[key];
        ++nlos_anchor_counts_[m.idA];
        ++nlos_anchor_counts_[m.idB];
        ROS_WARN("[NLOS] t=%.3f rejecting meas[%d] pair %d-%d: residual=%.3f m "
                 "(%.1f sigma, sigma=%.3f). pair_count=%d anchor_counts=(%d:%d,%d:%d)",
                 t_mid, idx, m.idA, m.idB, res, std::fabs(res) / sigma, sigma,
                 pair_count,
                 m.idA, nlos_anchor_counts_[m.idA],
                 m.idB, nlos_anchor_counts_[m.idB]);
      }

      std::ostringstream pairs;
      bool first_pair = true;
      for (const int idx : drop)
      {
        if (!first_pair) pairs << ";";
        first_pair = false;
        pairs << meas[idx].idA << "-" << meas[idx].idB;
      }
      last_nlos_rejected_count_ = static_cast<int>(drop.size());
      last_nlos_rejected_pairs_ = pairs.str();

      meas = std::move(filtered);
      wls_p = filtered_p;
      const Eigen::VectorXd dynamic_sigmas = computeDynamicTdoaSigmas(meas, wls_p);
      sigmas = filtered_sigmas.size() == dynamic_sigmas.size()
                   ? filtered_sigmas.cwiseMax(dynamic_sigmas)
                   : dynamic_sigmas;
      return true;
    }

    // =========================================================================
    bool rejectInconsistentWls(double t_mid,
                               const Eigen::Vector3d &candidate,
                               const IntegrityResult &ir,
                               std::string           &reason) const
    // =========================================================================
    {
      if (!wls_consistency_gate_enabled_ || !last_good_wls_have_)
        return false;

      const Eigen::Vector3d reference =
          lc_.initialized
              ? Eigen::Vector3d(lc_.pose.x(), lc_.pose.y(), lc_.pose.z())
              : last_good_wls_;

      const double ref_dist = (candidate - reference).norm();
      const double z_dist = std::fabs(candidate.z() - reference.z());
      const double step = (candidate - last_good_wls_).norm();

      std::vector<std::string> causes;
      if (wls_reference_gate_ > 0.0 && ref_dist > wls_reference_gate_)
        causes.push_back("reference_dist=" + formatDouble(ref_dist) + "m");
      if (wls_vertical_gate_ > 0.0 && z_dist > wls_vertical_gate_)
        causes.push_back("vertical_dist=" + formatDouble(z_dist) + "m");
      if (wls_max_step_ > 0.0 && step > wls_max_step_)
        causes.push_back("step=" + formatDouble(step) + "m");

      if (causes.empty())
        return false;

      std::ostringstream oss;
      for (size_t i = 0; i < causes.size(); ++i)
      {
        if (i > 0) oss << ",";
        oss << causes[i];
      }
      oss << " chi2=" << std::fixed << std::setprecision(2)
          << ir.chi2_stat << "/" << ir.chi2_threshold;
      reason = oss.str();

      ROS_WARN("[WLSGate] t=%.3f rejecting internally-consistent but implausible WLS "
               "[%.3f,%.3f,%.3f]; ref=[%.3f,%.3f,%.3f] last=[%.3f,%.3f,%.3f] %s",
               t_mid, candidate.x(), candidate.y(), candidate.z(),
               reference.x(), reference.y(), reference.z(),
               last_good_wls_.x(), last_good_wls_.y(), last_good_wls_.z(),
               reason.c_str());
      return true;
    }

    static std::string formatDouble(double v)
    {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(3) << v;
      return oss.str();
    }

    static std::string csvEscape(const std::string &s)
    {
      if (s.find_first_of(",\"\n\r") == std::string::npos)
        return s;

      std::string out = "\"";
      for (const char c : s)
      {
        if (c == '"') out += "\"\"";
        else out += c;
      }
      out += "\"";
      return out;
    }

    // =========================================================================
    void gtCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &msg)
    // =========================================================================
    {
      std::lock_guard<std::mutex> lk(mutex_);
      gt_pos_  = gtsam::Point3(msg->pose.pose.position.x,
                               msg->pose.pose.position.y,
                               msg->pose.pose.position.z);
      gt_have_ = true;

      geometry_msgs::PoseStamped ps;
      ps.header             = msg->header;
      ps.header.frame_id    = odom_frame_;
      ps.pose               = msg->pose.pose;
      gt_path_.header.stamp    = msg->header.stamp;
      gt_path_.header.frame_id = odom_frame_;
      gt_path_.poses.push_back(ps);
      gt_path_pub_.publish(gt_path_);
    }

    // =========================================================================
    void printTdoaWeights(double t_mid,
                          const std::vector<TdoaMeas> &meas,
                          const IntegrityResult       &ir) const
    // =========================================================================
    {
      if (ir.sigmas.size() != meas.size()) return;

      Eigen::MatrixXd W = Eigen::MatrixXd::Zero(
          static_cast<int>(meas.size()), static_cast<int>(meas.size()));

      ROS_INFO("----- [TDOA weights | t=%.3f] -----", t_mid);
      for (int i = 0; i < static_cast<int>(meas.size()); ++i)
      {
        const double sigma  = std::max(ir.sigmas[i], 1e-9);
        const double weight = 1.0 / (sigma * sigma);
        W(i, i) = weight;
        ROS_INFO("  meas[%d] anchors %d-%d: sigma=%.6f m  weight=%.6f",
                 i, meas[i].idA, meas[i].idB, sigma, weight);
      }

      std::ostringstream oss;
      oss << "\n" << W;
      ROS_INFO("  Final W = diag(1/sigma^2):%s", oss.str().c_str());
    }

    // =========================================================================
    void processCycle(const std::vector<TdoaMeas> &meas)
    // =========================================================================
    {
      double t_mid = 0.0;
      for (const auto &m : meas) t_mid += m.t;
      t_mid /= static_cast<double>(meas.size());
      const ros::Time stamp(t_mid);
      last_nlos_rejected_count_ = 0;
      last_nlos_rejected_pairs_ = "none";
      size_t integrity_meas_count = meas.size();
      bool integrity_gate = false;
      bool accept_wls = false;
      bool consistency_reject = false;
      bool wls_held_last = false;

      std::vector<TdoaMeas> wls_meas = meas;

      // ── Initial unweighted WLS solve ────────────────────────────────────────
      Eigen::Vector3d wls_p  = wls_guess_;
      bool            wls_ok = wls_solver_.solve(wls_meas, anchors_, wls_p);
      Eigen::VectorXd tdoa_sigmas;

      // ── Weighted WLS refinement with dynamic sigmas ─────────────────────────
      if (wls_ok && dynamic_tdoa_sigma_enabled_)
      {
        tdoa_sigmas = computeDynamicTdoaSigmas(wls_meas, wls_p);
        applyTemporalNlosSigmas(wls_meas, tdoa_sigmas);

        Eigen::Vector3d weighted_wls_p = wls_p;
        Eigen::VectorXd robust_sigmas = tdoa_sigmas;
        const bool weighted_ok = nlos_robust_weighting_enabled_
            ? wls_solver_.solveRobustWeighted(wls_meas, anchors_, tdoa_sigmas,
                                              weighted_wls_p, nlos_huber_k_,
                                              nlos_sigma_inflation_max_,
                                              nlos_robust_outer_iterations_,
                                              &robust_sigmas)
            : wls_solver_.solveWeighted(wls_meas, anchors_, tdoa_sigmas, weighted_wls_p);
        if (weighted_ok)
        {
          wls_p       = weighted_wls_p;
          // Recompute at refined position, then preserve robust NLOS inflation
          // for integrity and any later hard rejection.
          const Eigen::VectorXd dynamic_sigmas = computeDynamicTdoaSigmas(wls_meas, wls_p);
          updateTemporalNlosScores(t_mid, wls_meas, dynamic_sigmas, wls_p);
          if (nlos_robust_weighting_enabled_ &&
              robust_sigmas.size() == dynamic_sigmas.size())
            tdoa_sigmas = robust_sigmas.cwiseMax(dynamic_sigmas);
          else
            tdoa_sigmas = dynamic_sigmas;
          if (!rejectNlosBySubsetConsensus(t_mid, wls_meas, tdoa_sigmas, wls_p))
            rejectNlosAnchor(t_mid, wls_meas, tdoa_sigmas, wls_p);
          rejectNlosMeasurements(t_mid, wls_meas, tdoa_sigmas, wls_p);
        }
        else
        {
          ROS_WARN_THROTTLE(1.0,
              "[WLS] Dynamic weighted refinement failed; using initial solution.");
        }
      }

      // ── Integrity monitoring ────────────────────────────────────────────────
      IntegrityResult ir;
      if (wls_ok)
      {
        // FIX D: warn when detection power is very low (DOF == 1)
        if ((int)wls_meas.size() == 4)
        {
          ROS_WARN_THROTTLE(5.0,
              "[Integrity] M=4 measurements: DOF=1, chi-squared detection power"
              " is minimal. Add more anchors or measurements per cycle.");
        }

        Eigen::VectorXd anchor_sigmas = computeAnchorSigmas(wls_p);
        ir = integ_monitor_.check(wls_meas, anchors_, wls_p, tdoa_sigmas,
                                  anchor_sigmas);

        std::vector<TdoaMeas> integrity_meas = wls_meas;
        bool excluded_idx_is_original = false;
        if (ir.excluded_idx >= 0)
        {
          const int excluded_idx = ir.excluded_idx;
          const auto excluded_meas = wls_meas[excluded_idx];

          std::vector<TdoaMeas> reduced_meas;
          reduced_meas.reserve(wls_meas.size() - 1);
          Eigen::VectorXd reduced_sigmas;
          if (tdoa_sigmas.size() == static_cast<int>(wls_meas.size()))
            reduced_sigmas.resize(static_cast<int>(wls_meas.size()) - 1);

          int row = 0;
          for (int i = 0; i < static_cast<int>(wls_meas.size()); ++i)
          {
            if (i == excluded_idx) continue;
            reduced_meas.push_back(wls_meas[i]);
            if (reduced_sigmas.size() == static_cast<int>(wls_meas.size()) - 1)
              reduced_sigmas(row++) = tdoa_sigmas(i);
          }

          Eigen::Vector3d recovered_wls_p = wls_p;
          const bool recovered_ok =
              wls_solver_.solveWeighted(reduced_meas, anchors_,
                                        reduced_sigmas, recovered_wls_p);
          if (recovered_ok)
          {
            wls_p = recovered_wls_p;
            if (dynamic_tdoa_sigma_enabled_)
              reduced_sigmas = computeDynamicTdoaSigmas(reduced_meas, wls_p);

            const Eigen::VectorXd reduced_anchor_sigmas = computeAnchorSigmas(wls_p);
            ir = integ_monitor_.check(reduced_meas, anchors_, wls_p,
                                      reduced_sigmas, reduced_anchor_sigmas);
            ir.excluded_idx   = excluded_idx;
            ir.chi2_after_fde = ir.chi2_stat;
            integrity_meas    = reduced_meas;
            excluded_idx_is_original = true;
            integrity_meas_count = integrity_meas.size();

            ROS_WARN("[Integrity] FDE recovered by excluding original meas[%d] "
                     "(pair %d-%d); downstream WLS re-solved on %zu measurements.",
                     excluded_idx, excluded_meas.idA, excluded_meas.idB,
                     reduced_meas.size());
          }
          else
          {
            ir.available = false;
            ROS_WARN("[Integrity] FDE selected original meas[%d] (pair %d-%d), "
                     "but reduced WLS did not converge; solution will be gated out.",
                     excluded_idx, excluded_meas.idA, excluded_meas.idB);
          }
        }

        printIntegrity(t_mid, ir, integrity_meas, excluded_idx_is_original);
        printTdoaWeights(t_mid, integrity_meas, ir);
        integrity_meas_count = integrity_meas.size();

        // FIX C: gate mode
        //   FULL mode  — pass only when chi2 ok AND HPL < HAL AND VPL < VAL
        //   CHI2 mode  — pass when chi2 ok (or FDE succeeded) regardless of PL
        integrity_gate = integrity_gate_full_avail_
            ? ir.available
            : (!ir.fault_detected || (ir.excluded_idx >= 0));

        std::string consistency_reason;
        if (integrity_gate)
        {
          consistency_reject =
              rejectInconsistentWls(t_mid, wls_p, ir, consistency_reason);
        }

        accept_wls = integrity_gate && !consistency_reject;

        if (accept_wls)
        {
          wls_guess_ = wls_p;
          last_good_wls_ = wls_p;
          last_good_wls_t_ = t_mid;
          last_good_wls_have_ = true;
          if (!lc_.initialized)
            lcInitGraph(t_mid, wls_p);
          else
            lcUpdateGraph(t_mid, wls_p);
        }
        else
        {
          if (consistency_reject)
          {
            ir.available = false;
            ir.fault_detected = true;
            last_nlos_rejected_count_ = std::max(last_nlos_rejected_count_, 1);
            last_nlos_rejected_pairs_ =
                std::string("wls_gate:") + consistency_reason;
          }

          ROS_WARN_THROTTLE(1.0,
              "[Integrity] Solution gated out of LC-FGO "
              "(χ²=%.2f/%.2f  HPL=%.3f/%.3f m  VPL=%.3f/%.3f m  avail=%s)",
              ir.chi2_stat,  ir.chi2_threshold,
              ir.hpl,        integ_monitor_.hal,
              ir.vpl,        integ_monitor_.val,
              ir.available ? "YES" : "NO");

          if (wls_hold_last_on_reject_ && last_good_wls_have_)
          {
            wls_p = last_good_wls_;
            wls_held_last = true;
          }
        }

        ROS_INFO("[Integrity] Decision t=%.3f raw_meas=%zu used_meas=%zu "
                 "gate=%s accepted=%s consistency_reject=%s held_last=%s "
                 "nlos_rejected=%d (%s)",
                 t_mid, meas.size(), integrity_meas_count,
                 integrity_gate ? "PASS" : "FAIL",
                 accept_wls ? "YES" : "NO",
                 consistency_reject ? "YES" : "NO",
                 wls_held_last ? "YES" : "NO",
                 last_nlos_rejected_count_,
                 last_nlos_rejected_pairs_.c_str());
      }
      else
      {
        ROS_WARN_THROTTLE(2.0, "[LC-FGO] WLS failed — cycle skipped.");
      }

      printCycleInfo(t_mid, wls_ok, wls_p);

      if (wls_ok)
      {
        publishOdom(wls_odom_pub_, stamp,
                    wls_p.x(), wls_p.y(), wls_p.z(),
                    gtsam::Rot3::identity(), gtsam::Vector3::Zero());
        appendPath(wls_path_, wls_path_pub_, stamp,
                   wls_p.x(), wls_p.y(), wls_p.z(), gtsam::Rot3::identity());
        publishIntegrityOdom(stamp, wls_p, ir);
      }

      if (lc_.initialized)
      {
        const auto &lp = lc_.pose.translation();
        publishOdom(lc_odom_pub_, stamp,
                    lp.x(), lp.y(), lp.z(), lc_.pose.rotation(), lc_.vel);
        appendPath(lc_path_, lc_path_pub_, stamp,
                   lp.x(), lp.y(), lp.z(), lc_.pose.rotation());
        broadcastTf(stamp, lp, lc_.pose.rotation(), base_frame_ + "_lc");
      }

      logRow(t_mid, wls_ok, wls_p, ir,
             meas.size(), integrity_meas_count,
             integrity_gate, accept_wls, consistency_reject, wls_held_last);
    }

    // =========================================================================
    int replayImu(double from_t, double to_t,
                  gtsam::PreintegratedImuMeasurements &preint)
    // =========================================================================
    {
      int    n      = 0;
      double prev_t = from_t;
      for (const auto &imu : imu_buf_)
      {
        if (imu.t <= from_t) continue;
        if (imu.t >  to_t)   break;
        const double dt = imu.t - prev_t;
        if (dt <= 0.0 || dt > 1.0) { prev_t = imu.t; continue; }
        preint.integrateMeasurement(imu.acc, imu.gyr, dt);
        prev_t = imu.t;
        ++n;
      }
      return n;
    }

    // =========================================================================
    gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>
    biasBetween(size_t k_pre, size_t k, double dt)
    // =========================================================================
    {
      const double tau = std::max(dt, 1e-3);
      gtsam::Vector6 bs;
      bs << gtsam::Vector3::Constant(accel_bias_rw_sigma_ * std::sqrt(tau)),
            gtsam::Vector3::Constant(gyro_bias_rw_sigma_  * std::sqrt(tau));
      return gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
          B(k_pre), B(k),
          gtsam::imuBias::ConstantBias(),
          gtsam::noiseModel::Diagonal::Sigmas(bs));
    }

    // =========================================================================
    void lcInitGraph(double t_mid, const Eigen::Vector3d &wls_p)
    // =========================================================================
    {
      lc_.cycle_idx = 0;
      lc_.pose  = gtsam::Pose3(initial_pose_.rotation(),
                               gtsam::Point3(wls_p.x(), wls_p.y(), wls_p.z()));
      lc_.vel   = gtsam::Vector3::Zero();
      lc_.bias  = gtsam::imuBias::ConstantBias();

      gtsam::NonlinearFactorGraph graph;
      gtsam::Values               values;

      gtsam::Vector6 pose_sig;
      pose_sig << prior_pose_sigmas_(0), prior_pose_sigmas_(1), prior_pose_sigmas_(2),
                  wls_sigma_, wls_sigma_, wls_sigma_;
      graph.addPrior(X(0), lc_.pose,
                     gtsam::noiseModel::Diagonal::Sigmas(pose_sig));
      graph.addPrior(V(0), lc_.vel,
                     gtsam::noiseModel::Isotropic::Sigma(3, prior_vel_sigma_));
      graph.addPrior(B(0), lc_.bias,
                     gtsam::noiseModel::Isotropic::Sigma(6, prior_bias_sigma_));

      values.insert(X(0), lc_.pose);
      values.insert(V(0), lc_.vel);
      values.insert(B(0), lc_.bias);

      lc_isam_->update(graph, values);
      lc_isam_->update();

      const gtsam::Values est = lc_isam_->calculateEstimate();
      lc_.pose = est.at<gtsam::Pose3>(X(0));
      lc_.vel  = est.at<gtsam::Vector3>(V(0));
      lc_.bias = est.at<gtsam::imuBias::ConstantBias>(B(0));

      lc_preint_->resetIntegrationAndSetBias(lc_.bias);
      lc_.initialized  = true;
      lc_.last_cycle_t = t_mid;

      ROS_INFO("[LC-FGO] Init  t=%.3f  wls=[%.3f,%.3f,%.3f]  pos=[%.3f,%.3f,%.3f]",
               t_mid, wls_p.x(), wls_p.y(), wls_p.z(),
               lc_.pose.x(), lc_.pose.y(), lc_.pose.z());
    }

    // =========================================================================
    void lcUpdateGraph(double t_mid, const Eigen::Vector3d &wls_p)
    // =========================================================================
    {
      const Eigen::Vector3d lc_p(lc_.pose.x(), lc_.pose.y(), lc_.pose.z());
      const double lc_wls_gap = (lc_p - wls_p).norm();
      if (lc_wls_reset_threshold_ > 0.0 && lc_wls_gap > lc_wls_reset_threshold_)
      {
        ROS_WARN("[LC-FGO] LC/WLS divergence %.3f m exceeds %.3f m at t=%.3f; "
                 "resetting LC from integrity-gated WLS.",
                 lc_wls_gap, lc_wls_reset_threshold_, t_mid);
        resetLcFromWls(t_mid, wls_p, "LC/WLS divergence");
        return;
      }

      lc_preint_->resetIntegrationAndSetBias(lc_.bias);
      const int n_imu = replayImu(lc_.last_cycle_t, t_mid, *lc_preint_);

      if (n_imu < min_imu_per_cycle_)
      {
        ROS_WARN_THROTTLE(1.0,
            "[LC-FGO] %d IMU (need %d); using WLS-only fallback.",
            n_imu, min_imu_per_cycle_);

        ++lc_.cycle_idx;
        const size_t k     = lc_.cycle_idx;

        gtsam::NonlinearFactorGraph graph;
        gtsam::Values               values;

        const gtsam::Pose3 wls_pose(
            lc_.pose.rotation(),
            gtsam::Point3(wls_p.x(), wls_p.y(), wls_p.z()));
        const gtsam::Vector3 zero_vel = gtsam::Vector3::Zero();
        gtsam::Vector6 lc_sig;
        lc_sig << prior_pose_sigmas_(0), prior_pose_sigmas_(1), prior_pose_sigmas_(2),
                  wls_sigma_, wls_sigma_, wls_sigma_;
        graph.addPrior(X(k), wls_pose,
                       gtsam::noiseModel::Diagonal::Sigmas(lc_sig));
        graph.addPrior(V(k), zero_vel,
                       gtsam::noiseModel::Isotropic::Sigma(3, lc_update_vel_sigma_));
        graph.addPrior(B(k), lc_.bias,
                       gtsam::noiseModel::Isotropic::Sigma(6, lc_update_bias_sigma_));

        values.insert(X(k), wls_pose);
        values.insert(V(k), zero_vel);
        values.insert(B(k), lc_.bias);

        try
        {
          lc_isam_->update(graph, values);
          lc_isam_->update();

          const gtsam::Values est = lc_isam_->calculateEstimate();
          lc_.pose = est.at<gtsam::Pose3>(X(k));
          lc_.vel  = est.at<gtsam::Vector3>(V(k));
          lc_.bias = est.at<gtsam::imuBias::ConstantBias>(B(k));
        }
        catch (const std::exception &e)
        {
          ROS_ERROR("[LC-FGO] WLS-only fallback failed at k=%zu t=%.3f: %s",
                    k, t_mid, e.what());
          resetLcFromWls(t_mid, wls_p, "WLS-only fallback failure");
        }

        lc_.last_cycle_t = t_mid;
        lc_preint_->resetIntegrationAndSetBias(lc_.bias);
        return;
      }

      ++lc_.cycle_idx;
      const size_t k     = lc_.cycle_idx;
      const size_t k_pre = lc_.cycle_idx - 1;

      const gtsam::NavState pred_nav =
          lc_preint_->predict(gtsam::NavState(lc_.pose, lc_.vel), lc_.bias);

      gtsam::NonlinearFactorGraph graph;
      gtsam::Values               values;

      graph.add(gtsam::ImuFactor(
          X(k_pre), V(k_pre), X(k), V(k), B(k_pre), *lc_preint_));
      graph.add(biasBetween(k_pre, k, t_mid - lc_.last_cycle_t));

      const gtsam::Pose3 wls_pose(
          pred_nav.pose().rotation(),
          gtsam::Point3(wls_p.x(), wls_p.y(), wls_p.z()));
      gtsam::Vector6 lc_sig;
      lc_sig << prior_pose_sigmas_(0), prior_pose_sigmas_(1), prior_pose_sigmas_(2),
                wls_sigma_, wls_sigma_, wls_sigma_;
      graph.addPrior(X(k), wls_pose,
                     gtsam::noiseModel::Diagonal::Sigmas(lc_sig));
      graph.addPrior(V(k), pred_nav.velocity(),
                     gtsam::noiseModel::Isotropic::Sigma(3, lc_update_vel_sigma_));
      graph.addPrior(B(k), lc_.bias,
                     gtsam::noiseModel::Isotropic::Sigma(6, lc_update_bias_sigma_));

      values.insert(X(k), pred_nav.pose());
      values.insert(V(k), pred_nav.velocity());
      values.insert(B(k), lc_.bias);

      try
      {
        lc_isam_->update(graph, values);
        lc_isam_->update();

        const gtsam::Values est = lc_isam_->calculateEstimate();
        lc_.pose = est.at<gtsam::Pose3>(X(k));
        lc_.vel  = est.at<gtsam::Vector3>(V(k));
        lc_.bias = est.at<gtsam::imuBias::ConstantBias>(B(k));
      }
      catch (const std::exception &e)
      {
        ROS_ERROR("[LC-FGO] ISAM update failed at k=%zu t=%.3f: %s",
                  k, t_mid, e.what());
        resetLcFromWls(t_mid, wls_p, "ISAM update failure");
        return;
      }

      lc_preint_->resetIntegrationAndSetBias(lc_.bias);
      lc_.last_cycle_t = t_mid;
    }

    // =========================================================================
    void resetLcFromWls(double t_mid, const Eigen::Vector3d &wls_p,
                        const std::string &reason)
    // =========================================================================
    {
      ROS_WARN("[LC-FGO] Resetting LC graph from WLS at t=%.3f after %s.",
               t_mid, reason.c_str());

      lc_isam_ = makeIsam2();
      lc_.initialized = false;
      lc_preint_->resetIntegrationAndSetBias(lc_.bias);
      lcInitGraph(t_mid, wls_p);
    }

    // =========================================================================
    boost::shared_ptr<gtsam::ISAM2> makeIsam2() const
    // =========================================================================
    {
      gtsam::ISAM2Params isam_p;
      isam_p.relinearizeThreshold = 0.1;
      isam_p.relinearizeSkip      = 1;
      isam_p.factorization        = gtsam::ISAM2Params::CHOLESKY;
      return boost::make_shared<gtsam::ISAM2>(isam_p);
    }

    // =========================================================================
    void printIntegrity(double t_mid, const IntegrityResult &ir,
                        const std::vector<TdoaMeas> &meas,
                        bool excluded_idx_is_original = false) const
    // =========================================================================
    {
      ROS_INFO("----- [Integrity | t=%.3f] -----", t_mid);
      ROS_INFO("  Ring closure : loops=%d  worst=%.4f m  %s",
               ir.ring_closed_loops, ir.ring_sum,
               ir.ring_ok ? "OK" : "WARN");
      ROS_INFO("  Chi2 test    : T=%.3f  threshold=%.3f  dof=%d  → %s",
               ir.chi2_stat, ir.chi2_threshold, ir.dof,
               ir.fault_detected ? "FAULT" : "OK");
      if (!std::isnan(ir.hpl))
        ROS_INFO("  HPL=%.4f m  VPL=%.4f m  (HAL=%.2f VAL=%.2f)  avail=%s",
                 ir.hpl, ir.vpl,
                 integ_monitor_.hal, integ_monitor_.val,
                 ir.available ? "YES" : "NO");
      if (ir.excluded_idx >= 0)
      {
        if (!excluded_idx_is_original &&
            ir.excluded_idx < static_cast<int>(meas.size()) &&
            meas.size() == ir.std_resid.size())
        {
          const auto &mx = meas[ir.excluded_idx];
          ROS_WARN("  FDE excluded meas[%d] (pair %d-%d)  chi2_after=%.3f",
                   ir.excluded_idx, mx.idA, mx.idB, ir.chi2_after_fde);
        }
        else
        {
          ROS_WARN("  FDE excluded original meas[%d]  chi2_after=%.3f",
                   ir.excluded_idx, ir.chi2_after_fde);
        }
      }
      for (int i = 0; i < (int)ir.std_resid.size(); ++i)
      {
        if (i >= static_cast<int>(meas.size())) break;
        const double sigma = (i < (int)ir.sigmas.size())
                                 ? ir.sigmas[i]
                                 : std::numeric_limits<double>::quiet_NaN();
        ROS_INFO("    meas[%d] (%d-%d): res=%.4f m  sig=%.4f m  res/sig=%.2f%s",
                 i, meas[i].idA, meas[i].idB,
                 ir.residuals[i], sigma, ir.std_resid[i],
                 (!excluded_idx_is_original &&
                  meas.size() == ir.std_resid.size() && i == ir.excluded_idx)
                     ? "  ← excluded" : "");
      }
    }

    // =========================================================================
    void publishIntegrityOdom(const ros::Time      &stamp,
                              const Eigen::Vector3d &wls_p,
                              const IntegrityResult &ir)
    // =========================================================================
    {
      nav_msgs::Odometry msg;
      msg.header.stamp     = stamp;
      msg.header.frame_id  = odom_frame_;
      msg.child_frame_id   = base_frame_ + "_integ";
      msg.pose.pose.position.x    = wls_p.x();
      msg.pose.pose.position.y    = wls_p.y();
      msg.pose.pose.position.z    = wls_p.z();
      msg.pose.pose.orientation.w = 1.0;

      // Covariance diagonal encodes HPL²/VPL² for downstream visualisation.
      // A receiving node should treat sqrt(cov[0,0]) as the 1-sigma horizontal
      // bound, not as a standard Gaussian sigma.
      const double hpl2 = std::isnan(ir.hpl) ? 9999.0 : ir.hpl * ir.hpl;
      const double vpl2 = std::isnan(ir.vpl) ? 9999.0 : ir.vpl * ir.vpl;
      msg.pose.covariance[0]  = hpl2;   // xx
      msg.pose.covariance[7]  = hpl2;   // yy
      msg.pose.covariance[14] = vpl2;   // zz

      // Twist encodes scalar integrity flags for easy plotting / monitoring
      msg.twist.twist.linear.x = ir.fault_detected ? 1.0 : 0.0;  // fault flag
      msg.twist.twist.linear.y = ir.chi2_stat;                    // chi² value
      msg.twist.twist.linear.z = ir.ring_sum;                     // ring residual

      integ_odom_pub_.publish(msg);
    }

    // =========================================================================
    void printCycleInfo(double t_mid, bool wls_ok,
                        const Eigen::Vector3d &wls_p) const
    // =========================================================================
    {
      const auto err3 = [&](const gtsam::Point3 &p) -> double {
        return gt_have_
            ? std::sqrt(std::pow(p.x() - gt_pos_.x(), 2) +
                        std::pow(p.y() - gt_pos_.y(), 2) +
                        std::pow(p.z() - gt_pos_.z(), 2))
            : -1.0;
      };
      ROS_INFO("=================================================");
      ROS_INFO("[LC=%4zu | t=%.3f s]", lc_.cycle_idx, t_mid);
      if (wls_ok)
        ROS_INFO("  WLS    : [%7.3f,%7.3f,%7.3f]", wls_p.x(), wls_p.y(), wls_p.z());
      else
        ROS_INFO("  WLS    : DID NOT CONVERGE");
      if (lc_.initialized)
      {
        const auto &lp  = lc_.pose.translation();
        const auto  rpy = lc_.pose.rotation().rpy();
        ROS_INFO("  LC-FGO : [%7.3f,%7.3f,%7.3f]  rpy=[%.1f,%.1f,%.1f]deg",
                 lp.x(), lp.y(), lp.z(),
                 rpy.x() * 180.0 / M_PI,
                 rpy.y() * 180.0 / M_PI,
                 rpy.z() * 180.0 / M_PI);
        ROS_INFO("           vel=[%.3f,%.3f,%.3f]  ba=[%.4f,%.4f,%.4f]",
                 lc_.vel.x(), lc_.vel.y(), lc_.vel.z(),
                 lc_.bias.accelerometer().x(),
                 lc_.bias.accelerometer().y(),
                 lc_.bias.accelerometer().z());
      }
      if (gt_have_)
      {
        ROS_INFO("  GT     : [%7.3f,%7.3f,%7.3f]", gt_pos_.x(), gt_pos_.y(), gt_pos_.z());
        if (wls_ok)
          ROS_INFO("  |WLS-GT| = %.4f m",
                   (wls_p - Eigen::Vector3d(gt_pos_.x(), gt_pos_.y(), gt_pos_.z())).norm());
        if (lc_.initialized)
          ROS_INFO("  |LC -GT| = %.4f m", err3(lc_.pose.translation()));
      }
      ROS_INFO("=================================================");
    }

    // =========================================================================
    void publishOdom(ros::Publisher      &pub,
                     const ros::Time     &stamp,
                     double x, double y, double z,
                     const gtsam::Rot3   &rot,
                     const gtsam::Vector3 &vel)
    // =========================================================================
    {
      const Eigen::Quaterniond q(rot.matrix());
      nav_msgs::Odometry odom;
      odom.header.stamp            = stamp;
      odom.header.frame_id         = odom_frame_;
      odom.child_frame_id          = base_frame_;
      odom.pose.pose.position.x    = x;
      odom.pose.pose.position.y    = y;
      odom.pose.pose.position.z    = z;
      odom.pose.pose.orientation.x = q.x();
      odom.pose.pose.orientation.y = q.y();
      odom.pose.pose.orientation.z = q.z();
      odom.pose.pose.orientation.w = q.w();
      odom.twist.twist.linear.x    = vel.x();
      odom.twist.twist.linear.y    = vel.y();
      odom.twist.twist.linear.z    = vel.z();
      pub.publish(odom);
    }

    // =========================================================================
    void appendPath(nav_msgs::Path  &path,
                    ros::Publisher  &pub,
                    const ros::Time &stamp,
                    double x, double y, double z,
                    const gtsam::Rot3 &rot)
    // =========================================================================
    {
      const Eigen::Quaterniond q(rot.matrix());
      geometry_msgs::PoseStamped ps;
      ps.header.stamp              = stamp;
      ps.header.frame_id           = odom_frame_;
      ps.pose.position.x           = x;
      ps.pose.position.y           = y;
      ps.pose.position.z           = z;
      ps.pose.orientation.x        = q.x();
      ps.pose.orientation.y        = q.y();
      ps.pose.orientation.z        = q.z();
      ps.pose.orientation.w        = q.w();
      path.header.stamp            = stamp;
      path.header.frame_id         = odom_frame_;
      path.poses.push_back(ps);
      pub.publish(path);
    }

    // =========================================================================
    void broadcastTf(const ros::Time    &stamp,
                     const gtsam::Point3 &pos,
                     const gtsam::Rot3  &rot,
                     const std::string  &child)
    // =========================================================================
    {
      const Eigen::Quaterniond q(rot.matrix());
      geometry_msgs::TransformStamped tf;
      tf.header.stamp            = stamp;
      tf.header.frame_id         = odom_frame_;
      tf.child_frame_id          = child;
      tf.transform.translation.x = pos.x();
      tf.transform.translation.y = pos.y();
      tf.transform.translation.z = pos.z();
      tf.transform.rotation.x    = q.x();
      tf.transform.rotation.y    = q.y();
      tf.transform.rotation.z    = q.z();
      tf.transform.rotation.w    = q.w();
      tf_broadcaster_.sendTransform(tf);
    }

    // =========================================================================
    void publishAnchorMarkers()
    // =========================================================================
    {
      visualization_msgs::MarkerArray markers;
      const ros::Time stamp = ros::Time::now();

      for (int i = 0; i < static_cast<int>(anchors_.size()); ++i)
      {
        const auto &a = anchors_[i];

        visualization_msgs::Marker sphere;
        sphere.header.frame_id = odom_frame_;
        sphere.header.stamp    = stamp;
        sphere.ns              = "uwb_anchors";
        sphere.id              = i;
        sphere.type            = visualization_msgs::Marker::SPHERE;
        sphere.action          = visualization_msgs::Marker::ADD;
        sphere.pose.position.x = a.x();
        sphere.pose.position.y = a.y();
        sphere.pose.position.z = a.z();
        sphere.pose.orientation.w = 1.0;
        sphere.scale.x = 0.25;
        sphere.scale.y = 0.25;
        sphere.scale.z = 0.25;
        sphere.color.r = 0.95f;
        sphere.color.g = 0.20f;
        sphere.color.b = 0.10f;
        sphere.color.a = 0.90f;
        markers.markers.push_back(sphere);

        visualization_msgs::Marker label;
        label.header.frame_id = odom_frame_;
        label.header.stamp    = stamp;
        label.ns              = "uwb_anchor_ids";
        label.id              = i;
        label.type            = visualization_msgs::Marker::TEXT_VIEW_FACING;
        label.action          = visualization_msgs::Marker::ADD;
        label.pose.position.x = a.x();
        label.pose.position.y = a.y();
        label.pose.position.z = a.z() + 0.35;
        label.pose.orientation.w = 1.0;
        label.scale.z = 0.30;
        label.color.r = 1.0f;
        label.color.g = 1.0f;
        label.color.b = 1.0f;
        label.color.a = 1.0f;
        label.text = std::to_string(i);
        markers.markers.push_back(label);
      }

      anchor_marker_pub_.publish(markers);
    }

    // ── ROS handles ─────────────────────────────────────────────────────────
    ros::NodeHandle nh_, pnh_;
    ros::Subscriber imu_sub_, uwb_sub_, gt_sub_;
    ros::Publisher  wls_odom_pub_, lc_odom_pub_;
    ros::Publisher  wls_path_pub_, lc_path_pub_, gt_path_pub_;
    ros::Publisher  integ_odom_pub_;
    ros::Publisher  anchor_marker_pub_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    std::mutex mutex_;

    // ── Anchors ──────────────────────────────────────────────────────────────
    std::vector<gtsam::Point3> anchors_;
    std::string odom_frame_{"map"}, base_frame_{"base_link"};

    // ── Cycle accumulator ────────────────────────────────────────────────────
    std::vector<TdoaMeas>              current_cycle_;
    std::set<std::pair<int, int>>      cycle_pairs_;
    int    measurements_per_cycle_{8};
    int    min_imu_per_cycle_{2};
    double cycle_timeout_{0.1};

    // ── WLS ──────────────────────────────────────────────────────────────────
    TdoaWlsSolver    wls_solver_;
    Eigen::Vector3d  wls_guess_{0, 0, 1};

    // ── IMU buffer ───────────────────────────────────────────────────────────
    std::deque<ImuMeas> imu_buf_;
    double last_imu_t_{-1.0};

    // ── IMU noise params ─────────────────────────────────────────────────────
    double accel_scale_{1.0};
    bool   imu_accel_in_g_{true};
    bool   gyro_is_degrees_{false};
    double accel_sigma_{0.003},        gyro_sigma_{0.0003};
    double accel_bias_rw_sigma_{0.0003}, gyro_bias_rw_sigma_{0.00003};
    double integration_sigma_{1e-6},   gravity_mag_{9.80665};

    // ── WLS/FGO params ───────────────────────────────────────────────────────
    double wls_sigma_{0.15};
    gtsam::Vector6 prior_pose_sigmas_;
    double prior_vel_sigma_{0.1},      prior_bias_sigma_{0.01};
    double lc_update_vel_sigma_{2.0},  lc_update_bias_sigma_{0.5};
    double lc_wls_reset_threshold_{1.0};
    bool   wls_consistency_gate_enabled_{true};
    double wls_reference_gate_{0.75};
    double wls_vertical_gate_{0.55};
    double wls_max_step_{0.85};
    bool   wls_hold_last_on_reject_{true};
    bool   last_good_wls_have_{false};
    double last_good_wls_t_{0.0};
    Eigen::Vector3d last_good_wls_{0, 0, 1};

    // ── Dynamic sigma model (FIX A: new defaults) ────────────────────────────
    bool   dynamic_tdoa_sigma_enabled_{true};
    double tdoa_range_sigma_base_{0.05};       // was 0.10
    double tdoa_range_sigma_per_meter_{0.005}; // was 0.02
    double tdoa_pair_sigma_min_{1e-3};
    double tdoa_pair_sigma_max_{0.50};         // was 10.0

    // ── NLOS residual rejection ───────────────────────────────────────────────
    bool   nlos_rejection_enabled_{true};
    bool   nlos_robust_weighting_enabled_{true};
    double nlos_huber_k_{1.5};
    double nlos_sigma_inflation_max_{8.0};
    int    nlos_robust_outer_iterations_{4};
    bool   subset_consensus_enabled_{false};
    int    subset_consensus_trials_{80};
    int    subset_consensus_anchor_count_{0};
    int    subset_consensus_min_measurements_{3};
    double subset_consensus_cluster_gate_{0.25};
    double subset_consensus_reject_score_{0.55};
    int    subset_consensus_min_support_{3};
    bool   nlos_anchor_rejection_enabled_{true};
    int    nlos_anchor_min_incident_measurements_{2};
    int    nlos_anchor_min_remaining_measurements_{6};
    double nlos_anchor_solution_shift_gate_{0.25};
    double nlos_anchor_prior_improvement_gate_{0.15};
    double nlos_anchor_chi2_ratio_{0.85};
    bool   nlos_temporal_filter_enabled_{true};
    double nlos_score_decay_{0.85};
    double nlos_score_gate_sigma_{2.5};
    double nlos_score_increment_{1.0};
    double nlos_score_threshold_{1.0};
    double nlos_score_sigma_scale_max_{5.0};
    double nlos_residual_gate_sigma_{2.8};
    double nlos_residual_gate_abs_{0.35};
    int    nlos_max_rejections_{1};
    int    nlos_min_measurements_{6};
    std::map<std::pair<int, int>, int> nlos_pair_counts_;
    std::map<int, int>                 nlos_anchor_counts_;
    std::map<std::pair<int, int>, double> nlos_pair_scores_;
    std::map<int, double>                 nlos_anchor_scores_;
    int         last_nlos_rejected_count_{0};
    std::string last_nlos_rejected_pairs_{"none"};

    // ── Integrity gate mode (FIX C) ──────────────────────────────────────────
    bool integrity_gate_full_avail_{true};  // true: gate on HPL+VPL+chi2

    // ── Poses / state ────────────────────────────────────────────────────────
    gtsam::Pose3 initial_pose_;
    boost::shared_ptr<gtsam::PreintegrationParams>         imu_params_;
    FgoState                                               lc_;
    boost::shared_ptr<gtsam::PreintegratedImuMeasurements> lc_preint_;
    boost::shared_ptr<gtsam::ISAM2>                        lc_isam_;

    // ── Ground truth ─────────────────────────────────────────────────────────
    bool           gt_have_{false};
    gtsam::Point3  gt_pos_{gtsam::Point3::Zero()};

    // ── Paths & logging ──────────────────────────────────────────────────────
    nav_msgs::Path wls_path_, lc_path_, gt_path_;
    std::string    traj_log_path_{"/tmp/uwb_imu_trajectory.csv"};
    std::ofstream  traj_log_;

    // ── Integrity monitor ────────────────────────────────────────────────────
    TdoaIntegrityMonitor integ_monitor_;
  };

}  // namespace uwb_imu_fusion

int main(int argc, char **argv)
{
#ifdef UWB_SUBSET_CONSENSUS_INTEGRITY
  ros::init(argc, argv, "subset_integrity_uwb_node");
#else
  ros::init(argc, argv, "integrity_uwb_node");
#endif
  ros::NodeHandle nh, pnh("~");
  uwb_imu_fusion::UwbImuLcWlsFusionNode node(nh, pnh);
  ros::AsyncSpinner spinner(2);
  spinner.start();
  ros::waitForShutdown();
  return 0;
}
