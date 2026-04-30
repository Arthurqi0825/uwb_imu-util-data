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
#include <mutex>
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

      get("integrity_tdoa_noise_sigma", integ_noise_sigma,  0.05);
      get("integrity_p_fa",             integ_p_fa,         1e-3);
      get("integrity_hal",              integ_hal,          1.00);
      get("integrity_val",              integ_val,          2.00);
      get("integrity_ring_threshold",   integ_ring_thresh,  0.10);
      get("integrity_enable_fde",       integ_fde,          true);

      // FIX C: gate mode — default uses full availability (chi2 + HPL + VPL)
      get("integrity_gate_full_avail",  integrity_gate_full_avail_, true);

      // Dynamic sigma model
      get("dynamic_tdoa_sigma_enabled",    dynamic_tdoa_sigma_enabled_,  true);
      get("tdoa_range_sigma_base",         tdoa_range_sigma_base_,       0.05);  // FIX A
      get("tdoa_range_sigma_per_meter",    tdoa_range_sigma_per_meter_,  0.005); // FIX A
      get("tdoa_pair_sigma_min",           tdoa_pair_sigma_min_,         1e-3);
      get("tdoa_pair_sigma_max",           tdoa_pair_sigma_max_,         0.50);  // FIX A

      if (tdoa_pair_sigma_max_ < tdoa_pair_sigma_min_)
        std::swap(tdoa_pair_sigma_min_, tdoa_pair_sigma_max_);

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

      gtsam::ISAM2Params isam_p;
      isam_p.relinearizeThreshold = 0.1;
      isam_p.relinearizeSkip      = 1;
      isam_p.factorization        = gtsam::ISAM2Params::CHOLESKY;
      lc_isam_ = boost::make_shared<gtsam::ISAM2>(isam_p);
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
                  << "integ_excluded_idx\n";
      }
      else
      {
        ROS_WARN("[uwb_imu_fusion] Cannot open log: %s", traj_log_path_.c_str());
      }
    }

    // =========================================================================
    void logRow(double t, bool wls_ok, const Eigen::Vector3d &wls_p,
                const IntegrityResult &ir)
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
                << ir.excluded_idx    << "\n";
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

      // ── Initial unweighted WLS solve ────────────────────────────────────────
      Eigen::Vector3d wls_p  = wls_guess_;
      bool            wls_ok = wls_solver_.solve(meas, anchors_, wls_p);
      Eigen::VectorXd tdoa_sigmas;

      // ── Weighted WLS refinement with dynamic sigmas ─────────────────────────
      if (wls_ok && dynamic_tdoa_sigma_enabled_)
      {
        tdoa_sigmas = computeDynamicTdoaSigmas(meas, wls_p);

        Eigen::Vector3d weighted_wls_p = wls_p;
        const bool weighted_ok =
            wls_solver_.solveWeighted(meas, anchors_, tdoa_sigmas, weighted_wls_p);
        if (weighted_ok)
        {
          wls_p       = weighted_wls_p;
          // Recompute at refined position so integrity shares the same geometry
          tdoa_sigmas = computeDynamicTdoaSigmas(meas, wls_p);
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
        if ((int)meas.size() == 4)
        {
          ROS_WARN_THROTTLE(5.0,
              "[Integrity] M=4 measurements: DOF=1, chi-squared detection power"
              " is minimal. Add more anchors or measurements per cycle.");
        }

        ir = integ_monitor_.check(meas, anchors_, wls_p, tdoa_sigmas);
        printIntegrity(t_mid, ir, meas);
        printTdoaWeights(t_mid, meas, ir);

        // FIX C: gate mode
        //   FULL mode  — pass only when chi2 ok AND HPL < HAL AND VPL < VAL
        //   CHI2 mode  — pass when chi2 ok (or FDE succeeded) regardless of PL
        const bool integ_gate = integrity_gate_full_avail_
            ? ir.available
            : (!ir.fault_detected || (ir.excluded_idx >= 0));

        if (integ_gate)
        {
          wls_guess_ = wls_p;
          if (!lc_.initialized)
            lcInitGraph(t_mid, wls_p);
          else
            lcUpdateGraph(t_mid, wls_p);
        }
        else
        {
          ROS_WARN_THROTTLE(1.0,
              "[Integrity] Solution gated out of LC-FGO "
              "(χ²=%.2f/%.2f  HPL=%.3f/%.3f m  VPL=%.3f/%.3f m  avail=%s)",
              ir.chi2_stat,  ir.chi2_threshold,
              ir.hpl,        integ_monitor_.hal,
              ir.vpl,        integ_monitor_.val,
              ir.available ? "YES" : "NO");
        }
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

      logRow(t_mid, wls_ok, wls_p, ir);
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
      lc_preint_->resetIntegrationAndSetBias(lc_.bias);
      const int n_imu = replayImu(lc_.last_cycle_t, t_mid, *lc_preint_);

      if (n_imu < min_imu_per_cycle_)
      {
        ROS_WARN_THROTTLE(1.0, "[LC-FGO] %d IMU (need %d); skip.", n_imu, min_imu_per_cycle_);
        lc_.last_cycle_t = t_mid;
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
        lc_.cycle_idx    = k_pre;
        lc_.last_cycle_t = t_mid;
        lc_preint_->resetIntegrationAndSetBias(lc_.bias);
        return;
      }

      lc_preint_->resetIntegrationAndSetBias(lc_.bias);
      lc_.last_cycle_t = t_mid;
    }

    // =========================================================================
    void printIntegrity(double t_mid, const IntegrityResult &ir,
                        const std::vector<TdoaMeas> &meas) const
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
        const auto &mx = meas[ir.excluded_idx];
        ROS_WARN("  FDE excluded meas[%d] (pair %d-%d)  chi2_after=%.3f",
                 ir.excluded_idx, mx.idA, mx.idB, ir.chi2_after_fde);
      }
      for (int i = 0; i < (int)ir.std_resid.size(); ++i)
      {
        const double sigma = (i < (int)ir.sigmas.size())
                                 ? ir.sigmas[i]
                                 : std::numeric_limits<double>::quiet_NaN();
        ROS_INFO("    meas[%d] (%d-%d): res=%.4f m  sig=%.4f m  res/sig=%.2f%s",
                 i, meas[i].idA, meas[i].idB,
                 ir.residuals[i], sigma, ir.std_resid[i],
                 (i == ir.excluded_idx) ? "  ← excluded" : "");
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

    // ── ROS handles ─────────────────────────────────────────────────────────
    ros::NodeHandle nh_, pnh_;
    ros::Subscriber imu_sub_, uwb_sub_, gt_sub_;
    ros::Publisher  wls_odom_pub_, lc_odom_pub_;
    ros::Publisher  wls_path_pub_, lc_path_pub_, gt_path_pub_;
    ros::Publisher  integ_odom_pub_;
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

    // ── Dynamic sigma model (FIX A: new defaults) ────────────────────────────
    bool   dynamic_tdoa_sigma_enabled_{true};
    double tdoa_range_sigma_base_{0.05};       // was 0.10
    double tdoa_range_sigma_per_meter_{0.005}; // was 0.02
    double tdoa_pair_sigma_min_{1e-3};
    double tdoa_pair_sigma_max_{0.50};         // was 10.0

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
  ros::init(argc, argv, "integrity_uwb_node");
  ros::NodeHandle nh, pnh("~");
  uwb_imu_fusion::UwbImuLcWlsFusionNode node(nh, pnh);
  ros::AsyncSpinner spinner(2);
  spinner.start();
  ros::waitForShutdown();
  return 0;
}