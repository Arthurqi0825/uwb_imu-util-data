// =============================================================================
// uwb_imu_lc_wls_fusion_node.cpp  —  WLS + Loosely Coupled IMU FGO
//
// PIPELINE
// ─────────────────────────────────────────────────────────────────────────────
//  UWB stream → cycle accumulator (validated, unchanged)
//                    │
//                    ├─► WLS solver  ──────────────────────────────────────────►  uwb_wls / uwb_wls_path
//                    │
//                    └─► LC-FGO  (WLS position prior + ImuFactor + BiasBetween) ► lc_fusion / lc_fusion_path
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

#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

// ── Boost math for chi-squared quantile ──────────────────────────────────────
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

namespace uwb_imu_fusion {

struct ImuMeas {
  double t;
  gtsam::Vector3 acc;
  gtsam::Vector3 gyr;
};

struct TdoaMeas {
  double t;
  int    idA, idB;
  double tdoa;
};

class TdoaWlsSolver {
 public:
  explicit TdoaWlsSolver(int max_iter = 50, double tol = 1e-7)
      : max_iter_(max_iter), tol_(tol) {}

  bool solve(const std::vector<TdoaMeas>& meas,
             const std::vector<gtsam::Point3>& anchors,
             Eigen::Vector3d& p) const {
    if ((int)meas.size() < 3) return false;
    for (int iter = 0; iter < max_iter_; ++iter) {
      const int n = static_cast<int>(meas.size());
      Eigen::MatrixXd J(n, 3);
      Eigen::VectorXd r(n);
      for (int i = 0; i < n; ++i) {
        const auto& m = meas[i];
        const Eigen::Vector3d aA(anchors[m.idA].x(), anchors[m.idA].y(), anchors[m.idA].z());
        const Eigen::Vector3d aB(anchors[m.idB].x(), anchors[m.idB].y(), anchors[m.idB].z());
        const Eigen::Vector3d dA = p - aA;
        const Eigen::Vector3d dB = p - aB;
        const double rA = std::max(dA.norm(), 1e-9);
        const double rB = std::max(dB.norm(), 1e-9);
        r(i) = (rB - rA) - m.tdoa;
        J.row(i) = (dB / rB - dA / rA).transpose();
      }
      const Eigen::Vector3d dp =
          -(J.transpose() * J).ldlt().solve(J.transpose() * r);
      p += dp;
      if (dp.norm() < tol_) {
        ROS_WARN("[TdoaWlsSolver] Converged in %d iterations (residual norm=%.3f), dp norm=%.3f",
                 iter + 1, r.norm(), dp.norm());
        return true;
      }
    }
    return false;
  }

 private:
  int max_iter_;
  double tol_;
};

// =============================================================================
// IntegrityResult  —  output bundle from one integrity epoch
// =============================================================================
struct IntegrityResult {
  // ── Fault detection ──────────────────────────────────────────────────────
  double  chi2_stat{0.0};       // weighted SSR test statistic T = rᵀ Σ⁻¹ r
  double  chi2_threshold{0.0};  // threshold at configured P_FA
  int     dof{0};               // degrees of freedom = M - 3
  bool    fault_detected{false};

  // ── Ring-closure check (always available, no anchor positions needed) ────
  double  ring_sum{0.0};        // Σ tdoa around anchor ring (should be ≈ 0)
  bool    ring_ok{false};       // |ring_sum| < ring_threshold

  // ── Protection levels ────────────────────────────────────────────────────
  double  hpl{std::numeric_limits<double>::quiet_NaN()};   // [m]
  double  vpl{std::numeric_limits<double>::quiet_NaN()};   // [m]

  // ── Fault exclusion (FDE) ────────────────────────────────────────────────
  int     excluded_idx{-1};     // index into meas vector (-1 = none)
  double  chi2_after_fde{std::numeric_limits<double>::quiet_NaN()};

  // ── Per-measurement residuals ─────────────────────────────────────────────
  std::vector<double> residuals;   // raw post-fit residuals r(i)  [m]
  std::vector<double> std_resid;   // normalised: r(i) / sqrt(Σᵢᵢ)

  // ── Availability ─────────────────────────────────────────────────────────
  bool    available{false};     // HPL < HAL && VPL < VAL
};

// =============================================================================
// TdoaIntegrityMonitor  —  RAIM-style integrity monitoring for TDOA WLS
//
// Sign convention matches TdoaWlsSolver:
//   r(i) = (||p - aB|| - ||p - aA||) - tdoa_measured
//
// Noise model: each TDOA pair (A,B) has independent ranging noise σ_meas,
// so Cov(r_i) = 2 * σ_meas².  Cross-pair covariance is zero when pairs do NOT
// share an anchor.  For adjacent-pair ring topologies (A→B→C→…) shared anchors
// would create off-diagonals; this implementation uses the diagonal
// approximation (independent pair noise), which is conservative and appropriate
// for the ring-style TDOA used in the Crazyflie / cf_msgs data stream.
//
// References:
//   Brown (1992) – RAIM baseline; Groves (2013) §13.4
// =============================================================================
class TdoaIntegrityMonitor {
 public:
  // ── Parameters ─────────────────────────────────────────────────────────────
  double tdoa_noise_sigma{0.10};    // [m]  per-pair 1-σ TDOA noise
  double p_fa{1e-3};                // false-alarm probability
  double hal{0.50};                 // horizontal alert limit [m]
  double val{1.00};                 // vertical alert limit   [m]
  double k_hpl{5.33};              // N(0,1) quantile for integrity risk (≈ 1e-7)
  double ring_threshold{0.30};      // |ring_sum| alarm limit [m]
  bool   enable_fde{true};          // run fault-detection-and-exclusion

  // ── Main entry point ───────────────────────────────────────────────────────
  // Call AFTER WLS has converged.  p_wls is the converged position.
  IntegrityResult check(const std::vector<TdoaMeas>& meas,
                        const std::vector<gtsam::Point3>& anchors,
                        const Eigen::Vector3d& p_wls) const {
    IntegrityResult res;
    const int M = static_cast<int>(meas.size());

    // ── Ring-closure (always) ───────────────────────────────────────────────
    res.ring_sum = 0.0;
    for (const auto& m : meas) res.ring_sum += m.tdoa;
    res.ring_ok = (std::fabs(res.ring_sum) < ring_threshold);

    if (M < 4) {
      // Need ≥4 measurements for 3-D position + 1 redundancy
      res.fault_detected = false;
      res.available      = false;
      return res;
    }

    // ── Build H matrix and residual vector ─────────────────────────────────
    Eigen::MatrixXd H(M, 3);
    Eigen::VectorXd r(M);
    buildHAndResiduals(meas, anchors, p_wls, H, r);

    res.residuals.resize(M);
    for (int i = 0; i < M; ++i) res.residuals[i] = r(i);

    // ── Diagonal noise covariance (σ² per measurement) ─────────────────────
    const double sig2 = tdoa_noise_sigma * tdoa_noise_sigma;
    // For adjacent-pair TDOAs each pair has variance 2σ² (two independent ranges)
    const double pair_sig2 = 2.0 * sig2;
    const Eigen::VectorXd diag_inv =
        Eigen::VectorXd::Constant(M, 1.0 / pair_sig2);

    // ── Chi-squared test statistic T = rᵀ Σ⁻¹ r ────────────────────────────
    res.chi2_stat = r.cwiseProduct(diag_inv).dot(r);  // = rᵀ Σ⁻¹ r

    const int dof = M - 3;
    res.dof = dof;
    res.chi2_threshold = chi2Quantile(dof, 1.0 - p_fa);
    res.fault_detected = (res.chi2_stat > res.chi2_threshold);

    // ── Normalised residuals ─────────────────────────────────────────────────
    const double sig_meas = std::sqrt(pair_sig2);
    res.std_resid.resize(M);
    for (int i = 0; i < M; ++i)
      res.std_resid[i] = r(i) / sig_meas;

    // ── Position covariance P = (Hᵀ Σ⁻¹ H)⁻¹ ──────────────────────────────
    const Eigen::Matrix3d HTSH =
        H.transpose() * diag_inv.asDiagonal() * H;
    const Eigen::Matrix3d P_pos = HTSH.inverse();    // 3×3 position covariance

    // ── Protection levels ────────────────────────────────────────────────────
    // σ_H = sqrt(Pxx + Pyy),  σ_V = sqrt(Pzz)
    const double sigma_h = std::sqrt(P_pos(0, 0) + P_pos(1, 1));
    const double sigma_v = std::sqrt(P_pos(2, 2));

    // No-fault HPL/VPL
    const double hpl0 = k_hpl * sigma_h;
    const double vpl0 = k_hpl * sigma_v;

    // Single-fault worst-case slope correction
    // W = (Hᵀ Σ⁻¹ H)⁻¹ Hᵀ Σ⁻¹   (3×M)
    const Eigen::MatrixXd W =
        P_pos * H.transpose() * diag_inv.asDiagonal();

    double hpl_fault = hpl0;
    double vpl_fault = vpl0;
    const double k_md = 4.265;  // ≈ P_MD = 10⁻⁵ normal quantile

    for (int k = 0; k < M; ++k) {
      // Horizontal fault slope: horizontal components of W[:,k]
      const double bias_h =
          std::sqrt(W(0, k) * W(0, k) + W(1, k) * W(1, k));
      const double bias_v = std::fabs(W(2, k));

      // Subset covariance (exclude measurement k) for σ after exclusion
      double sigma_h_k = sigma_h;
      double sigma_v_k = sigma_v;
      if (M > 4) {
        Eigen::MatrixXd H_k(M - 1, 3);
        Eigen::VectorXd d_k(M - 1);
        int row = 0;
        for (int i = 0; i < M; ++i) {
          if (i == k) continue;
          H_k.row(row) = H.row(i);
          d_k(row)     = diag_inv(i);
          ++row;
        }
        const Eigen::Matrix3d HTSH_k =
            H_k.transpose() * d_k.asDiagonal() * H_k;
        const Eigen::Matrix3d P_k = HTSH_k.inverse();
        sigma_h_k = std::sqrt(P_k(0, 0) + P_k(1, 1));
        sigma_v_k = std::sqrt(P_k(2, 2));
      }

      // Worst-case slope (see Brown 1992, eq. 16)
      // Maximum bias that still passes detection test
      const double det_threshold =
          std::sqrt(res.chi2_threshold / diag_inv(k));
      hpl_fault = std::max(hpl_fault,
                           bias_h * det_threshold + k_md * sigma_h_k);
      vpl_fault = std::max(vpl_fault,
                           bias_v * det_threshold + k_md * sigma_v_k);
    }

    res.hpl = std::max(hpl0, hpl_fault);
    res.vpl = std::max(vpl0, vpl_fault);

    // ── FDE: try excluding each measurement ─────────────────────────────────
    if (enable_fde && res.fault_detected && M > 4) {
      double best_chi2 = std::numeric_limits<double>::max();
      int    best_k    = -1;
      for (int k = 0; k < M; ++k) {
        const double t_k = subsetChi2(meas, anchors, p_wls, diag_inv, H, r, k);
        if (t_k < best_chi2) {
          best_chi2 = t_k;
          best_k    = k;
        }
      }
      const double thresh_sub = chi2Quantile(dof - 1, 1.0 - p_fa);
      if (best_chi2 < thresh_sub) {
        res.excluded_idx    = best_k;
        res.chi2_after_fde  = best_chi2;
      }
    }

    // ── Availability ─────────────────────────────────────────────────────────
    res.available = (!res.fault_detected || res.excluded_idx >= 0) &&
                    (res.hpl < hal) && (res.vpl < val);

    return res;
  }

 private:
  // ── Build H and r from current WLS solution ────────────────────────────────
  void buildHAndResiduals(const std::vector<TdoaMeas>& meas,
                          const std::vector<gtsam::Point3>& anchors,
                          const Eigen::Vector3d& p,
                          Eigen::MatrixXd& H,
                          Eigen::VectorXd& r) const {
    const int M = static_cast<int>(meas.size());
    H.resize(M, 3);
    r.resize(M);
    for (int i = 0; i < M; ++i) {
      const auto& m = meas[i];
      const Eigen::Vector3d aA(anchors[m.idA].x(), anchors[m.idA].y(),
                                anchors[m.idA].z());
      const Eigen::Vector3d aB(anchors[m.idB].x(), anchors[m.idB].y(),
                                anchors[m.idB].z());
      const Eigen::Vector3d dA = p - aA;
      const Eigen::Vector3d dB = p - aB;
      const double rA = std::max(dA.norm(), 1e-9);
      const double rB = std::max(dB.norm(), 1e-9);
      // Same sign convention as TdoaWlsSolver
      r(i) = (rB - rA) - m.tdoa;
      H.row(i) = (dB / rB - dA / rA).transpose();
    }
  }

  // ── Chi-squared SSR after excluding measurement k ──────────────────────────
  double subsetChi2(const std::vector<TdoaMeas>& meas,
                    const std::vector<gtsam::Point3>& anchors,
                    const Eigen::Vector3d& p_wls,
                    const Eigen::VectorXd& diag_inv,
                    const Eigen::MatrixXd& H_full,
                    const Eigen::VectorXd& r_full,
                    int excl) const {
    const int M  = static_cast<int>(meas.size());
    const int Ms = M - 1;
    Eigen::MatrixXd H_s(Ms, 3);
    Eigen::VectorXd d_s(Ms), r_s(Ms);
    int row = 0;
    for (int i = 0; i < M; ++i) {
      if (i == excl) continue;
      H_s.row(row) = H_full.row(i);
      d_s(row)     = diag_inv(i);
      r_s(row)     = r_full(i);
      ++row;
    }
    // Re-solve with subset (one GN step around p_wls)
    const Eigen::Matrix3d HTSH_s = H_s.transpose() * d_s.asDiagonal() * H_s;
    const Eigen::Vector3d dp =
        -HTSH_s.ldlt().solve(H_s.transpose() * (d_s.asDiagonal() * r_s));
    const Eigen::VectorXd r_new = r_s + H_s * dp;
    return r_new.cwiseProduct(d_s).dot(r_new);
  }

  // ── Chi-squared upper quantile (inverse CDF) via boost ─────────────────────
  static double chi2Quantile(int dof, double p) {
    if (dof <= 0) return 0.0;
    boost::math::chi_squared dist(static_cast<double>(dof));
    return boost::math::quantile(dist, p);
  }
};

struct FgoState {
  gtsam::Pose3                 pose;
  gtsam::Vector3               vel{gtsam::Vector3::Zero()};
  gtsam::imuBias::ConstantBias bias;
  size_t                       cycle_idx{0};
  bool                         initialized{false};
  double                       last_cycle_t{0.0};
};

class UwbImuLcWlsFusionNode {
 public:
  UwbImuLcWlsFusionNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
      : nh_(nh), pnh_(pnh) {
    loadParams();
    setupGtsam();
    openLog();

    wls_odom_pub_  = nh_.advertise<nav_msgs::Odometry>("uwb_wls", 50);
    lc_odom_pub_   = nh_.advertise<nav_msgs::Odometry>("lc_fusion", 50);
    wls_path_pub_  = nh_.advertise<nav_msgs::Path>("uwb_wls_path", 10);
    lc_path_pub_   = nh_.advertise<nav_msgs::Path>("lc_fusion_path", 10);
    gt_path_pub_   = nh_.advertise<nav_msgs::Path>("uwb_imu_path_gt", 10);
    integ_odom_pub_= nh_.advertise<nav_msgs::Odometry>("uwb_integrity", 50);

    std::string imu_topic, uwb_topic, gt_topic;
    pnh_.param<std::string>("imu_topic", imu_topic, "/imu_data");
    pnh_.param<std::string>("uwb_topic", uwb_topic, "/tdoa_data");
    pnh_.param<std::string>("gt_topic", gt_topic, "/pose_data");
    imu_sub_ = nh_.subscribe(imu_topic, 1000, &UwbImuLcWlsFusionNode::imuCallback, this);
    uwb_sub_ = nh_.subscribe(uwb_topic, 500, &UwbImuLcWlsFusionNode::uwbCallback, this);
    gt_sub_  = nh_.subscribe(gt_topic, 50, &UwbImuLcWlsFusionNode::gtCallback, this);

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
  }

  ~UwbImuLcWlsFusionNode() {
    if (traj_log_.is_open()) traj_log_.close();
  }

 private:
  void loadParams() {
    auto get = [&](const std::string& k, auto& v, auto def) {
      if (!nh_.getParam("/uwb_imu_fusion/" + k, v) && !pnh_.getParam(k, v)) v = def;
    };

    XmlRpc::XmlRpcValue axv;
    const bool ok = nh_.getParam("/uwb_imu_fusion/anchors", axv) ||
                    pnh_.getParam("anchors", axv);
    if (ok && axv.getType() == XmlRpc::XmlRpcValue::TypeArray) {
      for (int i = 0; i < axv.size(); ++i) {
        auto& row = axv[i];
        if (row.getType() == XmlRpc::XmlRpcValue::TypeArray && row.size() == 3) {
          anchors_.emplace_back((double)row[0], (double)row[1], (double)row[2]);
        }
      }
    }
    if (anchors_.empty()) {
      ROS_ERROR("[uwb_imu_fusion] No anchors!");
      ros::shutdown();
      return;
    }

    std::vector<double> ext_t{0, 0, 0}, ext_q{0, 0, 0, 1};
    if (!nh_.getParam("/uwb_imu_fusion/anchor_extrinsic_translation", ext_t)) {
      pnh_.getParam("anchor_extrinsic_translation", ext_t);
    }
    if (!nh_.getParam("/uwb_imu_fusion/anchor_extrinsic_rotation", ext_q)) {
      pnh_.getParam("anchor_extrinsic_rotation", ext_q);
    }
    Eigen::Quaterniond qe(ext_q[3], ext_q[0], ext_q[1], ext_q[2]);
    qe.normalize();
    const Eigen::Vector3d te(ext_t[0], ext_t[1], ext_t[2]);
    for (auto& a : anchors_) {
      const Eigen::Vector3d pw =
          qe.toRotationMatrix() * Eigen::Vector3d(a.x(), a.y(), a.z()) + te;
      a = gtsam::Point3(pw.x(), pw.y(), pw.z());
    }

    get("gyro_is_degrees", gyro_is_degrees_, false);
    get("gravity_magnitude", gravity_mag_, 9.80665);
    get("imu_accel_in_g", imu_accel_in_g_, true);
    const double default_accel_scale = imu_accel_in_g_ ? gravity_mag_ : 1.0;
    get("accel_scale", accel_scale_, default_accel_scale);

    get("accel_noise_sigma", accel_sigma_, 0.003);
    get("gyro_noise_sigma", gyro_sigma_, 0.0003);
    get("accel_bias_rw_sigma", accel_bias_rw_sigma_, 0.0003);
    get("gyro_bias_rw_sigma", gyro_bias_rw_sigma_, 0.00003);
    get("integration_noise_sigma", integration_sigma_, 1e-6);

    get("cycle_timeout", cycle_timeout_, 0.1);
    int mpc = 8, min_imu = 2;
    get("measurements_per_cycle", mpc, 8);
    get("min_imu_per_cycle", min_imu, 2);
    measurements_per_cycle_ = mpc;
    min_imu_per_cycle_ = min_imu;

    get("wls_position_sigma", wls_sigma_, 0.15);

    std::vector<double> pps{0.5, 0.5, 0.5, 0.1, 0.1, 0.1};
    if (!nh_.getParam("/uwb_imu_fusion/prior_pose_sigmas", pps)) {
      pnh_.getParam("prior_pose_sigmas", pps);
    }
    prior_pose_sigmas_ = gtsam::Vector6::Map(pps.data());
    get("prior_vel_sigma", prior_vel_sigma_, 0.1);
    get("prior_bias_sigma", prior_bias_sigma_, 0.01);
    get("lc_update_vel_sigma", lc_update_vel_sigma_, 2.0);
    get("lc_update_bias_sigma", lc_update_bias_sigma_, 0.5);

    std::vector<double> ip{0, 0, 1};
    if (!nh_.getParam("/uwb_imu_fusion/initial_position", ip)) {
      pnh_.getParam("initial_position", ip);
    }
    std::vector<double> iq{0, 0, 0, 1};
    if (!nh_.getParam("/uwb_imu_fusion/initial_orientation", iq)) {
      pnh_.getParam("initial_orientation", iq);
    }
    Eigen::Quaterniond q0(iq[3], iq[0], iq[1], iq[2]);
    q0.normalize();
    initial_pose_ = gtsam::Pose3(gtsam::Rot3(q0.toRotationMatrix()),
                                 gtsam::Point3(ip[0], ip[1], ip[2]));

    get("odom_frame", odom_frame_, std::string("map"));
    get("base_frame", base_frame_, std::string("base_link"));
    get("trajectory_log_path", traj_log_path_,
        std::string("/tmp/uwb_imu_trajectory.csv"));

    // ── Integrity monitoring parameters ──────────────────────────────────────
    double integ_noise_sigma = 0.10;
    double integ_p_fa        = 1e-3;
    double integ_hal         = 0.50;
    double integ_val         = 1.00;
    double integ_ring_thresh = 0.30;
    bool   integ_fde         = true;
    get("integrity_tdoa_noise_sigma", integ_noise_sigma, 0.10);
    get("integrity_p_fa",             integ_p_fa,        1e-3);
    get("integrity_hal",              integ_hal,         0.50);
    get("integrity_val",              integ_val,         1.00);
    get("integrity_ring_threshold",   integ_ring_thresh, 0.30);
    get("integrity_enable_fde",       integ_fde,         true);

    integ_monitor_.tdoa_noise_sigma = integ_noise_sigma;
    integ_monitor_.p_fa             = integ_p_fa;
    integ_monitor_.hal              = integ_hal;
    integ_monitor_.val              = integ_val;
    integ_monitor_.ring_threshold   = integ_ring_thresh;
    integ_monitor_.enable_fde       = integ_fde;
  }

  void setupGtsam() {
    auto p = gtsam::PreintegrationParams::MakeSharedU(gravity_mag_);
    p->setAccelerometerCovariance(
        gtsam::Matrix33::Identity() * std::pow(accel_sigma_, 2));
    p->setGyroscopeCovariance(
        gtsam::Matrix33::Identity() * std::pow(gyro_sigma_, 2));
    p->setIntegrationCovariance(
        gtsam::Matrix33::Identity() * std::pow(integration_sigma_, 2));
    imu_params_ = p;

    lc_.bias = gtsam::imuBias::ConstantBias();
    lc_preint_ = boost::make_shared<gtsam::PreintegratedImuMeasurements>(
        imu_params_, lc_.bias);

    gtsam::ISAM2Params isam_p;
    isam_p.relinearizeThreshold = 0.1;
    isam_p.relinearizeSkip = 1;
    isam_p.factorization = gtsam::ISAM2Params::CHOLESKY;
    lc_isam_ = boost::make_shared<gtsam::ISAM2>(isam_p);
  }

  void openLog() {
    traj_log_.open(traj_log_path_, std::ios::out | std::ios::trunc);
    if (traj_log_.is_open()) {
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
    } else {
      ROS_WARN("[uwb_imu_fusion] Cannot open log: %s", traj_log_path_.c_str());
    }
  }

  void logRow(double t, bool wls_ok, const Eigen::Vector3d& wls_p,
              const IntegrityResult& ir) {
    if (!traj_log_.is_open()) return;
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    const auto v3 = [](const gtsam::Point3& p) {
      return Eigen::Vector3d(p.x(), p.y(), p.z());
    };
    const Eigen::Vector3d lp = lc_.initialized ? v3(lc_.pose.translation())
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
              // ── integrity columns ──────────────────────────────────────────
              << ir.chi2_stat        << ","
              << ir.chi2_threshold   << ","
              << ir.dof              << ","
              << (ir.fault_detected ? 1 : 0) << ","
              << ir.ring_sum         << ","
              << (ir.ring_ok ? 1 : 0) << ","
              << ir.hpl              << ","
              << ir.vpl              << ","
              << (ir.available ? 1 : 0) << ","
              << ir.excluded_idx     << "\n";
    traj_log_.flush();
  }

  void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    const double t = msg->header.stamp.toSec();
    if (last_imu_t_ >= 0.0 && t <= last_imu_t_) {
      ROS_WARN_THROTTLE(1.0, "[uwb_imu_fusion] IMU out-of-order %.6f", t);
      return;
    }

    double gx = msg->angular_velocity.x;
    double gy = msg->angular_velocity.y;
    double gz = msg->angular_velocity.z;
    if (gyro_is_degrees_) {
      const double d2r = M_PI / 180.0;
      gx *= d2r;
      gy *= d2r;
      gz *= d2r;
    }

    ImuMeas s;
    s.t = t;
    s.acc = gtsam::Vector3(msg->linear_acceleration.x * accel_scale_,
                           msg->linear_acceleration.y * accel_scale_,
                           msg->linear_acceleration.z * accel_scale_);
    s.gyr = gtsam::Vector3(gx, gy, gz);
    imu_buf_.push_back(s);
    last_imu_t_ = t;

    const double fence = lc_.initialized ? lc_.last_cycle_t - 0.5 : t - 5.0;
    while (!imu_buf_.empty() && imu_buf_.front().t < fence) {
      imu_buf_.pop_front();
    }
  }

  void uwbCallback(const cf_msgs::Tdoa::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (msg->idA < 0 || msg->idB < 0 ||
        msg->idA >= (int)anchors_.size() ||
        msg->idB >= (int)anchors_.size()) {
      ROS_WARN_THROTTLE(2.0, "[uwb_imu_fusion] Anchor OOB (%d,%d)", msg->idA, msg->idB);
      return;
    }

    TdoaMeas ts{msg->header.stamp.toSec(), msg->idA, msg->idB, msg->data};
    if (!current_cycle_.empty() &&
        (ts.t - current_cycle_.front().t) > cycle_timeout_) {
      current_cycle_.clear();
      cycle_pairs_.clear();
    }

    const auto pr = std::make_pair(std::min(ts.idA, ts.idB), std::max(ts.idA, ts.idB));
    if (cycle_pairs_.count(pr)) {
      if ((int)current_cycle_.size() >= measurements_per_cycle_) {
        processCycle(current_cycle_);
      }
      current_cycle_.clear();
      cycle_pairs_.clear();
    }

    current_cycle_.push_back(ts);
    cycle_pairs_.insert(pr);
    if ((int)current_cycle_.size() >= measurements_per_cycle_) {
      processCycle(current_cycle_);
      current_cycle_.clear();
      cycle_pairs_.clear();
    }
  }

  void gtCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg) {
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

  void processCycle(const std::vector<TdoaMeas>& meas) {
    double t_mid = 0.0;
    for (const auto& m : meas) t_mid += m.t;
    t_mid /= static_cast<double>(meas.size());
    const ros::Time stamp(t_mid);

    Eigen::Vector3d wls_p = wls_guess_;
    const bool wls_ok = wls_solver_.solve(meas, anchors_, wls_p);

    // ── Integrity monitoring ──────────────────────────────────────────────────
    IntegrityResult ir;
    if (wls_ok) {
      ir = integ_monitor_.check(meas, anchors_, wls_p);
      printIntegrity(t_mid, ir, meas);

      // Gate LC update: only feed WLS to FGO when integrity passes
      // (or when FDE successfully excluded one measurement)
      const bool integ_gate = !ir.fault_detected || (ir.excluded_idx >= 0);

      if (integ_gate) {
        wls_guess_ = wls_p;
        if (!lc_.initialized) {
          lcInitGraph(t_mid, wls_p);
        } else {
          lcUpdateGraph(t_mid, wls_p);
        }
      } else {
        ROS_WARN_THROTTLE(1.0,
            "[Integrity] FAULT detected — WLS solution gated out of LC-FGO "
            "(χ²=%.2f > thresh=%.2f, HPL=%.3f m, VPL=%.3f m)",
            ir.chi2_stat, ir.chi2_threshold, ir.hpl, ir.vpl);
      }
    } else {
      ROS_WARN_THROTTLE(2.0, "[LC-FGO] WLS failed — cycle skipped.");
    }

    printCycleInfo(t_mid, wls_ok, wls_p);

    if (wls_ok) {
      publishOdom(wls_odom_pub_, stamp,
                  wls_p.x(), wls_p.y(), wls_p.z(),
                  gtsam::Rot3::identity(), gtsam::Vector3::Zero());
      appendPath(wls_path_, wls_path_pub_, stamp,
                 wls_p.x(), wls_p.y(), wls_p.z(), gtsam::Rot3::identity());

      // Publish integrity odometry: position = WLS, covariance diagonal = HPL²
      publishIntegrityOdom(stamp, wls_p, ir);
    }

    if (lc_.initialized) {
      const auto& lp = lc_.pose.translation();
      publishOdom(lc_odom_pub_, stamp,
                  lp.x(), lp.y(), lp.z(), lc_.pose.rotation(), lc_.vel);
      appendPath(lc_path_, lc_path_pub_, stamp,
                 lp.x(), lp.y(), lp.z(), lc_.pose.rotation());
      broadcastTf(stamp, lp, lc_.pose.rotation(), base_frame_ + "_lc");
    }

    logRow(t_mid, wls_ok, wls_p, ir);
  }

  int replayImu(double from_t, double to_t,
                gtsam::PreintegratedImuMeasurements& preint) {
    int n = 0;
    double prev_t = from_t;
    for (const auto& imu : imu_buf_) {
      if (imu.t <= from_t) continue;
      if (imu.t > to_t) break;
      const double dt = imu.t - prev_t;
      if (dt <= 0.0 || dt > 1.0) {
        prev_t = imu.t;
        continue;
      }
      preint.integrateMeasurement(imu.acc, imu.gyr, dt);
      prev_t = imu.t;
      ++n;
    }
    return n;
  }

  gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>
  biasBetween(size_t k_pre, size_t k, double dt) {
    const double tau = std::max(dt, 1e-3);
    gtsam::Vector6 bs;
    bs << gtsam::Vector3::Constant(accel_bias_rw_sigma_ * std::sqrt(tau)),
          gtsam::Vector3::Constant(gyro_bias_rw_sigma_ * std::sqrt(tau));
    return gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
        B(k_pre), B(k),
        gtsam::imuBias::ConstantBias(),
        gtsam::noiseModel::Diagonal::Sigmas(bs));
  }

  void lcInitGraph(double t_mid, const Eigen::Vector3d& wls_p) {
    lc_.cycle_idx = 0;
    lc_.pose = gtsam::Pose3(initial_pose_.rotation(),
                            gtsam::Point3(wls_p.x(), wls_p.y(), wls_p.z()));
    lc_.vel = gtsam::Vector3::Zero();
    lc_.bias = gtsam::imuBias::ConstantBias();

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;

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
    lc_.vel = est.at<gtsam::Vector3>(V(0));
    lc_.bias = est.at<gtsam::imuBias::ConstantBias>(B(0));

    lc_preint_->resetIntegrationAndSetBias(lc_.bias);
    lc_.initialized = true;
    lc_.last_cycle_t = t_mid;

    ROS_INFO("[LC-FGO] Init  t=%.3f  wls=[%.3f,%.3f,%.3f]  pos=[%.3f,%.3f,%.3f]",
             t_mid, wls_p.x(), wls_p.y(), wls_p.z(),
             lc_.pose.x(), lc_.pose.y(), lc_.pose.z());
  }

  void lcUpdateGraph(double t_mid, const Eigen::Vector3d& wls_p) {
    lc_preint_->resetIntegrationAndSetBias(lc_.bias);
    const int n_imu = replayImu(lc_.last_cycle_t, t_mid, *lc_preint_);

    if (n_imu < min_imu_per_cycle_) {
      ROS_WARN_THROTTLE(1.0, "[LC-FGO] %d IMU (need %d); skip.", n_imu, min_imu_per_cycle_);
      lc_.last_cycle_t = t_mid;
      return;
    }

    ++lc_.cycle_idx;
    const size_t k = lc_.cycle_idx;
    const size_t k_pre = lc_.cycle_idx - 1;

    const gtsam::NavState pred_nav =
        lc_preint_->predict(gtsam::NavState(lc_.pose, lc_.vel), lc_.bias);

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;

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

    try {
      lc_isam_->update(graph, values);
      lc_isam_->update();

      const gtsam::Values est = lc_isam_->calculateEstimate();
      lc_.pose = est.at<gtsam::Pose3>(X(k));
      lc_.vel = est.at<gtsam::Vector3>(V(k));
      lc_.bias = est.at<gtsam::imuBias::ConstantBias>(B(k));
    } catch (const std::exception& e) {
      ROS_ERROR("[LC-FGO] ISAM update failed at k=%zu t=%.3f: %s",
                k, t_mid, e.what());
      lc_.cycle_idx = k_pre;
      lc_.last_cycle_t = t_mid;
      lc_preint_->resetIntegrationAndSetBias(lc_.bias);
      return;
    }

    lc_preint_->resetIntegrationAndSetBias(lc_.bias);
    lc_.last_cycle_t = t_mid;
  }

  // ── Integrity print ─────────────────────────────────────────────────────────
  void printIntegrity(double t_mid, const IntegrityResult& ir,
                      const std::vector<TdoaMeas>& meas) const {
    ROS_INFO("----- [Integrity | t=%.3f] -----", t_mid);
    ROS_INFO("  Ring closure : sum=%.4f m  %s",
             ir.ring_sum, ir.ring_ok ? "OK" : "WARN");
    ROS_INFO("  Chi2 test    : T=%.3f  threshold=%.3f  dof=%d  → %s",
             ir.chi2_stat, ir.chi2_threshold, ir.dof,
             ir.fault_detected ? "FAULT" : "OK");
    if (!std::isnan(ir.hpl))
      ROS_INFO("  HPL=%.4f m  VPL=%.4f m  (HAL=%.2f VAL=%.2f)  avail=%s",
               ir.hpl, ir.vpl,
               integ_monitor_.hal, integ_monitor_.val,
               ir.available ? "YES" : "NO");
    if (ir.excluded_idx >= 0) {
      const auto& mx = meas[ir.excluded_idx];
      ROS_WARN("  FDE excluded meas[%d] (pair %d-%d)  chi2_after=%.3f",
               ir.excluded_idx, mx.idA, mx.idB, ir.chi2_after_fde);
    }
    // Per-measurement normalised residuals
    for (int i = 0; i < (int)ir.std_resid.size(); ++i) {
      ROS_INFO("    meas[%d] (%d-%d): r=%.4f m  r/σ=%.2f%s",
               i, meas[i].idA, meas[i].idB,
               ir.residuals[i], ir.std_resid[i],
               (i == ir.excluded_idx) ? "  ← excluded" : "");
    }
  }

  // ── Publish integrity odometry (HPL encoded as covariance diagonal) ─────────
  void publishIntegrityOdom(const ros::Time& stamp,
                            const Eigen::Vector3d& wls_p,
                            const IntegrityResult& ir) {
    nav_msgs::Odometry msg;
    msg.header.stamp    = stamp;
    msg.header.frame_id = odom_frame_;
    msg.child_frame_id  = base_frame_ + "_integ";
    msg.pose.pose.position.x = wls_p.x();
    msg.pose.pose.position.y = wls_p.y();
    msg.pose.pose.position.z = wls_p.z();
    msg.pose.pose.orientation.w = 1.0;

    // Encode HPL/VPL into covariance diagonal so RViz / tools can visualise
    const double hpl2 = std::isnan(ir.hpl) ? 9999.0 : ir.hpl * ir.hpl;
    const double vpl2 = std::isnan(ir.vpl) ? 9999.0 : ir.vpl * ir.vpl;
    msg.pose.covariance[0]  = hpl2;   // xx
    msg.pose.covariance[7]  = hpl2;   // yy
    msg.pose.covariance[14] = vpl2;   // zz

    // Encode fault flag into twist.linear.x (1=fault, 0=ok) for easy monitoring
    msg.twist.twist.linear.x = ir.fault_detected ? 1.0 : 0.0;
    msg.twist.twist.linear.y = ir.chi2_stat;
    msg.twist.twist.linear.z = ir.ring_sum;

    integ_odom_pub_.publish(msg);
  }

  void printCycleInfo(double t_mid, bool wls_ok, const Eigen::Vector3d& wls_p) const {
    const auto err3 = [&](const gtsam::Point3& p) -> double {
      return gt_have_ ? std::sqrt(std::pow(p.x() - gt_pos_.x(), 2) +
                                  std::pow(p.y() - gt_pos_.y(), 2) +
                                  std::pow(p.z() - gt_pos_.z(), 2))
                      : -1.0;
    };
    ROS_INFO("=================================================");
    ROS_INFO("[LC=%4zu | t=%.3f s]", lc_.cycle_idx, t_mid);
    if (wls_ok) {
      ROS_INFO("  WLS    : [%7.3f,%7.3f,%7.3f]", wls_p.x(), wls_p.y(), wls_p.z());
    } else {
      ROS_INFO("  WLS    : DID NOT CONVERGE");
    }
    if (lc_.initialized) {
      const auto& lp = lc_.pose.translation();
      const auto rpy = lc_.pose.rotation().rpy();
      ROS_INFO("  LC-FGO : [%7.3f,%7.3f,%7.3f]  rpy=[%.1f,%.1f,%.1f]deg",
               lp.x(), lp.y(), lp.z(),
               rpy.x() * 180.0 / M_PI, rpy.y() * 180.0 / M_PI, rpy.z() * 180.0 / M_PI);
      ROS_INFO("           vel=[%.3f,%.3f,%.3f]  ba=[%.4f,%.4f,%.4f]",
               lc_.vel.x(), lc_.vel.y(), lc_.vel.z(),
               lc_.bias.accelerometer().x(),
               lc_.bias.accelerometer().y(),
               lc_.bias.accelerometer().z());
    }
    if (gt_have_) {
      ROS_INFO("  GT     : [%7.3f,%7.3f,%7.3f]", gt_pos_.x(), gt_pos_.y(), gt_pos_.z());
      if (wls_ok) {
        ROS_INFO("  |WLS-GT| = %.4f m",
                 (wls_p - Eigen::Vector3d(gt_pos_.x(), gt_pos_.y(), gt_pos_.z())).norm());
      }
      if (lc_.initialized) {
        ROS_INFO("  |LC -GT| = %.4f m", err3(lc_.pose.translation()));
      }
    }
    ROS_INFO("=================================================");
  }

  void publishOdom(ros::Publisher& pub, const ros::Time& stamp,
                   double x, double y, double z,
                   const gtsam::Rot3& rot, const gtsam::Vector3& vel) {
    const Eigen::Quaterniond q(rot.matrix());
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

  void appendPath(nav_msgs::Path& path, ros::Publisher& pub,
                  const ros::Time& stamp,
                  double x, double y, double z, const gtsam::Rot3& rot) {
    const Eigen::Quaterniond q(rot.matrix());
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

  void broadcastTf(const ros::Time& stamp, const gtsam::Point3& pos,
                   const gtsam::Rot3& rot, const std::string& child) {
    const Eigen::Quaterniond q(rot.matrix());
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

  ros::NodeHandle nh_, pnh_;
  ros::Subscriber imu_sub_, uwb_sub_, gt_sub_;

  ros::Publisher wls_odom_pub_, lc_odom_pub_;
  ros::Publisher wls_path_pub_, lc_path_pub_, gt_path_pub_;
  ros::Publisher integ_odom_pub_;

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

  double accel_scale_{1.0};
  bool imu_accel_in_g_{true};
  bool gyro_is_degrees_{false};
  double accel_sigma_{0.003}, gyro_sigma_{0.0003};
  double accel_bias_rw_sigma_{0.0003}, gyro_bias_rw_sigma_{0.00003};
  double integration_sigma_{1e-6}, gravity_mag_{9.80665};

  double wls_sigma_{0.15};
  gtsam::Vector6 prior_pose_sigmas_;
  double prior_vel_sigma_{0.1}, prior_bias_sigma_{0.01};
  double lc_update_vel_sigma_{2.0}, lc_update_bias_sigma_{0.5};

  gtsam::Pose3 initial_pose_;
  boost::shared_ptr<gtsam::PreintegrationParams> imu_params_;

  FgoState lc_;
  boost::shared_ptr<gtsam::PreintegratedImuMeasurements> lc_preint_;
  boost::shared_ptr<gtsam::ISAM2> lc_isam_;

  bool gt_have_{false};
  gtsam::Point3 gt_pos_{gtsam::Point3::Zero()};

  nav_msgs::Path wls_path_, lc_path_, gt_path_;

  std::string traj_log_path_{"/tmp/uwb_imu_trajectory.csv"};
  std::ofstream traj_log_;

  TdoaIntegrityMonitor integ_monitor_;  // integrity monitor instance
};

}  // namespace uwb_imu_fusion

int main(int argc, char** argv) {
  ros::init(argc, argv, "integrity_uwb_node");
  ros::NodeHandle nh, pnh("~");
  uwb_imu_fusion::UwbImuLcWlsFusionNode node(nh, pnh);
  ros::AsyncSpinner spinner(2);
  spinner.start();
  ros::waitForShutdown();
  return 0;
}