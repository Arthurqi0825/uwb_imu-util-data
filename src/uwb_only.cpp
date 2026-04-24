// uwb_imu_fusion_node.cpp  —  UWB diagnostic / examination build
//
// Two parallel pure-UWB estimators (no IMU) to examine raw UWB quality.
//
//  1. Pure-UWB WLS  →  "uwb_wls"
//     Gauss-Newton TDOA solver.  Sign convention matches cf_msgs/Tdoa:
//       sensor measurement  z  =  d(tag, anchorB) - d(tag, anchorA)
//       residual            r  =  (rB - rA) - z
//     This is the ORIGINAL working version — do NOT change the sign.
//
//  2. Pure-UWB GTSAM  →  "uwb_gtsam"
//     Fresh LevenbergMarquardt solve every cycle.
//     Uses uwb_imu_fusion::TdoaFactor from tdoa_factor.h which computes:
//       predicted  =  distA - distB  =  rA - rB
//       residual   =  predicted - measured_tdoa
//     So to feed the sensor value z (= rB - rA) we pass  measured = -z
//     i.e.  -m.tdoa  to TdoaFactor.
//
//     GTSAM crash fix: TDOA has zero rotation information → 3 rotation
//     DOFs are unobservable → Hessian is singular.
//     Fix: PriorFactor on Pose3 with tight rotation σ=1e-4 rad (nearly
//     fixed) and loose translation σ=10 m (let TDOA decide).
//     Fresh graph + LM per cycle avoids iSAM2 growing-keyframe singularity.
//
//  Terminal every cycle: WLS | GTSAM | GT positions + errors.

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <cf_msgs/Tdoa.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/slam/PriorFactor.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <mutex>
#include <set>
#include <utility>
#include <vector>
#include <cmath>
#include <string>

#include "tdoa_factor.h"   // uwb_imu_fusion::TdoaFactor

using gtsam::symbol_shorthand::X;

namespace uwb_diag {

// ============================================================================
struct TdoaMeas {
  double t;
  int    idA, idB;
  double tdoa;   // = d(tag,B) - d(tag,A)   [cf_msgs/Tdoa::data convention]
};

// ============================================================================
// Gauss-Newton WLS  —  ORIGINAL sign, verified working
//   residual:  r_i = (rB - rA) - tdoa_i
//   Jacobian:  dr/dp = dB/rB - dA/rA
// ============================================================================
class TdoaWlsSolver {
 public:
  explicit TdoaWlsSolver(int max_iter = 50, double tol = 1e-6)
      : max_iter_(max_iter), tol_(tol) {}

  bool solve(const std::vector<TdoaMeas>& meas,
             const std::vector<gtsam::Point3>& anchors,
             Eigen::Vector3d& p) const {
    if ((int)meas.size() < 3) return false;

    for (int iter = 0; iter < max_iter_; ++iter) {
      const int n = (int)meas.size();
      Eigen::MatrixXd J(n, 3);
      Eigen::VectorXd r(n);

      for (int i = 0; i < n; ++i) {
        const auto& m = meas[i];
        Eigen::Vector3d aA(anchors[m.idA].x(), anchors[m.idA].y(), anchors[m.idA].z());
        Eigen::Vector3d aB(anchors[m.idB].x(), anchors[m.idB].y(), anchors[m.idB].z());
        Eigen::Vector3d dA = p - aA;
        Eigen::Vector3d dB = p - aB;
        double rA = std::max(dA.norm(), 1e-9);
        double rB = std::max(dB.norm(), 1e-9);

        r(i)     = (rB - rA) - m.tdoa;          // data = rB - rA
        J.row(i) = (dB / rB - dA / rA).transpose();
      }

      Eigen::Vector3d dp = -(J.transpose() * J).ldlt().solve(J.transpose() * r);
      p += dp;
      if (dp.norm() < tol_) return true;
    }
    return false;
  }

 private:
  int    max_iter_;
  double tol_;
};

// ============================================================================
class UwbDiagNode {
 public:
  UwbDiagNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
      : nh_(nh), pnh_(pnh) {
    loadParams();

    wls_pub_   = nh_.advertise<nav_msgs::Odometry>("uwb_wls",   50);
    gtsam_pub_ = nh_.advertise<nav_msgs::Odometry>("uwb_gtsam", 50);

    wls_path_pub_   = nh_.advertise<nav_msgs::Path>("uwb_wls_path",    10);
    gtsam_path_pub_ = nh_.advertise<nav_msgs::Path>("uwb_gtsam_path",  10);
    gt_path_pub_    = nh_.advertise<nav_msgs::Path>("uwb_imu_path_gt", 10);

    uwb_sub_ = nh_.subscribe("uwb_tdoa",  500, &UwbDiagNode::uwbCallback, this);
    gt_sub_  = nh_.subscribe("/pose_data", 50, &UwbDiagNode::gtCallback,  this);

    // Anchor centroid — fallback initial guess
    gtsam::Point3 cen(0, 0, 0);
    for (const auto& a : anchors_) cen = cen + a;
    if (!anchors_.empty())
      anchor_centroid_ = gtsam::Point3(cen.x() / anchors_.size(),
                                       cen.y() / anchors_.size(),
                                       cen.z() / anchors_.size());

    current_guess_ = Eigen::Vector3d(anchor_centroid_.x(),
                                     anchor_centroid_.y(),
                                     anchor_centroid_.z());
    std::vector<double> ip{0, 0, 1};
    if (nh_.getParam("/uwb_imu_fusion/initial_position", ip) ||
        pnh_.getParam("initial_position", ip))
      current_guess_ = Eigen::Vector3d(ip[0], ip[1], ip[2]);

    ROS_INFO("[uwb_diag] Ready.  %zu anchors.  meas/cycle=%d.",
             anchors_.size(), measurements_per_cycle_);
    ROS_INFO("[uwb_diag] tdoa_sigma=%.3f  robust=%s  huber_k=%.2f",
             tdoa_sigma_, use_robust_noise_ ? "true" : "false", huber_k_);
    ROS_INFO("[uwb_diag] Initial guess: [%.3f, %.3f, %.3f]",
             current_guess_.x(), current_guess_.y(), current_guess_.z());
    ROS_INFO("[uwb_diag] Publishing: uwb_wls | uwb_gtsam");
  }

 private:
  // --------------------------------------------------------------------------
  void loadParams() {
    auto get = [&](const std::string& k, auto& v, auto def) {
      if (!nh_.getParam("/uwb_imu_fusion/" + k, v) && !pnh_.getParam(k, v))
        v = def;
    };

    XmlRpc::XmlRpcValue axv;
    bool ok = nh_.getParam("/uwb_imu_fusion/anchors", axv) ||
              pnh_.getParam("anchors", axv);
    if (ok && axv.getType() == XmlRpc::XmlRpcValue::TypeArray) {
      for (int i = 0; i < axv.size(); ++i) {
        auto& row = axv[i];
        if (row.getType() == XmlRpc::XmlRpcValue::TypeArray && row.size() == 3)
          anchors_.emplace_back((double)row[0], (double)row[1], (double)row[2]);
      }
    }
    if (anchors_.empty())
      ROS_ERROR("[uwb_diag] No anchors loaded!");

    std::vector<double> ext_t{0,0,-0.05}, ext_q{0,0,0,1};
    if (!nh_.getParam("/uwb_imu_fusion/anchor_extrinsic_translation", ext_t))
      pnh_.getParam("anchor_extrinsic_translation", ext_t);
    if (!nh_.getParam("/uwb_imu_fusion/anchor_extrinsic_rotation", ext_q))
      pnh_.getParam("anchor_extrinsic_rotation", ext_q);
    Eigen::Quaterniond qe(ext_q[3], ext_q[0], ext_q[1], ext_q[2]);
    qe.normalize();
    Eigen::Matrix3d Re = qe.toRotationMatrix();
    Eigen::Vector3d te(ext_t[0], ext_t[1], ext_t[2]);
    for (auto& a : anchors_) {
      Eigen::Vector3d pw = Re * Eigen::Vector3d(a.x(), a.y(), a.z()) + te;
      a = gtsam::Point3(pw.x(), pw.y(), pw.z());
    }

    get("tdoa_noise_sigma",       tdoa_sigma_,           0.15);
    get("use_robust_noise",       use_robust_noise_,     true);
    get("huber_k",                huber_k_,              1.0);
    get("cycle_timeout",          cycle_timeout_,        0.1);
    int na = 8, mpc = 8;
    get("num_anchors",            na,  8);
    get("measurements_per_cycle", mpc, 8);
    num_anchors_            = na;
    measurements_per_cycle_ = mpc;
    get("odom_frame", odom_frame_, std::string("map"));
  }

  // --------------------------------------------------------------------------
  void gtCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    gt_pos_  = gtsam::Point3(msg->pose.pose.position.x,
                             msg->pose.pose.position.y,
                             msg->pose.pose.position.z);
    gt_have_ = true;

    // Append to GT path and publish
    geometry_msgs::PoseStamped ps;
    ps.header = msg->header;
    ps.header.frame_id = odom_frame_;
    ps.pose.position    = msg->pose.pose.position;
    ps.pose.orientation = msg->pose.pose.orientation;
    gt_path_.header.stamp    = msg->header.stamp;
    gt_path_.header.frame_id = odom_frame_;
    gt_path_.poses.push_back(ps);
    gt_path_pub_.publish(gt_path_);
  }

  // --------------------------------------------------------------------------
  void uwbCallback(const cf_msgs::Tdoa::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(mutex_);

    if (msg->idA < 0 || msg->idB < 0 ||
        msg->idA >= (int)anchors_.size() ||
        msg->idB >= (int)anchors_.size()) {
      ROS_WARN_THROTTLE(2.0, "[uwb_diag] Anchor id out of range (%d,%d).",
                        msg->idA, msg->idB);
      return;
    }

    TdoaMeas ts{msg->header.stamp.toSec(), msg->idA, msg->idB, msg->data};

    if (!current_cycle_.empty() &&
        (ts.t - current_cycle_.front().t) > cycle_timeout_) {
      current_cycle_.clear();
      cycle_pairs_.clear();
    }

    auto pr = std::make_pair(std::min(ts.idA, ts.idB), std::max(ts.idA, ts.idB));
    if (cycle_pairs_.count(pr)) {
      if ((int)current_cycle_.size() >= measurements_per_cycle_)
        processCycle(current_cycle_);
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

  // --------------------------------------------------------------------------
  void processCycle(const std::vector<TdoaMeas>& meas) {
    double t_mid = 0.0;
    for (const auto& m : meas) t_mid += m.t;
    t_mid /= (double)meas.size();
    ros::Time stamp(t_mid);
    ++cycle_index_;

    // ── 1. WLS ─────────────────────────────────────────────────────────────
    // Original sign: (rB - rA) - tdoa.  Do NOT change.
    Eigen::Vector3d wls_p = current_guess_;
    bool wls_ok = wls_solver_.solve(meas, anchors_, wls_p);

    // ── 2. GTSAM ───────────────────────────────────────────────────────────
    //
    // TdoaFactor::evaluateError (tdoa_factor.h):
    //   predicted  = distA - distB  = rA - rB
    //   residual   = predicted - measured_tdoa
    //
    // Sensor (cf_msgs/Tdoa::data):
    //   z = rB - rA
    //
    // Therefore:  measured_tdoa to pass = -z = -m.tdoa
    // so that residual = (rA - rB) - (-m.tdoa) = (rA - rB) + m.tdoa
    //                  = -[(rB - rA) - m.tdoa]    ← same zero as WLS ✓
    //
    // Rotation fix: TDOA gives zero rotation info.  Add a Pose3 prior with
    //   σ_rot = 1e-4 rad  (effectively fixed, ~0.006°)
    //   σ_trans = 10 m    (very loose, let TDOA solve translation)
    // This makes all 6 DOFs of the Hessian full-rank with negligible bias.
    //
    // Warm-start: use WLS result so linearisation starts at the solution.
    // ──────────────────────────────────────────────────────────────────────
    Eigen::Vector3d init_t = wls_ok ? wls_p : current_guess_;
    gtsam::Pose3 init_pose(gtsam::Rot3::identity(),
                           gtsam::Point3(init_t.x(), init_t.y(), init_t.z()));

    auto base_noise = gtsam::noiseModel::Isotropic::Sigma(1, tdoa_sigma_);
    gtsam::SharedNoiseModel tdoa_noise;
    if (use_robust_noise_)
      tdoa_noise = gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Huber::Create(huber_k_), base_noise);
    else
      tdoa_noise = base_noise;

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values               values;
    values.insert(X(0), init_pose);

    // Pose3 tangent: [rot(3), trans(3)]
    gtsam::Vector6 prior_sigmas;
    prior_sigmas << 1e-4, 1e-4, 1e-4,   // rotation  — pinned
                    10.0, 10.0, 10.0;    // translation — free
    graph.addPrior(X(0), init_pose,
                   gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas));

    for (const auto& m : meas) {
      // Pass -m.tdoa to match TdoaFactor's (distA - distB) sign convention
      graph.add(uwb_imu_fusion::TdoaFactor(
          X(0), anchors_[m.idA], anchors_[m.idB],
          -m.tdoa,
          tdoa_noise));
    }

    gtsam::LevenbergMarquardtParams lm_params;
    lm_params.setMaxIterations(100);
    lm_params.setRelativeErrorTol(1e-7);
    lm_params.setAbsoluteErrorTol(1e-7);
    lm_params.setVerbosity("SILENT");

    gtsam::Point3 gtsam_pos;
    bool gtsam_ok = false;
    try {
      gtsam::LevenbergMarquardtOptimizer lm(graph, values, lm_params);
      gtsam_pos = lm.optimize().at<gtsam::Pose3>(X(0)).translation();
      gtsam_ok  = true;
    } catch (const std::exception& e) {
      ROS_WARN("[uwb_diag] GTSAM LM failed (cycle %zu): %s",
               cycle_index_, e.what());
    }

    // Update warm-start
    if (wls_ok)
      current_guess_ = wls_p;
    else if (gtsam_ok)
      current_guess_ = Eigen::Vector3d(gtsam_pos.x(), gtsam_pos.y(), gtsam_pos.z());

    // ── 3. Terminal ─────────────────────────────────────────────────────────
    ROS_INFO("=================================================");
    ROS_INFO("[Cycle %4zu | t=%.3f s]", cycle_index_, t_mid);

    if (wls_ok)
      ROS_INFO("  WLS   : [%7.3f, %7.3f, %7.3f]",
               wls_p.x(), wls_p.y(), wls_p.z());
    else
      ROS_INFO("  WLS   : DID NOT CONVERGE");

    if (gtsam_ok)
      ROS_INFO("  GTSAM : [%7.3f, %7.3f, %7.3f]",
               gtsam_pos.x(), gtsam_pos.y(), gtsam_pos.z());
    else
      ROS_INFO("  GTSAM : FAILED");

    if (gt_have_) {
      ROS_INFO("  GT    : [%7.3f, %7.3f, %7.3f]",
               gt_pos_.x(), gt_pos_.y(), gt_pos_.z());
      if (wls_ok) {
        double ew = std::sqrt(
            std::pow(wls_p.x() - gt_pos_.x(), 2) +
            std::pow(wls_p.y() - gt_pos_.y(), 2) +
            std::pow(wls_p.z() - gt_pos_.z(), 2));
        ROS_INFO("  |WLS   - GT| = %.4f m", ew);
      }
      if (gtsam_ok) {
        double eg = std::sqrt(
            std::pow(gtsam_pos.x() - gt_pos_.x(), 2) +
            std::pow(gtsam_pos.y() - gt_pos_.y(), 2) +
            std::pow(gtsam_pos.z() - gt_pos_.z(), 2));
        ROS_INFO("  |GTSAM - GT| = %.4f m", eg);
      }
    } else {
      ROS_INFO("  GT    : [waiting for /pose_data]");
    }
    ROS_INFO("=================================================");

    // ── 4. Publish ──────────────────────────────────────────────────────────
    if (wls_ok) {
      publishOdom(wls_pub_, stamp, wls_p.x(), wls_p.y(), wls_p.z());
      appendPath(wls_path_, wls_path_pub_, stamp, wls_p.x(), wls_p.y(), wls_p.z());
    }
    if (gtsam_ok) {
      publishOdom(gtsam_pub_, stamp, gtsam_pos.x(), gtsam_pos.y(), gtsam_pos.z());
      appendPath(gtsam_path_, gtsam_path_pub_, stamp,
                 gtsam_pos.x(), gtsam_pos.y(), gtsam_pos.z());
    }
  }

  // --------------------------------------------------------------------------
  void publishOdom(ros::Publisher& pub, const ros::Time& stamp,
                   double x, double y, double z) {
    nav_msgs::Odometry odom;
    odom.header.stamp    = stamp;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id  = "base_link";
    odom.pose.pose.position.x    = x;
    odom.pose.pose.position.y    = y;
    odom.pose.pose.position.z    = z;
    odom.pose.pose.orientation.w = 1.0;
    pub.publish(odom);
  }

  void appendPath(nav_msgs::Path& path, ros::Publisher& pub,
                  const ros::Time& stamp, double x, double y, double z) {
    geometry_msgs::PoseStamped ps;
    ps.header.stamp    = stamp;
    ps.header.frame_id = odom_frame_;
    ps.pose.position.x    = x;
    ps.pose.position.y    = y;
    ps.pose.position.z    = z;
    ps.pose.orientation.w = 1.0;
    path.header.stamp    = stamp;
    path.header.frame_id = odom_frame_;
    path.poses.push_back(ps);
    pub.publish(path);
  }

  // --------------------------------------------------------------------------
  ros::NodeHandle nh_, pnh_;
  ros::Subscriber uwb_sub_, gt_sub_;
  ros::Publisher  wls_pub_, gtsam_pub_;
  ros::Publisher  wls_path_pub_, gtsam_path_pub_, gt_path_pub_;
  std::mutex      mutex_;

  std::vector<gtsam::Point3>   anchors_;
  gtsam::Point3                anchor_centroid_{gtsam::Point3::Zero()};
  std::string                  odom_frame_{"map"};

  std::vector<TdoaMeas>        current_cycle_;
  std::set<std::pair<int,int>> cycle_pairs_;
  int    num_anchors_{8};
  int    measurements_per_cycle_{8};
  double cycle_timeout_{0.1};
  size_t cycle_index_{0};

  TdoaWlsSolver   wls_solver_;
  Eigen::Vector3d current_guess_{0, 0, 1};

  double tdoa_sigma_{0.15};
  bool   use_robust_noise_{true};
  double huber_k_{1.0};

  bool          gt_have_{false};
  gtsam::Point3 gt_pos_{gtsam::Point3::Zero()};

  // Accumulated paths
  nav_msgs::Path wls_path_;
  nav_msgs::Path gtsam_path_;
  nav_msgs::Path gt_path_;
};

}  // namespace uwb_diag

// ============================================================================
int main(int argc, char** argv) {
  ros::init(argc, argv, "uwb_imu_fusion_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  uwb_diag::UwbDiagNode node(nh, pnh);
  ros::AsyncSpinner spinner(2);
  spinner.start();
  ros::waitForShutdown();
  return 0;
}