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

#include "tdoa_factor.h" // uwb_imu_fusion::TdoaFactor

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

namespace uwb_imu_fusion
{

  // ============================================================================
  // ZeroVelocityFactor — soft 3-D prior pulling V(k) toward zero
  // sigma = zero_vel_sigma (tune to max expected drone speed)
  // Key role: prevents uncorrected IMU bias from freely integrating into velocity
  // ============================================================================
  class ZeroVelocityFactor : public gtsam::NoiseModelFactor1<gtsam::Vector3>
  {
  public:
    ZeroVelocityFactor(gtsam::Key key, const gtsam::SharedNoiseModel &model)
        : gtsam::NoiseModelFactor1<gtsam::Vector3>(model, key) {}

    gtsam::Vector evaluateError(
        const gtsam::Vector3 &v,
        boost::optional<gtsam::Matrix &> H = boost::none) const override
    {
      if (H)
        *H = gtsam::Matrix33::Identity();
      return v;
    }
    gtsam::NonlinearFactor::shared_ptr clone() const override
    {
      return boost::make_shared<ZeroVelocityFactor>(*this);
    }
  };

  // ============================================================================
  // AltitudeFactor — pulls translation-Z of Pose3 toward z_ref
  // Compensates for weak vertical TDOA geometry (anchors span only ~2.5 m Z)
  // ============================================================================
  class AltitudeFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3>
  {
    double z_ref_;

  public:
    AltitudeFactor(gtsam::Key key, double z_ref,
                   const gtsam::SharedNoiseModel &model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, key), z_ref_(z_ref) {}

    gtsam::Vector evaluateError(
        const gtsam::Pose3 &p,
        boost::optional<gtsam::Matrix &> H = boost::none) const override
    {
      gtsam::Matrix36 Ht;
      gtsam::Point3 t = p.translation(H ? &Ht : nullptr);
      if (H)
        *H = Ht.row(2);
      return (gtsam::Vector(1) << t.z() - z_ref_).finished();
    }
    gtsam::NonlinearFactor::shared_ptr clone() const override
    {
      return boost::make_shared<AltitudeFactor>(*this);
    }
  };

  // ============================================================================
  struct ImuMeas
  {
    double t;
    gtsam::Vector3 acc; // in whatever unit sensor reports
    gtsam::Vector3 gyr; // rad/s (convert if needed)
  };

  struct TdoaMeas
  {
    double t;
    int idA, idB;
    double tdoa; // cf_msgs/Tdoa::data = d(tag,B) − d(tag,A) [m]
  };

  // ============================================================================
  // Gauss-Newton WLS — sign verified, do not change
  // ============================================================================
  class TdoaWlsSolver
  {
  public:
    explicit TdoaWlsSolver(int max_iter = 100, double tol = 1e-6)
        : max_iter_(max_iter), tol_(tol) {}

    bool solve(const std::vector<TdoaMeas> &meas,
               const std::vector<gtsam::Point3> &anchors,
               Eigen::Vector3d &p) const
    {
      if ((int)meas.size() < 3)
        return false;
      for (int iter = 0; iter < max_iter_; ++iter)
      {
        const int n = (int)meas.size();
        Eigen::MatrixXd J(n, 3);
        Eigen::VectorXd r(n);
        for (int i = 0; i < n; ++i)
        {
          const auto &m = meas[i];
          Eigen::Vector3d anchar_A(anchors[m.idA].x(), anchors[m.idA].y(), anchors[m.idA].z());
          Eigen::Vector3d anchar_B(anchors[m.idB].x(), anchors[m.idB].y(), anchors[m.idB].z());
          Eigen::Vector3d dist_A = p - anchar_A, dist_B = p - anchar_B;
          double rA = std::max(dist_A.norm(), 1e-9), rB = std::max(dist_B.norm(), 1e-9);
          r(i) = (rB - rA) - m.tdoa;
          J.row(i) = (dist_B / rB - dist_A / rA).transpose();
        }
        Eigen::Vector3d dp = -(J.transpose() * J).ldlt().solve(J.transpose() * r);
        p += dp;
        if (dp.norm() < tol_)
          return true;
      }
      return false;
    }

  private:
    int max_iter_;
    double tol_;
  };

  // ============================================================================
  class UwbImuFusionNode
  {
  private:
    ros::NodeHandle nh_;
    ros::Subscriber imu_sub_, uwb_sub_, gt_sub_;
    ros::Publisher wls_odom_pub_, fusion_odom_pub_;
    ros::Publisher wls_path_pub_, fusion_path_pub_, gt_path_pub_;

    std::string imu_topic_, uwb_topic_, gt_topic_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    std::mutex mutex_;

    ros::Timer optimization_timer_;

    std::vector<gtsam::Point3> anchors_;
    std::string odom_frame_{"map"}, base_frame_{"base_link"};

    std::vector<TdoaMeas> current_cycle_;
    std::set<std::pair<int, int>> cycle_pairs_;
    int num_anchors_{8}, measurements_per_cycle_{8}, min_imu_per_cycle_{2};
    double cycle_timeout_{0.1};
    size_t cycle_index_{0};

    TdoaWlsSolver wls_solver_;
    Eigen::Vector3d wls_guess_{0, 0, 1};

    double last_cycle_t_{0.0};

    struct State
    {
      double timestamp;
      gtsam::Pose3 pose;
      gtsam::Vector3 vel;
      gtsam::imuBias::ConstantBias bias;
    };

    State current_state_;
    State propagated_state_;
    std::deque<State> state_history_;

    std::deque<ImuMeas> imu_buf_;
    std::deque<TdoaMeas> uwb_buf_;

    std::unique_ptr<gtsam::ISAM2> isam_;
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values initial_values_;
    gtsam::PreintegratedImuMeasurements *imu_preintegrated_;
    gtsam::PreintegratedImuMeasurements *imu_propagation_;

    gtsam::SharedNoiseModel uwb_position_noise_;   // UWB position measurement noise
    gtsam::SharedNoiseModel pose_prior_noise_;     // Prior pose noise
    gtsam::SharedNoiseModel bias_prior_noise_;     // Prior bias noise
    gtsam::SharedNoiseModel velocity_prior_noise_; // Prior velocity noise

    // State tracking variables
    bool initialized_ = false;      // System initialization flag
    size_t key_count_ = 0;          // Number of keyframes added
    double last_imu_time_ = -1;     // Timestamp of last processed IMU message
    double last_prop_time_ = -1;    // New: Timestamp of last propagation
    double last_gnss_time_ = -1;    // Timestamp of last processed GNSS message
    double last_imu_ros_time_ = -1; // ROS time of last IMU message

    nav_msgs::Path wls_path_, fusion_path_, gt_path_;
    std::string traj_log_path_{"/tmp/uwb_imu_residuals.csv"};
    std::ofstream traj_log_;

    /// Ai part

    // IMU noise

    double gravity_mag_{0.0};

    // UWB noise
    double tdoa_sigma_{0.15};
    bool use_robust_noise_{true};
    double huber_k_{1.345};
    gtsam::SharedNoiseModel tdoa_noise_;

    // Velocity damping
    double zero_vel_sigma_{0.05}; // FIXED from 0.3
    gtsam::SharedNoiseModel zero_vel_noise_;

    // Altitude constraint (now wired in)
    bool use_altitude_constraint_{false};
    double altitude_z_ref_{1.0};
    double altitude_sigma_{0.5};
    gtsam::SharedNoiseModel alt_noise_;

    // gtsam component:

    // Priors
    gtsam::Vector6 prior_pose_sigmas_;

    gtsam::Pose3 initial_pose_;

    boost::shared_ptr<gtsam::PreintegrationParams> imu_params_;
    boost::shared_ptr<gtsam::PreintegratedImuMeasurements> preint_;

    gtsam::Pose3 current_pose_;
    gtsam::Vector3 current_vel_{gtsam::Vector3::Zero()};
    gtsam::imuBias::ConstantBias current_bias_;
    bool initialized_{false};

    bool gt_have_{false};
    gtsam::Point3 gt_pos_{gtsam::Point3::Zero()};

  public:
    UwbImuFusionNode(ros::NodeHandle &nh, ros::NodeHandle &pnh)
        : nh_(nh), pnh_(pnh)
    {
      loadParams();
      loadingAnchors();
      initializeGTSAM();
      openLog();

      imu_sub_ = nh_.subscribe("imu", 1000, &UwbImuFusionNode::imuCallback, this);
      uwb_sub_ = nh_.subscribe("uwb_tdoa", 500, &UwbImuFusionNode::uwbCallback, this);
      gt_sub_ = nh_.subscribe("/pose_data", 50, &UwbImuFusionNode::gtCallback, this);

      wls_guess_ = Eigen::Vector3d(initial_pose_.translation().x(),
                                   initial_pose_.translation().y(),
                                   initial_pose_.translation().z());

      wls_odom_pub_ = nh_.advertise<nav_msgs::Odometry>("uwb_wls", 50);
      fusion_odom_pub_ = nh_.advertise<nav_msgs::Odometry>("uwb_imu_fusion", 50);
      wls_path_pub_ = nh_.advertise<nav_msgs::Path>("uwb_wls_path", 10);
      fusion_path_pub_ = nh_.advertise<nav_msgs::Path>("uwb_imu_fusion_path", 10);
      gt_path_pub_ = nh_.advertise<nav_msgs::Path>("uwb_imu_path_gt", 10);

      ROS_INFO("[uwb_imu_fusion] %zu anchors  meas/cycle=%d",
               anchors_.size(), measurements_per_cycle_);
      ROS_INFO("[uwb_imu_fusion] gravity=%.4f", gravity_mag_);
      ROS_INFO("[uwb_imu_fusion] accel_sigma=%.4f  bias_rw=%.5f  prior_bias_sigma=%.4f",
               accel_sigma_, accel_bias_rw_sigma_, prior_bias_sigma_);
      ROS_INFO("[uwb_imu_fusion] zero_vel_sigma=%.4f  alt_constraint=%s",
               zero_vel_sigma_, use_altitude_constraint_ ? "ON" : "OFF");
      ROS_INFO("[uwb_imu_fusion] Log: %s", traj_log_path_.c_str());
    }

    ~UwbImuFusionNode()
    {
      if (traj_log_.is_open())
        traj_log_.close();
    }

  private:
    // ============================================================================
    void loadParams()
    {
      auto get = [&](const std::string &k, auto &v, auto def)
      {
        if (!nh_.getParam("/uwb_imu_fusion/" + k, v) && !pnh_.getParam(k, v))
          v = def;
      };

      nh_.getParam("imu_topic", imu_topic_, std::string("imu"));
      nh_.getParam("uwb_topic", uwb_topic_, std::string("uwb_tdoa"));
      nh_.getParam("gt_topic", gt_topic_, std::string("/pose_data"));
      nh_.getParam("gravity_magnitude", gravity_mag_, 0.0);
      nh_.getParam("odom_frame", odom_frame_, std::string("map"));
      nh_.getParam("base_frame", base_frame_, std::string("base_link"));
      nh_.getParam("residual_log_path", traj_log_path_,
                   std::string("/tmp/uwb_imu_residuals.csv"));

      // IMU noise — in the SAME units as accel after applying accel_scale

      // UWB
      get("tdoa_noise_sigma", tdoa_sigma_, 0.15);
      get("use_robust_noise", use_robust_noise_, true);
      get("huber_k", huber_k_, 1.345);
      get("cycle_timeout", cycle_timeout_, 0.02);

      int na = 8, mpc = 8, min_imu = 2;
      get("num_anchors", na, 8);
      get("measurements_per_cycle", mpc, 8);
      get("min_imu_per_cycle", min_imu, 2);
      num_anchors_ = na;
      measurements_per_cycle_ = mpc;
      min_imu_per_cycle_ = min_imu;
    }

    void loadingAnchors()
    {
      // Anchors
      XmlRpc::XmlRpcValue axv;

      // Look only in the specific namespace
      bool ok = nh_.getParam("/uwb_imu_fusion/anchors", axv);

      if (ok && axv.getType() == XmlRpc::XmlRpcValue::TypeArray)
      {
        for (int i = 0; i < axv.size(); ++i)
        {
          auto &row = axv[i];
          if (row.getType() == XmlRpc::XmlRpcValue::TypeArray && row.size() == 3)
          {
            anchors_.emplace_back((double)row[0], (double)row[1], (double)row[2]);
          }
        }
      }

      if (anchors_.empty())
      {
        ROS_ERROR("[uwb_imu_fusion] No anchors found at /uwb_imu_fusion/anchors!");
      }

      std::vector<double> ext_t{0, 0, 0};
      if (!nh_.getParam("/uwb_imu_fusion/anchor_extrinsic_translation", ext_t))
        pnh_.getParam("anchor_extrinsic_translation", ext_t);

      Eigen::Vector3d te(ext_t[0], ext_t[1], ext_t[2]);
      for (auto &a : anchors_)
      {
        Eigen::Vector3d pw = Eigen::Vector3d(a.x(), a.y(), a.z()) + te;
        a = gtsam::Point3(pw.x(), pw.y(), pw.z());
      }
    }

    // ============================================================================
    void initializeGTSAM()
    {
      gtsam::ISAM2Params isam_params;
      isam_params.relinearizeThreshold = 0.1;
      isam_params.relinearizeSkip = 20;
      isam_params.factorization = gtsam::ISAM2Params::CHOLESKY;
      // isam_ = boost::make_shared<gtsam::ISAM2>(isam_params);
      isam_ = std::make_unique<gtsam::ISAM2>(isam_params);

      // Priors Noise Model
      double pose_prior_sigma, prior_vel_sigma_, prior_bias_sigma_;
      nh_.getParam("pose_prior_sigma", pose_prior_sigma, 0.05);
      nh_.getParam("vel_prior_sigma", prior_vel_sigma_, 0.1);
      nh_.getParam("bias_prior_sigma", prior_bias_sigma_, 0.3);

      pose_prior_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector(6) << pose_prior_sigma, pose_prior_sigma, pose_prior_sigma,
           pose_prior_sigma, pose_prior_sigma, pose_prior_sigma)
              .finished());
      velocity_prior_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector(3) << prior_vel_sigma_, prior_vel_sigma_, prior_vel_sigma_)
              .finished());
      bias_prior_noise_ = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector(6)
          << prior_bias_sigma_, prior_bias_sigma_, prior_bias_sigma_,
             prior_bias_sigma_, prior_bias_sigma_, prior_bias_sigma_)
          .finished());

      // IMU Noise Model
      double acc_noise_sigma, gyro_noise_sigma, acc_bias_rw_sigma, gyro_bias_rw_sigma;
      nh_.param<double>("imu_acc_noise", acc_noise_sigma, 0.015);
      nh_.param<double>("imu_gyro_noise", gyro_noise_sigma, 0.12);
      nh_.param<double>("imu_acc_bias_noise", acc_bias_rw_sigma, 0.0008);
      nh_.param<double>("imu_gyro_bias_noise", gyro_bias_rw_sigma, 0.0017);

      // Create IMU noise covariance matrices
      gtsam::Matrix33 measured_acc_cov = gtsam::I_3x3 * pow(acc_noise_sigma, 2);
      gtsam::Matrix33 measured_omega_cov = gtsam::I_3x3 * pow(gyro_noise_sigma, 2);
      gtsam::Matrix33 integration_error_cov = gtsam::I_3x3 * 1e-8; // Integration uncertainty
      gtsam::Matrix33 bias_acc_cov = gtsam::I_3x3 * pow(acc_bias_rw_sigma, 2);
      gtsam::Matrix33 bias_omega_cov = gtsam::I_3x3 * pow(gyro_bias_rw_sigma, 2);
      gtsam::Matrix66 bias_acc_omega_init = gtsam::I_6x6 * 1e-5; // Initial bias uncertainty

      // Create IMU preintegration parameters
      auto p = gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedU(gravity_mag_);
      p->accelerometerCovariance = measured_acc_cov;
      p->gyroscopeCovariance = measured_omega_cov;
      p->integrationCovariance = integration_error_cov;
      p->biasAccCovariance = bias_acc_cov;
      p->biasOmegaCovariance = bias_omega_cov;
      p->biasAccOmegaInt = bias_acc_omega_init;

      // Load initial bias values
      double initial_acc_bias_x, initial_acc_bias_y, initial_acc_bias_z;
      double initial_gyro_bias_x, initial_gyro_bias_y, initial_gyro_bias_z;
      nh_.param<double>("initial_acc_bias_x", initial_acc_bias_x, 0.0);
      nh_.param<double>("initial_acc_bias_y", initial_acc_bias_y, 0.0);
      nh_.param<double>("initial_acc_bias_z", initial_acc_bias_z, 0.0);
      nh_.param<double>("initial_gyro_bias_x", initial_gyro_bias_x, 0.0);
      nh_.param<double>("initial_gyro_bias_y", initial_gyro_bias_y, 0.0);
      nh_.param<double>("initial_gyro_bias_z", initial_gyro_bias_z, 0.0);

      // Create initial bias object
      gtsam::imuBias::ConstantBias initial_bias(
          (gtsam::Vector3() << initial_acc_bias_x, initial_acc_bias_y, initial_acc_bias_z).finished(),
          (gtsam::Vector3() << initial_gyro_bias_x, initial_gyro_bias_y, initial_gyro_bias_z).finished());

      // Initialize IMU preintegration with parameters and initial bias
      imu_preintegrated_ = new gtsam::PreintegratedImuMeasurements(p, initial_bias);
      imu_propagation_ = new gtsam::PreintegratedImuMeasurements(p, initial_bias); // For high-freq propagation

      // UWB Noise Model
      double tdoa_sigma;
      nh.getParam("uwb_tdoa_sigma", tdoa_sigma_, 0.5);
      if (use_robust_noise_)
        tdoa_noise_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(huber_k_), tdoa_sigma_);
      else
        tdoa_noise_ = tdoa_sigma_;

      /// Neex to modification part
      nh_.getParam("/uwb_imu_fusion/use_altitude_constraint", use_altitude_constraint_, false);
      nh_.getParam("/uwb_imu_fusion/altitude_z_ref", altitude_z_ref_, 1.0);
      nh_.getParam("/uwb_imu_fusion/altitude_sigma", altitude_sigma_, 0.5);

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

      auto base_noise = gtsam::noiseModel::Isotropic::Sigma(1, tdoa_sigma_);

      zero_vel_noise_ = gtsam::noiseModel::Isotropic::Sigma(3, zero_vel_sigma_);

      if (use_altitude_constraint_)
        alt_noise_ = gtsam::noiseModel::Isotropic::Sigma(1, altitude_sigma_);
    }

    // ============================================================================
    void openLog()
    {
      traj_log_.open(traj_log_path_, std::ios::out | std::ios::trunc);
      if (traj_log_.is_open())
      {
        traj_log_ << "timestamp,"
                  << "gt_x,gt_y,gt_z,"
                  << "wls_x,wls_y,wls_z,"
                  << "fusion_x,fusion_y,fusion_z,"
                  << "fusion_qx,fusion_qy,fusion_qz,fusion_qw,"
                  << "fusion_vx,fusion_vy,fusion_vz,"
                  << "bias_ax,bias_ay,bias_az,"
                  << "wls_gt_err,fusion_gt_err\n";
      }
      else
      {
        ROS_WARN("[uwb_imu_fusion] Cannot open log: %s", traj_log_path_.c_str());
      }
    }

    void logRow(double t, bool wls_ok, const Eigen::Vector3d &wls_p)
    {
      if (!traj_log_.is_open())
        return;
      const gtsam::Point3 &fp = current_pose_.translation();
      Eigen::Quaterniond fq(current_pose_.rotation().matrix());
      auto nan = std::numeric_limits<double>::quiet_NaN();

      double gt_x = gt_have_ ? gt_pos_.x() : nan;
      double gt_y = gt_have_ ? gt_pos_.y() : nan;
      double gt_z = gt_have_ ? gt_pos_.z() : nan;
      double wx = wls_ok ? wls_p.x() : nan;
      double wy = wls_ok ? wls_p.y() : nan;
      double wz = wls_ok ? wls_p.z() : nan;
      double wls_err = (wls_ok && gt_have_)
                           ? (wls_p - Eigen::Vector3d(gt_x, gt_y, gt_z)).norm()
                           : nan;
      double fus_err = gt_have_
                           ? std::sqrt(std::pow(fp.x() - gt_x, 2) + std::pow(fp.y() - gt_y, 2) + std::pow(fp.z() - gt_z, 2))
                           : nan;

      traj_log_ << std::fixed << std::setprecision(6)
                << t << ","
                << gt_x << "," << gt_y << "," << gt_z << ","
                << wx << "," << wy << "," << wz << ","
                << fp.x() << "," << fp.y() << "," << fp.z() << ","
                << fq.x() << "," << fq.y() << "," << fq.z() << "," << fq.w() << ","
                << current_vel_.x() << "," << current_vel_.y() << "," << current_vel_.z() << ","
                << current_bias_.accelerometer().x() << ","
                << current_bias_.accelerometer().y() << ","
                << current_bias_.accelerometer().z() << ","
                << wls_err << "," << fus_err << "\n";
      traj_log_.flush();
    }

    // ============================================================================
    // IMU callback — BUFFER ONLY
    // ============================================================================
    void imuCallback(const sensor_msgs::Imu::ConstPtr &msg)
    {
      std::lock_guard<std::mutex> lk(mutex_);
      const double t = msg->header.stamp.toSec();
      if (last_imu_time_ >= 0.0 && t <= last_imu_time_)
      {
        ROS_WARN_THROTTLE(1.0, "[uwb_imu_fusion] IMU out-of-order %.6f", t);
        return;
      }

      double ax = msg->linear_acceleration.x;
      double ay = msg->linear_acceleration.y;
      double az = msg->linear_acceleration.z;

      double gx = msg->angular_velocity.x;
      double gy = msg->angular_velocity.y;
      double gz = msg->angular_velocity.z;

      ImuMeas s;
      s.t = t;
      s.acc = gtsam::Vector3(ax, ay, az);
      s.gyr = gtsam::Vector3(gx, gy, gz);
      imu_buf_.push_back(s);
  
      while (imu_buf_.size() > 5000)
      {
        imu_buf_.pop_front();
      }

      if (last_imu_time_ > 0)
      {
        double dt = t - last_imu_time_;

        // Sanity check on time delta
        if (dt > 0 && dt < 1.0)
        {
          // Integrate for factor graph (between keyframes)
          imu_preintegrated_->integrateMeasurement(s.acc, s.gyr, dt);

          // Integrate for high-frequency propagation
          if (last_prop_time_ > 0)
          {
            double prop_dt = t - last_prop_time_;
            if (prop_dt > 0 && prop_dt < 1.0)
            {
              imu_propagation_->integrateMeasurement(s.acc, s.gyr, prop_dt);

              // Propagate state forward
              propagateStateForward();

              // Publish high-frequency pose
              publishHighFrequencyPose(t);
            }
          }
          last_prop_time_ = t;
        }
        else
        {
          ROS_WARN("Skipping IMU integration: dt = %.6f", dt);
        }
      }

      last_imu_time_ = t;
    }

    void propagateStateForward()
    {
      // Create navigation state from current optimized state
      gtsam::NavState current_nav_state(current_state_.pose, current_state_.velocity);

      // Predict forward using IMU propagation
      gtsam::NavState propagated_nav_state = imu_propagation_->predict(current_nav_state, current_state_.imu_bias);

      // Update propagated state
      propagated_state_.pose = propagated_nav_state.pose();
      propagated_state_.velocity = propagated_nav_state.v();
      propagated_state_.imu_bias = current_state_.imu_bias; // Bias doesn't change in propagation
      propagated_state_.timestamp = last_prop_time_;
    }

    void publishHighFrequencyPose(double timestamp)
    {
      // Create and publish high-frequency odometry message
      nav_msgs::Odometry odom_msg;
      odom_msg.header.stamp = ros::Time(timestamp);
      odom_msg.header.frame_id = world_frame_;
      odom_msg.child_frame_id = body_frame_;

      // Set position
      odom_msg.pose.pose.position.x = propagated_state_.pose.translation().x();
      odom_msg.pose.pose.position.y = propagated_state_.pose.translation().y();
      odom_msg.pose.pose.position.z = propagated_state_.pose.translation().z();

      // Set orientation (convert to quaternion)
      Quaternion q = propagated_state_.pose.rotation().toQuaternion();
      odom_msg.pose.pose.orientation.w = q.w();
      odom_msg.pose.pose.orientation.x = q.x();
      odom_msg.pose.pose.orientation.y = q.y();
      odom_msg.pose.pose.orientation.z = q.z();

      // Set velocity
      odom_msg.twist.twist.linear.x = propagated_state_.velocity.x();
      odom_msg.twist.twist.linear.y = propagated_state_.velocity.y();
      odom_msg.twist.twist.linear.z = propagated_state_.velocity.z();

      // Add covariance information (simplified diagonal covariance)
      // Position covariance increases with time since last optimization
      double time_since_opt = timestamp - current_state_.timestamp;
      double pos_variance = 0.01 + 0.1 * time_since_opt; // Simple linear growth
      double vel_variance = 0.001 + 0.01 * time_since_opt;

      for (int i = 0; i < 3; i++)
      {
        odom_msg.pose.covariance[i * 7] = pos_variance;
        odom_msg.twist.covariance[i * 7] = vel_variance;
      }
      for (int i = 3; i < 6; i++)
      {
        odom_msg.pose.covariance[i * 7] = 0.01; // Orientation uncertainty
      }

      odom_high_freq_pub_.publish(odom_msg);

      // Update high-frequency pose count
      high_freq_pose_count_++;

      // Broadcast TF transform at high frequency
      geometry_msgs::TransformStamped transform;
      transform.header = odom_msg.header;
      transform.child_frame_id = body_frame_ + "_high_freq";
      transform.transform.translation.x = odom_msg.pose.pose.position.x;
      transform.transform.translation.y = odom_msg.pose.pose.position.y;
      transform.transform.translation.z = odom_msg.pose.pose.position.z;
      transform.transform.rotation = odom_msg.pose.pose.orientation;

      tf_broadcaster_.sendTransform(transform);

      // Optionally update high-frequency path (decimated to avoid too many points)
      if (high_freq_pose_count_ % 10 == 0)
      { // Every 10th pose
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = odom_msg.header;
        pose_stamped.pose = odom_msg.pose.pose;
        high_freq_path_msg_.poses.push_back(pose_stamped);

        if (high_freq_path_msg_.poses.size() > 5000)
        {
          high_freq_path_msg_.poses.erase(high_freq_path_msg_.poses.begin());
        }
      }
    }
    // ============================================================================
    void uwbCallback(const cf_msgs::Tdoa::ConstPtr &msg)
    {
      std::lock_guard<std::mutex> lk(mutex_);
      if (msg->idA < 0 || msg->idB < 0 ||
          msg->idA >= (int)anchors_.size() ||
          msg->idB >= (int)anchors_.size())
      {
        ROS_WARN_THROTTLE(2.0, "[uwb_imu_fusion] Anchor OOB (%d,%d)", msg->idA, msg->idB);
        return;
      }

      TdoaMeas ts{msg->header.stamp.toSec(), msg->idA, msg->idB, msg->data};
      // Check cycle timeout, the first and last are too far.
      if (!current_cycle_.empty() &&
          (ts.t - current_cycle_.front().t) > cycle_timeout_)
      {
        current_cycle_.clear();
        cycle_pairs_.clear();
      }

      // check if the pair in cycle.
      auto pr = std::make_pair(std::min(ts.idA, ts.idB), std::max(ts.idA, ts.idB));
      if (cycle_pairs_.count(pr))
      {
        //
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

    // ============================================================================
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
      ps.pose.position = msg->pose.pose.position;
      ps.pose.orientation = msg->pose.pose.orientation;
      gt_path_.header.stamp = msg->header.stamp;
      gt_path_.header.frame_id = odom_frame_;
      gt_path_.poses.push_back(ps);
      gt_path_pub_.publish(gt_path_);
    }

    // ============================================================================
    void processCycle(const std::vector<TdoaMeas> &meas)
    {
      double t_mid = 0.0;
      for (const auto &m : meas)
        t_mid += m.t;
      t_mid /= (double)meas.size();
      const ros::Time stamp(t_mid);

      Eigen::Vector3d wls_p = wls_guess_;
      bool wls_ok = wls_solver_.solve(meas, anchors_, wls_p);
      if (wls_ok)
        wls_guess_ = wls_p;
      // WLS solve inital solution for factor graph

      if (!initialized_)
        initGraph(t_mid, meas);
      else
        updateGraph(t_mid, meas);

      const gtsam::Point3 &fp = current_pose_.translation();
      const auto rpy = current_pose_.rotation().rpy();
      const auto &ba = current_bias_.accelerometer();

      ROS_INFO("=================================================");
      ROS_INFO("[Cycle %4zu | t=%.3f s]", cycle_index_, t_mid);
      if (wls_ok)
        ROS_INFO("  WLS    : [%7.3f, %7.3f, %7.3f]", wls_p.x(), wls_p.y(), wls_p.z());
      else
        ROS_INFO("  WLS    : DID NOT CONVERGE");
      ROS_INFO("  FUSION : pos=[%7.3f,%7.3f,%7.3f]  rpy=[%.1f°,%.1f°,%.1f°]",
               fp.x(), fp.y(), fp.z(),
               rpy.x() * 180 / M_PI, rpy.y() * 180 / M_PI, rpy.z() * 180 / M_PI);
      ROS_INFO("           vel=[%6.3f,%6.3f,%6.3f]  bias_a=[%.4f,%.4f,%.4f]",
               current_vel_.x(), current_vel_.y(), current_vel_.z(),
               ba.x(), ba.y(), ba.z());
      if (gt_have_)
      {
        double ew = wls_ok
                        ? (wls_p - Eigen::Vector3d(gt_pos_.x(), gt_pos_.y(), gt_pos_.z())).norm()
                        : -1;
        double ef = std::sqrt(std::pow(fp.x() - gt_pos_.x(), 2) +
                              std::pow(fp.y() - gt_pos_.y(), 2) +
                              std::pow(fp.z() - gt_pos_.z(), 2));
        ROS_INFO("  GT     : [%7.3f,%7.3f,%7.3f]", gt_pos_.x(), gt_pos_.y(), gt_pos_.z());
        if (ew >= 0)
          ROS_INFO("  |WLS    - GT| = %.4f m", ew);
        ROS_INFO("  |FUSION - GT| = %.4f m", ef);
      }
      ROS_INFO("=================================================");

      if (wls_ok)
      {
        publishOdom(wls_odom_pub_, stamp, wls_p.x(), wls_p.y(), wls_p.z(),
                    gtsam::Rot3::identity());
        appendPath(wls_path_, wls_path_pub_, stamp,
                   wls_p.x(), wls_p.y(), wls_p.z(), gtsam::Rot3::identity());
      }
      publishOdom(fusion_odom_pub_, stamp, fp.x(), fp.y(), fp.z(),
                  current_pose_.rotation());
      appendPath(fusion_path_, fusion_path_pub_, stamp,
                 fp.x(), fp.y(), fp.z(), current_pose_.rotation());

      Eigen::Quaterniond q(current_pose_.rotation().matrix());
      geometry_msgs::TransformStamped tf;
      tf.header.stamp = stamp;
      tf.header.frame_id = odom_frame_;
      tf.child_frame_id = base_frame_;
      tf.transform.translation.x = fp.x();
      tf.transform.translation.y = fp.y();
      tf.transform.translation.z = fp.z();
      tf.transform.rotation.x = q.x();
      tf.transform.rotation.y = q.y();
      tf.transform.rotation.z = q.z();
      tf.transform.rotation.w = q.w();
      tf_broadcaster_.sendTransform(tf);

      logRow(t_mid, wls_ok, wls_p);
    }

    // ============================================================================
    void initGraph(double t_mid, const std::vector<TdoaMeas> &meas)
    {
      cycle_index_ = 0;
      current_pose_ = initial_pose_;
      current_vel_ = gtsam::Vector3::Zero();
      current_bias_ = gtsam::imuBias::ConstantBias();

      gtsam::NonlinearFactorGraph graph;
      gtsam::Values values;

      graph.addPrior(X(0), current_pose_,
                     gtsam::noiseModel::Diagonal::Sigmas(prior_pose_sigmas_));
      graph.addPrior(V(0), current_vel_,
                     gtsam::noiseModel::Isotropic::Sigma(3, prior_vel_sigma_));
      // FIXED: prior_bias_sigma = 0.3 (was 0.001) → bias can be estimated freely
      graph.addPrior(B(0), current_bias_,
                     gtsam::noiseModel::Isotropic::Sigma(6, prior_bias_sigma_));

      for (const auto &m : meas)
        graph.add(TdoaFactor(X(0), anchors_[m.idA], anchors_[m.idB],
                             -m.tdoa, tdoa_noise_));

      if (use_altitude_constraint_)
        graph.add(boost::make_shared<AltitudeFactor>(X(0), altitude_z_ref_, alt_noise_));

      values.insert(X(0), current_pose_);
      values.insert(V(0), current_vel_);
      values.insert(B(0), current_bias_);

      isam_->update(graph, values);
      isam_->update();
      gtsam::Values est = isam_->calculateEstimate();
      current_pose_ = est.at<gtsam::Pose3>(X(0));
      current_vel_ = est.at<gtsam::Vector3>(V(0));
      current_bias_ = est.at<gtsam::imuBias::ConstantBias>(B(0));

      preint_->resetIntegrationAndSetBias(current_bias_);
      initialized_ = true;
      last_cycle_t_ = t_mid;

      ROS_INFO("[uwb_imu_fusion] Init at t=%.3f  pos=[%.3f,%.3f,%.3f]",
               t_mid, current_pose_.x(), current_pose_.y(), current_pose_.z());
    }

    // ============================================================================
    void updateGraph(double t_mid, const std::vector<TdoaMeas> &meas)
    {
      // Replay IMU buffer between keyframes — single authoritative integration
      preint_->resetIntegrationAndSetBias(current_bias_);
      int n_imu = 0;
      double prev_t = last_cycle_t_;

      for (const auto &imu : imu_buf_)
      {
        if (imu.t <= last_cycle_t_)
          continue;
        if (imu.t > t_mid)
          break;
        const double dt = imu.t - prev_t;
        if (dt <= 0.0 || dt > 1.0)
        {
          prev_t = imu.t;
          continue;
        }
        preint_->integrateMeasurement(imu.acc, imu.gyr, dt);
        prev_t = imu.t;
        ++n_imu;
      }

      if (n_imu < min_imu_per_cycle_)
      {
        ROS_WARN_THROTTLE(1.0, "[uwb_imu_fusion] %d IMU (need %d); skipping.", n_imu, min_imu_per_cycle_);
        last_cycle_t_ = t_mid;
        return;
      }

      ++cycle_index_;

      gtsam::NonlinearFactorGraph graph;
      gtsam::Values values;

      // IMU factor — connects (X,V,B)(k-1) to (X,V)(k)
      // Gyroscope measurements used internally to track rotation at each step
      graph.add(gtsam::ImuFactor(
          X(cycle_index_ - 1), V(cycle_index_ - 1),
          X(cycle_index_), V(cycle_index_),
          B(cycle_index_ - 1), *preint_));

      // Bias random-walk — FIXED: accel_bias_rw_sigma = 0.01 (was 0.0004)
      // This allows the graph to track the real ~0.47 m/s² accelerometer bias
      const double tau = std::max(t_mid - last_cycle_t_, 1e-3);
      gtsam::Vector6 bias_sigmas;
      bias_sigmas << gtsam::Vector3::Constant(accel_bias_rw_sigma_ * std::sqrt(tau)),
          gtsam::Vector3::Constant(gyro_bias_rw_sigma_ * std::sqrt(tau));
      graph.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
          B(cycle_index_ - 1), B(cycle_index_),
          gtsam::imuBias::ConstantBias(),
          gtsam::noiseModel::Diagonal::Sigmas(bias_sigmas)));

      // TDOA factors — constrain translation only
      for (const auto &m : meas)
        graph.add(TdoaFactor(X(cycle_index_), anchors_[m.idA], anchors_[m.idB],
                             -m.tdoa, tdoa_noise_));

      // Zero-velocity prior — FIXED: sigma=0.05 (was 0.3), tighter damping
      // Prevents bias from integrating into velocity while bias is being estimated
      graph.add(boost::make_shared<ZeroVelocityFactor>(V(cycle_index_), zero_vel_noise_));

      // Altitude prior — FIXED: now actually wired in (was TODO in YAML)
      if (use_altitude_constraint_)
        graph.add(boost::make_shared<AltitudeFactor>(
            X(cycle_index_), altitude_z_ref_, alt_noise_));

      // Initial value prediction from IMU propagation
      const gtsam::NavState prev_nav(current_pose_, current_vel_);
      const gtsam::NavState pred_nav = preint_->predict(prev_nav, current_bias_);
      values.insert(X(cycle_index_), pred_nav.pose());
      values.insert(V(cycle_index_), pred_nav.velocity());
      values.insert(B(cycle_index_), current_bias_);

      isam_->update(graph, values);
      isam_->update();
      const gtsam::Values est = isam_->calculateEstimate();

      current_pose_ = est.at<gtsam::Pose3>(X(cycle_index_));
      current_vel_ = est.at<gtsam::Vector3>(V(cycle_index_));
      current_bias_ = est.at<gtsam::imuBias::ConstantBias>(B(cycle_index_));

      preint_->resetIntegrationAndSetBias(current_bias_);
      last_cycle_t_ = t_mid;
    }

    // ============================================================================
    void publishOdom(ros::Publisher &pub, const ros::Time &stamp,
                     double x, double y, double z, const gtsam::Rot3 &rot)
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
      odom.twist.twist.linear.x = current_vel_.x();
      odom.twist.twist.linear.y = current_vel_.y();
      odom.twist.twist.linear.z = current_vel_.z();
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

    // ============================================================================;
  };

} // namespace uwb_imu_fusion

int main(int argc, char **argv)
{
  ros::init(argc, argv, "uwb_imu_fusion_node");
  uwb_imu_fusion::UwbImuFusionNode node;
  ros::AsyncSpinner spinner(2);
  spinner.start();
  ros::waitForShutdown();
  return 0;
}