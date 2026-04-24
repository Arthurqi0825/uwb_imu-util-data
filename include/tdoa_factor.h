#pragma once

// Custom GTSAM factor for a UWB TDOA measurement between two anchors A and B.
// Measurement model: z = ||p - p_A|| - ||p - p_B||, where p is the translation
// of the body Pose3. The factor is unary on Pose3 (the anchor positions are
// assumed fixed/known).

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <Eigen/Core>
#include <cmath>

namespace uwb_imu_fusion {

class TdoaFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
 public:
  using Base = gtsam::NoiseModelFactor1<gtsam::Pose3>;

  TdoaFactor(gtsam::Key key,
             const gtsam::Point3& anchor_A,
             const gtsam::Point3& anchor_B,
             double measured_tdoa,
             const gtsam::SharedNoiseModel& model)
      : Base(model, key),
        anchor_A_(anchor_A),
        anchor_B_(anchor_B),
        measured_(measured_tdoa) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& pose,
      boost::optional<gtsam::Matrix&> H = boost::none) const override {
    const gtsam::Point3 p = pose.translation();
    const gtsam::Vector3 dA = p - anchor_A_;
    const gtsam::Vector3 dB = p - anchor_B_;

    const double eps = 1e-6;
    const double distA = std::max(dA.norm(), eps);
    const double distB = std::max(dB.norm(), eps);

    const double predicted = distA - distB;
    gtsam::Vector1 err;
    err(0) = predicted - measured_;

    if (H) {
      // d(distA - distB)/dp_world = dA/distA - dB/distB   (1x3 in world frame)
      Eigen::RowVector3d dH_dp =
          (dA.transpose() / distA) - (dB.transpose() / distB);

      // We need d/d(Pose3) where Pose3 has a 6-DoF tangent [rot(3), trans(3)].
      // GTSAM's Pose3::translation Jacobian w.r.t. the Pose3 tangent is
      // [ -R*skew(t_local), R ] but we can bypass that by using
      // pose.translation(Hp) to get the 3x6 jacobian of translation wrt pose.
      gtsam::Matrix36 Ht;
      (void)pose.translation(Ht);  // fills Ht
      gtsam::Matrix16 J = dH_dp * Ht;  // 1x6
      *H = J;
    }
    return err;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new TdoaFactor(*this)));
  }

  const gtsam::Point3& anchorA() const { return anchor_A_; }
  const gtsam::Point3& anchorB() const { return anchor_B_; }
  double measured() const { return measured_; }

 private:
  gtsam::Point3 anchor_A_;
  gtsam::Point3 anchor_B_;
  double measured_;
};

}  // namespace uwb_imu_fusion
