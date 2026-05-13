#pragma once

// Integrity-aware TDOA factor for tightly coupled UWB/IMU optimization.
//
// The residual is the normal TDOA geometry. Integrity monitoring is applied by
// the node before graph insertion: unhealthy measurements are excluded, and
// suspicious measurements receive larger sigma/robust noise. This factor keeps
// metadata so graph contents can be audited in logs/debugging.

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace uwb_imu_fusion {

class IntegrityTdoaFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
 public:
  using Base = gtsam::NoiseModelFactor1<gtsam::Pose3>;

  IntegrityTdoaFactor(gtsam::Key key,
                      const gtsam::Point3& anchor_A,
                      const gtsam::Point3& anchor_B,
                      double measured_tdoa,
                      const gtsam::SharedNoiseModel& model,
                      int id_a,
                      int id_b,
                      double integrity_sigma,
                      double standardized_residual,
                      const std::string& integrity_label)
      : Base(model, key),
        anchor_A_(anchor_A),
        anchor_B_(anchor_B),
        measured_(measured_tdoa),
        id_a_(id_a),
        id_b_(id_b),
        integrity_sigma_(integrity_sigma),
        standardized_residual_(standardized_residual),
        integrity_label_(integrity_label) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& pose,
      boost::optional<gtsam::Matrix&> H = boost::none) const override {
    const gtsam::Point3 p = pose.translation();
    const gtsam::Vector3 dA = p - anchor_A_;
    const gtsam::Vector3 dB = p - anchor_B_;

    const double eps = 1e-6;
    const double distA = std::max(dA.norm(), eps);
    const double distB = std::max(dB.norm(), eps);

    gtsam::Vector1 err;
    err(0) = (distA - distB) - measured_;

    if (H) {
      const Eigen::RowVector3d dH_dp =
          (dA.transpose() / distA) - (dB.transpose() / distB);
      gtsam::Matrix36 Ht;
      (void)pose.translation(Ht);
      *H = dH_dp * Ht;
    }
    return err;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new IntegrityTdoaFactor(*this)));
  }

  int anchorAId() const { return id_a_; }
  int anchorBId() const { return id_b_; }
  double integritySigma() const { return integrity_sigma_; }
  double standardizedResidual() const { return standardized_residual_; }
  const std::string& integrityLabel() const { return integrity_label_; }

 private:
  gtsam::Point3 anchor_A_;
  gtsam::Point3 anchor_B_;
  double measured_;
  int id_a_;
  int id_b_;
  double integrity_sigma_;
  double standardized_residual_;
  std::string integrity_label_;
};

}  // namespace uwb_imu_fusion
