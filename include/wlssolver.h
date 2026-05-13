// =============================================================================
// wlssolver.h  —  Gauss-Newton Weighted Least Squares (WLS) TDOA solver
//
// Solves the tag position from a set of Time Difference of Arrival (TDOA)
// measurements using iterative Gauss-Newton linearisation.
//
// RESIDUAL SIGN CONVENTION (validated on real data — DO NOT CHANGE):
//   r(i) = (||p - aB|| - ||p - aA||) - tdoa_measured
//
// where tdoa_measured = d(tag, B) - d(tag, A)
//
// USAGE:
//   TdoaWlsSolver solver(50, 1e-7);  // max_iter=50, tol=1e-7 m
//   Eigen::Vector3d p = initial_guess;
//   bool ok = solver.solve(measurements, anchors, p);
//   // Optional weighted refinement:
//   // bool ok_w = solver.solveWeighted(measurements, anchors, sigmas, p);
//   // p holds the solution if ok == true
//
// THREAD SAFETY:
//   solve() is const and stateless — safe to call from multiple threads
//   provided each thread supplies its own position vector.
// =============================================================================

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <ros/ros.h>        // ROS_WARN — remove if building outside ROS
#include <gtsam/geometry/Point3.h>

namespace uwb_imu_fusion {

// =============================================================================
// TdoaMeas — one TDOA range-difference measurement
//
// Represents  d(tag, anchor[idB]) − d(tag, anchor[idA])  in metres.
// Defined here so wlssolver.h is self-contained and can be included in any
// translation unit without a prior definition of TdoaMeas.
// =============================================================================
struct TdoaMeas {
  double t;     ///< ROS timestamp [s]
  int    idA;   ///< Index into the anchor array for anchor A
  int    idB;   ///< Index into the anchor array for anchor B
  double tdoa;  ///< d(tag,B) − d(tag,A)  [m]
};

// =============================================================================
// TdoaWlsSolver
//
// Gauss-Newton WLS solver for tag position from TDOA measurements.
//
// Parameters
// ----------
// max_iter : int   — maximum number of Gauss-Newton iterations  (default 50)
// tol      : double — convergence tolerance on the step norm [m] (default 1e-7)
// =============================================================================
class TdoaWlsSolver {
 public:
  // ---------------------------------------------------------------------------
  // Constructor
  // ---------------------------------------------------------------------------
  /// @param max_iter  Maximum Gauss-Newton iterations before declaring failure.
  /// @param tol       Convergence threshold on ||dp|| [m].
  explicit TdoaWlsSolver(int max_iter = 50, double tol = 1e-7)
      : max_iter_(max_iter), tol_(tol) {}

  // ---------------------------------------------------------------------------
  // solve()
  //
  // Runs Gauss-Newton WLS to find the tag position p that minimises the sum of
  // squared TDOA residuals.
  //
  // @param meas     Vector of TDOA measurements.  Requires >= 3 for a solution.
  // @param anchors  Anchor positions indexed by TdoaMeas::idA / idB.
  // @param p        [in/out]  Initial guess on entry; solution on exit.
  //
  // @return true  if the solver converged within max_iter_ iterations.
  //         false if the minimum measurement count is not met or iteration
  //               limit is reached without convergence.
  // ---------------------------------------------------------------------------
  bool solve(const std::vector<TdoaMeas>&      meas,
             const std::vector<gtsam::Point3>& anchors,
             Eigen::Vector3d&                  p) const {
    return solveWeighted(meas, anchors, Eigen::VectorXd(), p);
  }

  // ---------------------------------------------------------------------------
  // solveWeighted()
  //
  // Same Gauss-Newton solver, but with per-measurement sigmas. If sigma.size()
  // does not match meas.size(), this falls back to unit weights.
  // ---------------------------------------------------------------------------
  bool solveWeighted(const std::vector<TdoaMeas>&      meas,
                     const std::vector<gtsam::Point3>& anchors,
                     const Eigen::VectorXd&            sigma,
                     Eigen::Vector3d&                  p) const {
    if (static_cast<int>(meas.size()) < kMinMeasurements) return false;

    const bool use_weights =
        (sigma.size() == static_cast<int>(meas.size()));

    for (int iter = 0; iter < max_iter_; ++iter) {
      const int n = static_cast<int>(meas.size());

      // Build the Jacobian J (n×3) and residual vector r (n×1).
      Eigen::MatrixXd J(n, 3);
      Eigen::VectorXd r(n);
      Eigen::VectorXd w = Eigen::VectorXd::Ones(n);

      for (int i = 0; i < n; ++i) {
        const auto& m = meas[i];

        const Eigen::Vector3d aA(anchors[m.idA].x(),
                                 anchors[m.idA].y(),
                                 anchors[m.idA].z());
        const Eigen::Vector3d aB(anchors[m.idB].x(),
                                 anchors[m.idB].y(),
                                 anchors[m.idB].z());

        const Eigen::Vector3d dA = p - aA;
        const Eigen::Vector3d dB = p - aB;

        // Guard against division by zero at exactly the anchor location.
        const double rA = std::max(dA.norm(), kMinDist);
        const double rB = std::max(dB.norm(), kMinDist);

        // Residual: r(i) = (||p-aB|| - ||p-aA||) - tdoa_measured
        r(i)     = (rB - rA) - m.tdoa;
        J.row(i) = (dB / rB - dA / rA).transpose();

        if (use_weights) {
          const double s = std::max(sigma(i), kMinDist);
          w(i) = 1.0 / (s * s);
        }
      }

      // Weighted normal equations: (JᵀWJ) dp = -JᵀWr
      const Eigen::Matrix3d JT_W_J = J.transpose() * w.asDiagonal() * J;
      const Eigen::Vector3d JT_W_r = J.transpose() * (w.asDiagonal() * r);
      const Eigen::Vector3d dp =
          -JT_W_J.ldlt().solve(JT_W_r);
      p += dp;

      if (dp.norm() < tol_) {
        ROS_DEBUG("[TdoaWlsSolver] Converged in %d iterations "
                  "(residual norm=%.3f), dp norm=%.3f",
                  iter + 1, r.norm(), dp.norm());
        return true;
      }
    }

    // Iteration limit reached without convergence.
    return false;
  }

  // ---------------------------------------------------------------------------
  // solveRobustWeighted()
  //
  // Iteratively re-weighted WLS using a Huber loss in normalized residual space.
  // Large post-fit residuals keep the original measurement but inflate its
  // effective sigma, which is usually safer than dropping TDOAs until the
  // geometry has too little redundancy.
  // ---------------------------------------------------------------------------
  bool solveRobustWeighted(const std::vector<TdoaMeas>&      meas,
                           const std::vector<gtsam::Point3>& anchors,
                           const Eigen::VectorXd&            sigma,
                           Eigen::Vector3d&                  p,
                           double                            huber_k,
                           double                            max_sigma_scale,
                           int                               outer_iter,
                           Eigen::VectorXd*                  effective_sigma = nullptr) const {
    if (static_cast<int>(meas.size()) < kMinMeasurements) return false;
    if (sigma.size() != static_cast<int>(meas.size())) {
      const bool ok = solveWeighted(meas, anchors, sigma, p);
      if (effective_sigma) *effective_sigma = sigma;
      return ok;
    }

    const int n = static_cast<int>(meas.size());
    huber_k = std::max(huber_k, 1e-6);
    max_sigma_scale = std::max(max_sigma_scale, 1.0);
    outer_iter = std::max(1, outer_iter);

    Eigen::VectorXd eff_sigma = sigma;
    bool ok = false;
    for (int outer = 0; outer < outer_iter; ++outer) {
      ok = solveWeighted(meas, anchors, eff_sigma, p);
      if (!ok) break;

      Eigen::VectorXd residuals(n);
      computeResiduals(meas, anchors, p, residuals);

      Eigen::VectorXd next_sigma = sigma;
      for (int i = 0; i < n; ++i) {
        const double base_sigma = std::max(sigma(i), kMinDist);
        const double std_abs = std::fabs(residuals(i)) / base_sigma;
        if (std_abs > huber_k) {
          const double huber_weight = huber_k / std_abs;
          const double scale = std::min(max_sigma_scale,
                                        1.0 / std::sqrt(std::max(huber_weight, kMinDist)));
          next_sigma(i) = base_sigma * scale;
        }
      }

      if ((next_sigma - eff_sigma).norm() < 1e-6) break;
      eff_sigma = next_sigma;
    }

    if (ok) ok = solveWeighted(meas, anchors, eff_sigma, p);
    if (effective_sigma) *effective_sigma = eff_sigma;
    return ok;
  }

  // ---------------------------------------------------------------------------
  // Accessors
  // ---------------------------------------------------------------------------
  int    maxIter() const { return max_iter_; }
  double tol()     const { return tol_; }

 private:
  int    max_iter_;
  double tol_;

  /// Minimum number of TDOA measurements required for a 3-D solution.
  static constexpr int    kMinMeasurements = 3;
  /// Minimum distance used in the denominator to prevent division by zero.
  static constexpr double kMinDist         = 1e-9;

  static void computeResiduals(const std::vector<TdoaMeas>&      meas,
                               const std::vector<gtsam::Point3>& anchors,
                               const Eigen::Vector3d&            p,
                               Eigen::VectorXd&                  r) {
    const int n = static_cast<int>(meas.size());
    r.resize(n);
    for (int i = 0; i < n; ++i) {
      const auto& m = meas[i];
      const Eigen::Vector3d aA(anchors[m.idA].x(),
                               anchors[m.idA].y(),
                               anchors[m.idA].z());
      const Eigen::Vector3d aB(anchors[m.idB].x(),
                               anchors[m.idB].y(),
                               anchors[m.idB].z());
      const double rA = std::max((p - aA).norm(), kMinDist);
      const double rB = std::max((p - aB).norm(), kMinDist);
      r(i) = (rB - rA) - m.tdoa;
    }
  }
};

}  // namespace uwb_imu_fusion
