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
    if (static_cast<int>(meas.size()) < kMinMeasurements) return false;

    for (int iter = 0; iter < max_iter_; ++iter) {
      const int n = static_cast<int>(meas.size());

      // Build the Jacobian J (n×3) and residual vector r (n×1).
      Eigen::MatrixXd J(n, 3);
      Eigen::VectorXd r(n);

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
      }

      // Normal equations: (JᵀJ) dp = -Jᵀr
      const Eigen::Vector3d dp =
          -(J.transpose() * J).ldlt().solve(J.transpose() * r);
      p += dp;

      if (dp.norm() < tol_) {
        ROS_WARN("[TdoaWlsSolver] Converged in %d iterations "
                 "(residual norm=%.3f), dp norm=%.3f",
                 iter + 1, r.norm(), dp.norm());
        return true;
      }
    }

    // Iteration limit reached without convergence.
    return false;
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
};

}  // namespace uwb_imu_fusion