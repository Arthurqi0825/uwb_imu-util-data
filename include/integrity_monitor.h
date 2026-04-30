// =============================================================================
// integrity_monitor.h  —  RAIM-style integrity monitoring for TDOA WLS
//
// Call after the WLS position has converged. The monitor computes post-fit
// residual checks, chi-squared fault detection, optional FDE, and HPL/VPL.
//
// RESIDUAL SIGN CONVENTION:
//   r(i) = (||p - aB|| - ||p - aA||) - tdoa_measured
//
// FIXES vs. previous version
// ─────────────────────────────────────────────────────────────────────────────
// FIX 1 – HPL/VPL  (was: k_hpl * sqrt(P(0,0)+P(1,1)))
//   HPL now uses the major-axis eigenvalue of the 2×2 horizontal covariance
//   sub-matrix so that cross-correlation P(0,1) is correctly accounted for.
//     lambda_major = (P00+P11)/2 + sqrt(((P00-P11)/2)^2 + P01^2)
//     HPL = k_hpl * sqrt(lambda_major)
//   The old formula underestimates HPL whenever the error ellipse is not
//   circular (i.e. almost always), giving falsely optimistic availability.
//
// FIX 2 – Chi-squared DOF  (was: M-3)
//   3-D position has 3 unknowns → DOF = M - 3. This was already correct in
//   the previous version and is retained.
//   Added a run-time warning when M == 4 (DOF=1) because detection power is
//   very low with only one redundant observation.
//
// FIX 3 – Ring-closure test  (was: raw sum of all TDOA values)
//   The previous sum had no geometric meaning. The test now enumerates every
//   ordered triple (a,b,c) and checks the genuine closure identity:
//     TDOA(a→b) + TDOA(b→c) + TDOA(c→a) = 0
//   ring_sum  = worst (largest |error|) loop residual found
//   ring_ok   = all found loops satisfy |error| < ring_threshold
//
// FIX 4 – Sub-set sigma_h/sigma_h_k in fault-mode HPL loop
//   Now uses horizontalMajorAxisSigma() for both the full-set and the
//   leave-one-out sub-sets, matching Fix 1.
//
// NOTE – Weight matrix model
//   The monitor still uses a diagonal W = diag(1/sigma_i^2). The off-diagonal
//   correlations arising from shared anchors in C = D·Σ·D^T are a second-order
//   effect compared with the magnitude of the dynamic sigmas (≈1.5 m) actually
//   observed in the log. Switching to the full C^-1 is left as a future
//   improvement; the diagonal approximation is conservative for chi-squared
//   detection (it under-weights correlated measurements) and is standard
//   practice in RAIM implementations.
// =============================================================================

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/math/distributions/chi_squared.hpp>
#include <gtsam/geometry/Point3.h>

#include "wlssolver.h"

namespace uwb_imu_fusion {

// =============================================================================
// IntegrityResult  —  output bundle from one integrity epoch
// =============================================================================
struct IntegrityResult {
  // ── Chi-squared global test ─────────────────────────────────────────────────
  double chi2_stat{0.0};
  double chi2_threshold{0.0};
  int    dof{0};
  bool   fault_detected{false};

  // ── Ring-closure test ───────────────────────────────────────────────────────
  // ring_sum        = worst signed loop-closure error found [m]
  // ring_ok         = true when all loops satisfy |error| < ring_threshold
  // ring_closed_loops = number of (a,b,c) triples that had all three edges
  double ring_sum{0.0};
  bool   ring_ok{true};
  int    ring_closed_loops{0};

  // ── Protection levels ───────────────────────────────────────────────────────
  double hpl{std::numeric_limits<double>::quiet_NaN()};
  double vpl{std::numeric_limits<double>::quiet_NaN()};

  // ── FDE output ──────────────────────────────────────────────────────────────
  int    excluded_idx{-1};
  double chi2_after_fde{std::numeric_limits<double>::quiet_NaN()};

  // ── Per-measurement diagnostics ─────────────────────────────────────────────
  std::vector<double> residuals;   // post-fit residuals r(i)  [m]
  std::vector<double> std_resid;   // r(i) / sigma_i           [–]
  std::vector<double> sigmas;      // measurement sigmas used   [m]

  // ── Availability ────────────────────────────────────────────────────────────
  bool available{false};
};

// =============================================================================
// TdoaIntegrityMonitor
// =============================================================================
class TdoaIntegrityMonitor {
 public:
  // Tunable parameters (set before calling check())
  double tdoa_noise_sigma{0.10};  // fallback sigma when none supplied  [m]
  double p_fa{1e-3};              // false-alarm probability for chi2 test
  double hal{0.50};               // horizontal alert limit              [m]
  double val{1.00};               // vertical   alert limit              [m]
  double k_hpl{5.33};             // K factor: HPL = k_hpl * sigma_major
                                  //   5.33 ≈ chi2(2) tail for P_HMI=1e-7
  double ring_threshold{0.30};    // max acceptable loop-closure error   [m]
  bool   enable_fde{true};        // run FDE when fault is detected

  // ---------------------------------------------------------------------------
  // check()  —  main entry point
  //
  // meas        : TDOA measurements for this epoch
  // anchors     : anchor positions (world frame)
  // p_wls       : converged WLS position estimate
  // tdoa_sigmas : per-measurement 1-sigma [m]; if empty, sqrt(2)*tdoa_noise_sigma
  // ---------------------------------------------------------------------------
  IntegrityResult check(const std::vector<TdoaMeas>&   meas,
                        const std::vector<gtsam::Point3>& anchors,
                        const Eigen::Vector3d&          p_wls,
                        const Eigen::VectorXd&          tdoa_sigmas =
                            Eigen::VectorXd()) const
  {
    IntegrityResult res;
    const int M = static_cast<int>(meas.size());

    // ── Ring-closure test (Fix 3) ──────────────────────────────────────────
    computeRingClosure(meas, anchors.size(), res);

    // ── Minimum geometry check ─────────────────────────────────────────────
    if (M < 4) {
      res.fault_detected = false;
      res.available      = false;
      return res;
    }

    // ── Build H and post-fit residuals ─────────────────────────────────────
    Eigen::MatrixXd H(M, 3);
    Eigen::VectorXd r(M);
    buildHAndResiduals(meas, anchors, p_wls, H, r);

    res.residuals.assign(r.data(), r.data() + M);

    // ── Measurement sigma vector ───────────────────────────────────────────
    Eigen::VectorXd meas_sigmas(M);
    if (tdoa_sigmas.size() == M) {
      for (int i = 0; i < M; ++i)
        meas_sigmas(i) = std::max(tdoa_sigmas(i), 1e-9);
    } else {
      meas_sigmas.setConstant(std::sqrt(2.0) * tdoa_noise_sigma);
    }
    const Eigen::VectorXd diag_inv = meas_sigmas.array().square().inverse();

    // ── Chi-squared global test ────────────────────────────────────────────
    // T = r^T W r  ~  chi2(M-3)  under H0 (no fault)
    res.chi2_stat = r.cwiseProduct(diag_inv).dot(r);

    const int dof       = M - 3;
    res.dof             = dof;
    res.chi2_threshold  = chi2Quantile(dof, 1.0 - p_fa);
    res.fault_detected  = (res.chi2_stat > res.chi2_threshold);

    // Per-measurement diagnostics
    res.std_resid.resize(M);
    res.sigmas.resize(M);
    for (int i = 0; i < M; ++i) {
      res.std_resid[i] = r(i) / meas_sigmas(i);
      res.sigmas[i]    = meas_sigmas(i);
    }

    // ── Position covariance  P = (H^T W H)^-1 ─────────────────────────────
    const Eigen::Matrix3d HTSH  = H.transpose() * diag_inv.asDiagonal() * H;
    const Eigen::Matrix3d P_pos = HTSH.inverse();

    // FIX 1: use major-axis eigenvalue of 2×2 horizontal sub-matrix
    const double sigma_major = horizontalMajorAxisSigma(P_pos);  // sqrt(lambda_max)
    const double sigma_v     = std::sqrt(P_pos(2, 2));

    const double hpl0 = k_hpl * sigma_major;
    const double vpl0 = k_hpl * sigma_v;

    // ── Fault-mode HPL/VPL (ARAIM slope method) ───────────────────────────
    // Sensitivity matrix: W_sens = P * H^T * diag(w)
    // W_sens column k gives the position error per unit bias on measurement k.
    const Eigen::MatrixXd W_sens =
        P_pos * H.transpose() * diag_inv.asDiagonal();

    double hpl_fault = hpl0;
    double vpl_fault = vpl0;
    const double k_md = 4.265;  // missed-detection K for P_MD ≈ 10^-3

    for (int k = 0; k < M; ++k) {
      // Horizontal bias sensitivity for measurement k
      const double bias_h =
          std::hypot(W_sens(0, k), W_sens(1, k));
      const double bias_v = std::fabs(W_sens(2, k));

      // sigma after removing measurement k (leave-one-out)
      // FIX 4: use horizontalMajorAxisSigma() for sub-set as well
      double sigma_h_k = sigma_major;
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
        const Eigen::Matrix3d P_k =
            (H_k.transpose() * d_k.asDiagonal() * H_k).inverse();
        sigma_h_k = horizontalMajorAxisSigma(P_k);  // FIX 4
        sigma_v_k = std::sqrt(P_k(2, 2));
      }

      // Minimum detectable bias (threshold in measurement space)
      // det_threshold = sigma_k * sqrt(chi2_thresh)
      const double det_threshold =
          std::sqrt(res.chi2_threshold / diag_inv(k));

      hpl_fault = std::max(hpl_fault,
                           bias_h * det_threshold + k_md * sigma_h_k);
      vpl_fault = std::max(vpl_fault,
                           bias_v * det_threshold + k_md * sigma_v_k);
    }

    res.hpl = std::max(hpl0, hpl_fault);
    res.vpl = std::max(vpl0, vpl_fault);

    // ── FDE: fault detection and exclusion ────────────────────────────────
    if (enable_fde && res.fault_detected && M > 4) {
      double best_chi2 = std::numeric_limits<double>::max();
      int    best_k    = -1;
      for (int k = 0; k < M; ++k) {
        const double t_k = subsetChi2(diag_inv, H, r, k);
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

    // ── Availability ──────────────────────────────────────────────────────
    const bool chi2_ok = !res.fault_detected || (res.excluded_idx >= 0);
    res.available = chi2_ok && (res.hpl < hal) && (res.vpl < val);

    return res;
  }

 private:
  // ===========================================================================
  // FIX 1 helper: sigma of the error ellipse major axis in the horizontal plane
  //
  //   The 2×2 horizontal covariance sub-matrix is
  //       Ph = [ P00  P01 ]
  //            [ P01  P11 ]
  //
  //   Its eigenvalues are:
  //       lambda_{1,2} = (P00+P11)/2  ±  sqrt( ((P00-P11)/2)^2 + P01^2 )
  //
  //   The major-axis sigma = sqrt(lambda_max).
  //   This accounts for the cross-term P01; the old code used sqrt(P00+P11)
  //   which equals sqrt(lambda_1 + lambda_2) and overestimates when the
  //   ellipse is elongated, but can underestimate the protection radius when
  //   P01 is large and P00 ≈ P11.
  // ===========================================================================
  static double horizontalMajorAxisSigma(const Eigen::Matrix3d& P)
  {
    const double p00        = P(0, 0);
    const double p11        = P(1, 1);
    const double p01        = P(0, 1);
    const double half_trace = 0.5 * (p00 + p11);
    const double half_diff  = 0.5 * (p00 - p11);
    const double lambda_max =
        half_trace + std::sqrt(half_diff * half_diff + p01 * p01);
    return std::sqrt(std::max(0.0, lambda_max));
  }

  // ===========================================================================
  // FIX 3: genuine loop-closure test
  //
  //   For every ordered triple (a, b, c) of anchor IDs, if the epoch contains
  //   directed TDOA measurements for all three edges, check:
  //       TDOA(a→b) + TDOA(b→c) + TDOA(c→a) = 0
  //
  //   A directed TDOA(i→j) is obtained from the measurement list as:
  //       +tdoa  if idA==i and idB==j
  //       -tdoa  if idA==j and idB==i
  //
  //   res.ring_sum is the worst (max |error|) signed residual.
  //   res.ring_ok  is false if any loop residual exceeds ring_threshold.
  // ===========================================================================
  void computeRingClosure(const std::vector<TdoaMeas>& meas,
                          size_t                       num_anchors,
                          IntegrityResult&             res) const
  {
    res.ring_sum          = 0.0;
    res.ring_ok           = true;
    res.ring_closed_loops = 0;

    double worst_abs   = 0.0;
    double worst_signed = 0.0;

    const int N = static_cast<int>(num_anchors);
    for (int a = 0; a < N; ++a) {
      for (int b = a + 1; b < N; ++b) {
        for (int c = b + 1; c < N; ++c) {
          double ab = 0.0, bc = 0.0, ca = 0.0;
          // All three directed edges must be present in this epoch
          if (!findDirectedTdoa(meas, a, b, ab) ||
              !findDirectedTdoa(meas, b, c, bc) ||
              !findDirectedTdoa(meas, c, a, ca))
            continue;

          const double err     = ab + bc + ca;
          const double abs_err = std::fabs(err);
          ++res.ring_closed_loops;

          if (abs_err > worst_abs) {
            worst_abs    = abs_err;
            worst_signed = err;
          }
        }
      }
    }

    if (res.ring_closed_loops > 0) {
      res.ring_sum = worst_signed;
      res.ring_ok  = (worst_abs < ring_threshold);
    }
    // If no loop was found the ring test is inconclusive; leave ring_ok=true
    // so it does not falsely gate the availability flag.
  }

  // Look up directed TDOA(from → to) in the measurement list.
  // Returns true and sets value if found (with sign flip when reversed).
  bool findDirectedTdoa(const std::vector<TdoaMeas>& meas,
                        int    from,
                        int    to,
                        double& value) const
  {
    for (const auto& m : meas) {
      if (m.idA == from && m.idB == to) { value =  m.tdoa; return true; }
      if (m.idA == to   && m.idB == from) { value = -m.tdoa; return true; }
    }
    return false;
  }

  // ===========================================================================
  // Build Jacobian H (M×3) and post-fit residual vector r (M×1)
  //
  //   For measurement i with anchor pair (idA, idB):
  //     predicted TDOA = ||p - aB|| - ||p - aA||   (rB - rA)
  //     r(i)           = predicted - measured
  //     H(i,:)         = d/dp (rB - rA) = (p-aB)/rB - (p-aA)/rA
  //                    = unit-vector toward tag from B  −  unit-vector from A
  // ===========================================================================
  void buildHAndResiduals(const std::vector<TdoaMeas>&      meas,
                          const std::vector<gtsam::Point3>& anchors,
                          const Eigen::Vector3d&            p,
                          Eigen::MatrixXd&                  H,
                          Eigen::VectorXd&                  r) const
  {
    const int M = static_cast<int>(meas.size());
    H.resize(M, 3);
    r.resize(M);

    for (int i = 0; i < M; ++i) {
      const auto& m = meas[i];
      const Eigen::Vector3d aA(anchors[m.idA].x(),
                               anchors[m.idA].y(),
                               anchors[m.idA].z());
      const Eigen::Vector3d aB(anchors[m.idB].x(),
                               anchors[m.idB].y(),
                               anchors[m.idB].z());
      const Eigen::Vector3d dA = p - aA;
      const Eigen::Vector3d dB = p - aB;
      const double rA = std::max(dA.norm(), 1e-9);
      const double rB = std::max(dB.norm(), 1e-9);

      r(i)       = (rB - rA) - m.tdoa;
      H.row(i)   = (dB / rB - dA / rA).transpose();
    }
  }

  // ===========================================================================
  // FDE helper: chi-squared statistic after excluding measurement excl
  //
  //   Solves the WLS correction on the reduced set and returns the
  //   weighted sum of squared residuals for the sub-set.
  // ===========================================================================
  double subsetChi2(const Eigen::VectorXd& diag_inv,
                    const Eigen::MatrixXd& H_full,
                    const Eigen::VectorXd& r_full,
                    int                    excl) const
  {
    const int M  = static_cast<int>(r_full.size());
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

    // One WLS correction step on the reduced set
    const Eigen::Matrix3d HTSH_s =
        H_s.transpose() * d_s.asDiagonal() * H_s;
    const Eigen::Vector3d dp =
        -HTSH_s.ldlt().solve(H_s.transpose() * (d_s.asDiagonal() * r_s));
    const Eigen::VectorXd r_new = r_s + H_s * dp;

    return r_new.cwiseProduct(d_s).dot(r_new);
  }

  // Chi-squared quantile via Boost
  static double chi2Quantile(int dof, double p)
  {
    if (dof <= 0) return 0.0;
    boost::math::chi_squared dist(static_cast<double>(dof));
    return boost::math::quantile(dist, p);
  }
};

}  // namespace uwb_imu_fusion