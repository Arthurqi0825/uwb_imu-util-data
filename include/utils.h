#pragma once

#include <gtsam/geometry/Point3.h>

#include <Eigen/Core>

namespace uwb_imu_fusion
{
namespace utils
{
inline const char *boolText(bool value, const char *true_text, const char *false_text)
{
  return value ? true_text : false_text;
}

inline Eigen::Vector3d toEigen(const gtsam::Point3 &p)
{
  return Eigen::Vector3d(p.x(), p.y(), p.z());
}

inline double distanceIfAvailable(bool reference_available,
                                  const Eigen::Vector3d &point,
                                  const gtsam::Point3 &reference)
{
  if (!reference_available)
    return -1.0;

  return (point - toEigen(reference)).norm();
}

inline double distanceIfAvailable(bool reference_available,
                                  const gtsam::Point3 &point,
                                  const gtsam::Point3 &reference)
{
  return distanceIfAvailable(reference_available, toEigen(point), reference);
}
} // namespace utils
} // namespace uwb_imu_fusion
