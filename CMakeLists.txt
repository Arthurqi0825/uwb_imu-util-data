cmake_minimum_required(VERSION 3.0.2)
project(uwb_imu_fusion)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

add_compile_options(-Wall -Wextra)

set(Eigen3_DIR /usr/include/eigen3/cmake)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  cf_msgs
)

find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)

# Fix GTSAM variables
set(GTSAM_INCLUDE_DIRS ${GTSAM_INCLUDE_DIR})
set(GTSAM_LIBRARIES gtsam)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs geometry_msgs nav_msgs tf2 tf2_ros tf2_geometry_msgs cf_msgs
  DEPENDS EIGEN3 GTSAM
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(uwb_imu_fusion_node
  src/uwb_imu_fusion_node.cpp
)

add_executable(uwb_imu_fusion_viz_node
  src/uwb_imu_fusion_viz_node.cpp
)

add_dependencies(uwb_imu_fusion_node
  ${catkin_EXPORTED_TARGETS}
)

add_dependencies(uwb_imu_fusion_viz_node
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(uwb_imu_fusion_node
  ${catkin_LIBRARIES}
  gtsam
)

target_link_libraries(uwb_imu_fusion_viz_node
  ${catkin_LIBRARIES}
)
