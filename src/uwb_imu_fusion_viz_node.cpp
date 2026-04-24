// Visualization node for UWB anchors and ground truth trajectory.
// Publishes:
//   - visualization_marker_array: anchor positions (spheres)
//   - uwb_imu_path: ground truth trajectory path
// Subscribes:
//   - /pose_data: ground truth pose (geometry_msgs/PoseWithCovarianceStamped)

#include <ros/ros.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <nav_msgs/Path.h>

#include <Eigen/Core>
#include <vector>
#include <deque>
#include <mutex>

namespace uwb_imu_fusion {

class VisualizationNode {
 public:
  VisualizationNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
      : nh_(nh), pnh_(pnh), marker_id_(0) {
    loadAnchors();
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
        "visualization_marker_array", 10);
    path_pub_ = nh_.advertise<nav_msgs::Path>("uwb_imu_path_gt", 100);
    pose_sub_ = nh_.subscribe("/pose_data", 100,
                               &VisualizationNode::poseCallback, this);

    publish_timer_ = nh_.createTimer(ros::Duration(0.1),
                                      &VisualizationNode::publishMarkers, this);

    ROS_INFO("[uwb_imu_fusion_viz] Loaded %zu anchors. Publishing markers.",
             anchors_.size());
  }

 private:
  void loadAnchors() {
    // Load anchors from parameter server (set by launch file under uwb_imu_fusion ns)
    XmlRpc::XmlRpcValue anchors_xml;
    bool got_anchors = nh_.getParam("/uwb_imu_fusion/anchors", anchors_xml);

    if (got_anchors && anchors_xml.getType() == XmlRpc::XmlRpcValue::TypeArray) {
      for (int i = 0; i < anchors_xml.size(); ++i) {
        auto& row = anchors_xml[i];
        if (row.getType() == XmlRpc::XmlRpcValue::TypeArray && row.size() == 3) {
          Eigen::Vector3d p(static_cast<double>(row[0]),
                            static_cast<double>(row[1]),
                            static_cast<double>(row[2]));
          anchors_.push_back(p);
        }
      }
    }

    if (anchors_.empty()) {
      ROS_WARN("[uwb_imu_fusion_viz] No anchors loaded; check /uwb_imu_fusion/anchors");
    }
  }

  void poseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(mutex_);

    // Store ground truth pose for trajectory visualization
    Eigen::Vector3d p(msg->pose.pose.position.x, msg->pose.pose.position.y,
                      msg->pose.pose.position.z);

    // Keep last 500 poses (roughly 5 seconds at 100 Hz)
    trajectory_.push_back({msg->header.stamp, p});
    if (trajectory_.size() > 500) {
      trajectory_.pop_front();
    }
  }

  void publishMarkers(const ros::TimerEvent& event) {
    visualization_msgs::MarkerArray arr;
    marker_id_ = 0;

    // Anchor spheres
    for (size_t i = 0; i < anchors_.size(); ++i) {
      visualization_msgs::Marker m;
      m.header.frame_id = "map";
      m.header.stamp = ros::Time::now();
      m.ns = "anchors";
      m.id = marker_id_++;
      m.type = visualization_msgs::Marker::SPHERE;
      m.action = visualization_msgs::Marker::ADD;

      m.pose.position.x = anchors_[i].x();
      m.pose.position.y = anchors_[i].y();
      m.pose.position.z = anchors_[i].z();
      m.pose.orientation.w = 1.0;

      m.scale.x = m.scale.y = m.scale.z = 0.5;  // 1 meter diameter spheres
  

      
      m.color.r = 1.0f;
      m.color.g = 0.0f;
      m.color.b = 0.0f;
      m.color.a = 0.8f;

      arr.markers.push_back(m);
    }

    marker_pub_.publish(arr);
    publishTrajectory();
  }

  void publishTrajectory() {
    std::lock_guard<std::mutex> lk(mutex_);

    nav_msgs::Path path_msg;
    path_msg.header.frame_id = "map";
    path_msg.header.stamp = ros::Time::now();

    for (const auto& tp : trajectory_) {
      geometry_msgs::PoseStamped ps;
      ps.header.frame_id = "map";
      ps.header.stamp = tp.first;
      ps.pose.position.x = tp.second.x();
      ps.pose.position.y = tp.second.y();
      ps.pose.position.z = tp.second.z();
      ps.pose.orientation.w = 1.0;
      path_msg.poses.push_back(ps);
    }

    path_pub_.publish(path_msg);
  }

  ros::NodeHandle nh_, pnh_;
  ros::Publisher marker_pub_;
  ros::Publisher path_pub_;
  ros::Subscriber pose_sub_;
  ros::Timer publish_timer_;

  std::vector<Eigen::Vector3d> anchors_;
  std::mutex mutex_;
  std::deque<std::pair<ros::Time, Eigen::Vector3d>> trajectory_;
  int marker_id_;
};

}  // namespace uwb_imu_fusion

int main(int argc, char** argv) {
  ros::init(argc, argv, "uwb_imu_fusion_viz_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  uwb_imu_fusion::VisualizationNode node(nh, pnh);
  ros::spin();
  return 0;
}
