// Copyright (C) 2023 Alessandro Fornasier.
// Control of Networked Systems, University of Klagenfurt, Austria.
//
// All rights reserved.
//
// This software is licensed under the terms of the BSD-2-Clause-License with
// no commercial use allowed, the full terms of which are made available
// in the LICENSE file. No license in patents is granted.
//
// You can contact the authors at <alessandro.fornasier@ieee.org>

#include <ros/ros.h>
#include <Eigen/Eigen>

#include "msceqf_ros.hpp"
#include "utils/logger.hpp"

MSCEqFRos::MSCEqFRos(const ros::NodeHandle &nh,
                     const std::string &msceqf_config_filepath,
                     const std::string &imu_topic,
                     const std::string &cam_topic,
                     const std::string &features_topic,
                     const std::string &pose_topic,
                     const std::string &path_topic,
                     const std::string &image_topic,
                     const std::string &extrinsics_topic,
                     const std::string &intrinsics_topic,
                     const std::string &origin_topic,
                     const bool &record,
                     const std::string &bagfile)
    : nh_(nh), sys_(msceqf_config_filepath)
{
  sub_cam_ = nh_.subscribe(cam_topic, 10, &MSCEqFRos::callback_image, this);
  sub_imu_ = nh_.subscribe(imu_topic, 1000, &MSCEqFRos::callback_imu, this);
  sub_features_ = nh_.subscribe(features_topic, 10, &MSCEqFRos::callback_features, this);

  utils::Logger::info("Subscribing: " + std::string(sub_cam_.getTopic().c_str()));
  utils::Logger::info("Subscribing: " + std::string(sub_imu_.getTopic().c_str()));
  utils::Logger::info("Subscribing: " + std::string(sub_features_.getTopic().c_str()));

  pub_pose_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>(pose_topic, 1);
  pub_path_ = nh_.advertise<nav_msgs::Path>(path_topic, 1);
  pub_image_ = nh_.advertise<sensor_msgs::Image>(image_topic, 1);
  pub_extrinsics_ = nh_.advertise<geometry_msgs::PoseStamped>(extrinsics_topic, 1);
  pub_intrinsics_ = nh_.advertise<sensor_msgs::CameraInfo>(intrinsics_topic, 1);
  pub_origin_ = nh_.advertise<geometry_msgs::PoseStamped>(origin_topic, 1);

  utils::Logger::info("Publishing: " + std::string(pub_pose_.getTopic().c_str()));
  utils::Logger::info("Publishing: " + std::string(pub_path_.getTopic().c_str()));
  utils::Logger::info("Publishing: " + std::string(pub_image_.getTopic().c_str()));
  utils::Logger::info("Publishing: " + std::string(pub_extrinsics_.getTopic().c_str()));
  utils::Logger::info("Publishing: " + std::string(pub_intrinsics_.getTopic().c_str()));
  utils::Logger::info("Publishing: " + std::string(pub_origin_.getTopic().c_str()));

  record_ = record;
  if (record_)
  {
    bag_.open(bagfile, rosbag::bagmode::Write);
  }
}

void MSCEqFRos::callback_image(const sensor_msgs::Image::ConstPtr &msg)
{
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
  }
  catch (cv_bridge::Exception &e)
  {
    utils::Logger::err("cv_bridge exception: " + std::string(e.what()));
    return;
  }

  msceqf::Camera cam;

  cam.timestamp_ = cv_ptr->header.stamp.toSec();
  cam.image_ = cv_ptr->image.clone();

  {
    std::lock_guard<std::mutex> lock(mutex_);
    cams_.push_back(cam);
    std::sort(cams_.begin(), cams_.end());
  }
}

void MSCEqFRos::callback_features(const std_msgs::Float64MultiArray::ConstPtr &msg)
{
  if (msg->data.empty())
  {
      utils::Logger::warn("Received empty features message");
      return;
  }

  const size_t values_per_feature = 11;
  const size_t num_features = msg->data.size() / values_per_feature;

  if (msg->data.size() % values_per_feature != 0)
  {
      utils::Logger::err("Invalid feature data size: " + std::to_string(msg->data.size()));
      return;
  }

  if (num_features == 0)
  {
    utils::Logger::warn("No features in message");
    return;
  }

  // IMPORTANT: Extract timestamp from FIRST feature
  // All features in the message MUST have the same timestamp
  double timestamp_curr = msg->data[4];  // t_current from first feature

  // Verify all features have the same timestamp (optional sanity check)
  bool timestamp_mismatch = false;
  for (size_t i = 1; i < num_features; ++i)
  {
    double t_check = msg->data[i * values_per_feature + 4];
    if (std::abs(t_check - timestamp_curr) > 1e-6)
    {
      utils::Logger::warn("Feature " + std::to_string(i) + 
                         " has different timestamp: " + std::to_string(t_check) + 
                         " vs " + std::to_string(timestamp_curr));
      timestamp_mismatch = true;
      break;
    }
  }

  if (timestamp_mismatch)
  {
    utils::Logger::err("Features have inconsistent timestamps - skipping message");
    return;
  }

  utils::Logger::info("Received " + std::to_string(num_features) + 
                      " features at t=" + std::to_string(timestamp_curr));

  // Create TriangulatedFeatures
  msceqf::TriangulatedFeatures triangulated_features;
  triangulated_features.timestamp_ = timestamp_curr;  // Single timestamp for all features
  triangulated_features.features_.distorted_uvs_.reserve(num_features);
  triangulated_features.features_.uvs_.reserve(num_features);
  triangulated_features.features_.normalized_uvs_.reserve(num_features);
  triangulated_features.features_.ids_.reserve(num_features);
  triangulated_features.points_.reserve(num_features);

  // Get camera intrinsics and resolution from system options
  const msceqf::Vector4 cam_intrinsics_vec = sys_.stateOptions().initial_camera_intrinsics_.k();
  const auto& resolution = sys_.options().track_manager_options_.tracker_options_.cam_options_.resolution_;
  
double fx = cam_intrinsics_vec(0);
double fy = cam_intrinsics_vec(1);
double cx = cam_intrinsics_vec(2);
double cy = cam_intrinsics_vec(3);
  
  int img_width = static_cast<int>(resolution(0));   // Width
  int img_height = static_cast<int>(resolution(1));  // Height

  // Log camera parameters for debugging
  utils::Logger::info("Camera intrinsics: fx=" + std::to_string(fx) + 
                      ", fy=" + std::to_string(fy) + 
                      ", cx=" + std::to_string(cx) + 
                      ", cy=" + std::to_string(cy));
  utils::Logger::info("Image resolution: " + std::to_string(img_width) + "x" + std::to_string(img_height));


  // Parse each feature
  for (size_t i = 0; i < num_features; ++i)
  {
      size_t offset = i * values_per_feature;
      
      // Extract data
      // double cam_number = msg->data[offset + 0];     // Not used
      // double t_prev = msg->data[offset + 1];         // Not used for current implementation
      // double x_prev = msg->data[offset + 2];         // Not used
      // double y_prev = msg->data[offset + 3];         // Not used
      double t_current = msg->data[offset + 4];         // t_current
      float u_pixel = static_cast<float>(msg->data[offset + 5]);  // x_current
      float v_pixel = static_cast<float>(msg->data[offset + 6]);  // y_current
      double x_3d = msg->data[offset + 7];  // X_world
      double y_3d = msg->data[offset + 8];  // Y_world
      double z_3d = msg->data[offset + 9];  // Z_world
      uint32_t feature_id = static_cast<uint32_t>(msg->data[offset + 10]);

      // Store pixel coordinates (assume already undistorted by MATLAB)
      cv::Point2f pixel_coord(u_pixel, v_pixel);
      triangulated_features.features_.distorted_uvs_.push_back(pixel_coord);  // Already undistorted
      triangulated_features.features_.uvs_.push_back(pixel_coord);  // No undistortion needed
      triangulated_features.features_.ids_.push_back(feature_id);
      
    // Manually normalize using camera intrinsics
    // Normalized coordinates: [(u - cx) / fx, (v - cy) / fy]
    cv::Point2f normalized_coord;
    normalized_coord.x = (u_pixel - cx) / fx;
    normalized_coord.y = (v_pixel - cy) / fy;
    triangulated_features.features_.normalized_uvs_.push_back(normalized_coord);

    // Store 3D point (not used by filter but kept for completeness)
    msceqf::Vector3 point_3d;
    point_3d << x_3d, y_3d, z_3d;
    triangulated_features.points_.push_back(point_3d);

      if (i == 0)  // Just log first feature
      {
      utils::Logger::info("Feature 0: ID=" + std::to_string(feature_id) + 
                          ", pixel=(" + std::to_string(u_pixel) + "," + std::to_string(v_pixel) + ")" +
                          ", normalized=(" + std::to_string(normalized_coord.x) + "," + 
                          std::to_string(normalized_coord.y) + ")" +
                          ", 3D=(" + std::to_string(x_3d) + "," + std::to_string(y_3d) + 
                          "," + std::to_string(z_3d) + ")");
      }
  }

  // Validate features are within image bounds (optional sanity check)
  int valid_count = 0;
  for (const auto& uv : triangulated_features.features_.uvs_)
  {
    if (uv.x >= 0 && uv.x < img_width && uv.y >= 0 && uv.y < img_height)
    {
      valid_count++;
    }
    else
    {
      utils::Logger::warn("Feature outside image bounds: (" + 
                         std::to_string(uv.x) + ", " + std::to_string(uv.y) + ")");
    }
  }
  
  utils::Logger::info("Valid features: " + std::to_string(valid_count) + 
                      "/" + std::to_string(num_features));

  // After creating triangulated_features, log its size
  utils::Logger::info("Created TriangulatedFeatures with " + 
                     std::to_string(triangulated_features.features_.ids_.size()) + 
                     " features and " + 
                     std::to_string(triangulated_features.points_.size()) + " points");
  
  // Buffer for processing
  {
      std::lock_guard<std::mutex> lock(mutex_);
      triangulated_features_.push_back(triangulated_features);
      std::sort(triangulated_features_.begin(), triangulated_features_.end());
  }
}

void MSCEqFRos::callback_imu(const sensor_msgs::Imu::ConstPtr &msg)
{
  msceqf::Imu imu;

  auto timestamp = msg->header.stamp.toSec();

  imu.timestamp_ = timestamp;
  imu.ang_ << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
  imu.acc_ << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;

  sys_.processMeasurement(imu);

  if (!processing_)
  {
    processing_ = true;
    std::thread th([&, timestamp]
      {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        // Process features measurements
        while (!triangulated_features_.empty() && triangulated_features_.front().timestamp_ < timestamp)
        {
          // Get reference to features before processing
          auto& curr_features = triangulated_features_.front();
          double features_timestamp = curr_features.timestamp_;
          utils::Logger::info("Processing features at t=" + std::to_string(features_timestamp));

          // ADD THIS DEBUG OUTPUT:
          utils::Logger::info("Number of features in message: " + 
                             std::to_string(curr_features.features_.ids_.size()));
          
          // Process the features
          sys_.processMeasurement(curr_features);
          // Publish pose after processing features
          publishFromFeatures(features_timestamp);
          // Remove processed features
          triangulated_features_.pop_front();
        }
        // Process images measurements
        while (!cams_.empty() && cams_.front().timestamp_ < timestamp)
        {
          sys_.processMeasurement(cams_.front());
          publish(cams_.front());
          cams_.pop_front();
        }
      }
      processing_ = false; });
    th.detach();
  }
}

void MSCEqFRos::publish(const msceqf::Camera &cam)
{
  if (!sys_.isInit())
  {
    return;
  }

  auto est = sys_.stateEstimate();
  auto origin = sys_.stateOrigin();

  pose_.header.stamp.fromSec(cam.timestamp_);
  pose_.header.frame_id = "global";
  pose_.header.seq = seq_;

  pose_.pose.pose.orientation.x = est.T().q().x();
  pose_.pose.pose.orientation.y = est.T().q().y();
  pose_.pose.pose.orientation.z = est.T().q().z();
  pose_.pose.pose.orientation.w = est.T().q().w();

  pose_.pose.pose.position.x = est.T().p().x();
  pose_.pose.pose.position.y = est.T().p().y();
  pose_.pose.pose.position.z = est.T().p().z();

  // The covairance is published in the ROS convention order, postion first then orientation
  Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();
  cov.block<3, 3>(0, 0) = sys_.coreCovariance().block<3, 3>(6, 6);
  cov.block<3, 3>(0, 3) = sys_.coreCovariance().block<3, 3>(6, 0);
  cov.block<3, 3>(3, 0) = sys_.coreCovariance().block<3, 3>(0, 6);
  cov.block<3, 3>(3, 3) = sys_.coreCovariance().block<3, 3>(0, 0);
  for (int r = 0; r < 6; r++)
  {
    for (int c = 0; c < 6; c++)
    {
      pose_.pose.covariance[6 * r + c] = cov(r, c);
    }
  }

  pub_pose_.publish(pose_);

  if (record_)
  {
    bag_.write(pub_pose_.getTopic().c_str(), pose_.header.stamp, pose_);
  }

  if (pub_origin_.getNumSubscribers() != 0)
  {
    origin_.header.stamp.fromSec(cam.timestamp_);
    origin_.header.frame_id = "global";
    origin_.header.seq = seq_;

    origin_.pose.orientation.x = origin.T().q().x();
    origin_.pose.orientation.y = origin.T().q().y();
    origin_.pose.orientation.z = origin.T().q().z();
    origin_.pose.orientation.w = origin.T().q().w();

    origin_.pose.position.x = origin.T().p().x();
    origin_.pose.position.y = origin.T().p().y();
    origin_.pose.position.z = origin.T().p().z();

    pub_origin_.publish(origin_);

    if (record_)
    {
      bag_.write(pub_origin_.getTopic().c_str(), origin_.header.stamp, origin_);
    }
  }

  if (pub_path_.getNumSubscribers() != 0)
  {
    geometry_msgs::PoseStamped pose;
    pose.header = pose_.header;
    pose.pose = pose_.pose.pose;

    path_.header.stamp = ros::Time::now();
    path_.header.seq = seq_;
    path_.header.frame_id = "global";
    path_.poses.push_back(pose);

    pub_path_.publish(path_);
  }

  if (pub_image_.getNumSubscribers() != 0)
  {
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = "cam0";
    sensor_msgs::ImagePtr img = cv_bridge::CvImage(header, "bgr8", sys_.imageWithTracks(cam)).toImageMsg();
    pub_image_.publish(img);
  }

  if (pub_extrinsics_.getNumSubscribers() != 0)
  {
    extrinsics_.header.stamp.fromSec(cam.timestamp_);
    extrinsics_.header.frame_id = "imu";
    extrinsics_.header.seq = seq_;
    extrinsics_.pose.orientation.x = est.S().q().x();
    extrinsics_.pose.orientation.y = est.S().q().y();
    extrinsics_.pose.orientation.z = est.S().q().z();
    extrinsics_.pose.orientation.w = est.S().q().w();
    extrinsics_.pose.position.x = est.S().x().x();
    extrinsics_.pose.position.y = est.S().x().y();
    extrinsics_.pose.position.z = est.S().x().z();

    pub_extrinsics_.publish(extrinsics_);

    if (record_)
    {
      bag_.write(pub_extrinsics_.getTopic().c_str(), extrinsics_.header.stamp, extrinsics_);
    }
  }

  if (pub_intrinsics_.getNumSubscribers() != 0)
  {
    auto intr = est.k();

    intrinsics_.header.stamp.fromSec(cam.timestamp_);
    intrinsics_.header.frame_id = "cam";
    intrinsics_.header.seq = seq_;
    intrinsics_.height = cam.image_.rows;
    intrinsics_.width = cam.image_.cols;
    intrinsics_.distortion_model = "";
    intrinsics_.D = {0.0, 0.0, 0.0, 0.0, 0.0};
    intrinsics_.K = {intr(0), 0.0, intr(2), 0.0, intr(1), intr(3), 0.0, 0.0, 1.0};
    intrinsics_.R = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    intrinsics_.P = {intr(0), 0.0, intr(2), 0.0, 0.0, intr(1), intr(3), 0.0, 0.0, 0.0, 1.0, 0.0};

    pub_intrinsics_.publish(intrinsics_);

    if (record_)
    {
      bag_.write(pub_intrinsics_.getTopic().c_str(), intrinsics_.header.stamp, intrinsics_);
    }
  }

  ++seq_;
}

void MSCEqFRos::publishFromFeatures(const double& timestamp)
{
  if (!sys_.isInit())
  {
    utils::Logger::debug("System not initialized, skipping publish");
    return;
  }

  auto est = sys_.stateEstimate();
  auto origin = sys_.stateOrigin();

  pose_.header.stamp.fromSec(timestamp);
  pose_.header.frame_id = "global";
  pose_.header.seq = seq_;

  pose_.pose.pose.orientation.x = est.T().q().x();
  pose_.pose.pose.orientation.y = est.T().q().y();
  pose_.pose.pose.orientation.z = est.T().q().z();
  pose_.pose.pose.orientation.w = est.T().q().w();

  pose_.pose.pose.position.x = est.T().p().x();
  pose_.pose.pose.position.y = est.T().p().y();
  pose_.pose.pose.position.z = est.T().p().z();

  // Covariance
  Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();
  cov.block<3, 3>(0, 0) = sys_.coreCovariance().block<3, 3>(6, 6);
  cov.block<3, 3>(0, 3) = sys_.coreCovariance().block<3, 3>(6, 0);
  cov.block<3, 3>(3, 0) = sys_.coreCovariance().block<3, 3>(0, 6);
  cov.block<3, 3>(3, 3) = sys_.coreCovariance().block<3, 3>(0, 0);
  for (int r = 0; r < 6; r++)
  {
    for (int c = 0; c < 6; c++)
    {
      pose_.pose.covariance[6 * r + c] = cov(r, c);
    }
  }

  pub_pose_.publish(pose_);

  if (record_)
  {
    bag_.write(pub_pose_.getTopic().c_str(), pose_.header.stamp, pose_);
  }

  // Optionally publish path
  if (pub_path_.getNumSubscribers() != 0)
  {
    geometry_msgs::PoseStamped pose;
    pose.header = pose_.header;
    pose.pose = pose_.pose.pose;

    path_.header.stamp.fromSec(timestamp);
    path_.header.seq = seq_;
    path_.header.frame_id = "global";
    path_.poses.push_back(pose);

    pub_path_.publish(path_);
  }

  ++seq_;
}