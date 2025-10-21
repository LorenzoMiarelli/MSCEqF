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

#include "vision/camera.hpp"

#include <opencv2/core/eigen.hpp>

namespace msceqf
{
PinholeCamera::PinholeCamera(const VectorX& distortion_coefficients,
                             const Vector4 instrinsics,
                             const uint& width,
                             const uint& height)
    : distortion_coefficients_(distortion_coefficients), intrinsics_(instrinsics), width_(width), height_(height)
{
}

void PinholeCamera::setIntrinsics(const Vector4& intrinsics) { intrinsics_ = intrinsics; }

const Vector4& PinholeCamera::intrinsics() const { return intrinsics_; }

const VectorX& PinholeCamera::distortionCoefficients() const { return distortion_coefficients_; }

void PinholeCamera::normalize(std::vector<Eigen::Vector2f>& uv)
{
  for (auto& coords : uv)
  {
    coords(0) = (coords(0) - intrinsics_(2)) / intrinsics_(0);
    coords(1) = (coords(1) - intrinsics_(3)) / intrinsics_(1);
  }
}

void PinholeCamera::normalize(std::vector<cv::Point2f>& uv)
{
  for (auto& coords : uv)
  {
    coords.x = (coords.x - intrinsics_(2)) / intrinsics_(0);
    coords.y = (coords.y - intrinsics_(3)) / intrinsics_(1);
  }
}

void PinholeCamera::normalize(Eigen::Vector2f& uv)
{
  uv(0) = (uv(0) - intrinsics_(2)) / intrinsics_(0);
  uv(1) = (uv(1) - intrinsics_(3)) / intrinsics_(1);
}

void PinholeCamera::normalize(cv::Point2f& uv)
{
  uv.x = (uv.x - intrinsics_(2)) / intrinsics_(0);
  uv.y = (uv.y - intrinsics_(3)) / intrinsics_(1);
}

void PinholeCamera::denormalize(std::vector<Eigen::Vector2f>& uv)
{
  for (auto& coords : uv)
  {
    coords(0) = coords(0) * intrinsics_(0) + intrinsics_(2);
    coords(1) = coords(1) * intrinsics_(1) + intrinsics_(3);
  }
}

void PinholeCamera::denormalize(std::vector<cv::Point2f>& uv)
{
  for (auto& coords : uv)
  {
    coords.x = coords.x * intrinsics_(0) + intrinsics_(2);
    coords.y = coords.y * intrinsics_(1) + intrinsics_(3);
  }
}

void PinholeCamera::denormalize(Eigen::Vector2f& uv)
{
  uv(0) = uv(0) * intrinsics_(0) + intrinsics_(2);
  uv(1) = uv(1) * intrinsics_(1) + intrinsics_(3);
}

void PinholeCamera::denormalize(cv::Point2f& uv)
{
  uv.x = uv.x * intrinsics_(0) + intrinsics_(2);
  uv.y = uv.y * intrinsics_(1) + intrinsics_(3);
}

void PinholeCamera::undistort(std::vector<Eigen::Vector2f>& uv, const bool& normalize)
{
  std::vector<cv::Point2f> uv_cv;
  uv_cv.reserve(uv.size());

  for (const auto& coords : uv)
  {
    uv_cv.emplace_back(coords(0), coords(1));
  }

  undistort(uv_cv, normalize);

  for (size_t i = 0; i < uv_cv.size(); ++i)
  {
    uv[i](0) = uv_cv[i].x;
    uv[i](1) = uv_cv[i].y;
  }
}

RadtanCamera::RadtanCamera(const CameraOptions& opts, const Vector4& intrinsics)
    : PinholeCamera(opts.distortion_coefficients_, intrinsics, opts.resolution_(0), opts.resolution_(1))
{
}

void RadtanCamera::undistort(std::vector<cv::Point2f>& uv_cv, const bool& normalize)
{
  cv::Vec<fp, 4> dist_cv;
  cv::Matx<fp, 3, 3> K_cv;

  cv::eigen2cv(distortion_coefficients_, dist_cv);

  K_cv(0, 0) = intrinsics_(0);
  K_cv(1, 1) = intrinsics_(1);
  K_cv(0, 2) = intrinsics_(2);
  K_cv(1, 2) = intrinsics_(3);
  K_cv(2, 2) = 1.0f;

  if (normalize)
  {
    cv::undistortPoints(uv_cv, uv_cv, K_cv, dist_cv);
  }
  else
  {
    cv::undistortPoints(uv_cv, uv_cv, K_cv, dist_cv, cv::noArray(), K_cv);
  }
}

void RadtanCamera::undistortImage(const cv::Mat& image, cv::Mat& image_undistorted)
{
  cv::Vec<fp, 4> dist_cv;
  cv::Matx<fp, 3, 3> K_cv;

  cv::eigen2cv(distortion_coefficients_, dist_cv);

  K_cv(0, 0) = intrinsics_(0);
  K_cv(1, 1) = intrinsics_(1);
  K_cv(0, 2) = intrinsics_(2);
  K_cv(1, 2) = intrinsics_(3);
  K_cv(2, 2) = 1.0f;

  cv::undistort(image, image_undistorted, K_cv, dist_cv);
}

EquidistantCamera::EquidistantCamera(const CameraOptions& opts, const Vector4& intrinsics)
    : PinholeCamera(opts.distortion_coefficients_, intrinsics, opts.resolution_(0), opts.resolution_(1))
{
}

void EquidistantCamera::undistort(std::vector<cv::Point2f>& uv_cv, const bool& normalize)
{
  cv::Vec<fp, 4> dist_cv;
  cv::Matx<fp, 3, 3> K_cv;

  cv::eigen2cv(distortion_coefficients_, dist_cv);

  K_cv(0, 0) = intrinsics_(0);
  K_cv(1, 1) = intrinsics_(1);
  K_cv(0, 2) = intrinsics_(2);
  K_cv(1, 2) = intrinsics_(3);
  K_cv(2, 2) = 1.0f;

  if (normalize)
  {
    cv::fisheye::undistortPoints(uv_cv, uv_cv, K_cv, dist_cv);
  }
  else
  {
    cv::fisheye::undistortPoints(uv_cv, uv_cv, K_cv, dist_cv, cv::noArray(), K_cv);
  }
}

void EquidistantCamera::undistortImage(const cv::Mat& image, cv::Mat& image_undistorted)
{
  cv::Vec<fp, 4> dist_cv;
  cv::Matx<fp, 3, 3> K_cv;

  cv::eigen2cv(distortion_coefficients_, dist_cv);

  K_cv(0, 0) = intrinsics_(0);
  K_cv(1, 1) = intrinsics_(1);
  K_cv(0, 2) = intrinsics_(2);
  K_cv(1, 2) = intrinsics_(3);
  K_cv(2, 2) = 1.0f;

  // cv::fisheye::undistortImage(image, image_undistorted, K_cv, dist_cv);
  cv::Mat map1, map2;
  cv::fisheye::initUndistortRectifyMap(K_cv, dist_cv, cv::Matx33d::eye(), K_cv, cv::Size(image.cols, image.rows),
                                       CV_32FC1, map1, map2);
  cv::remap(image, image_undistorted, map1, map2, cv::INTER_LINEAR);
}

FOVCamera::FOVCamera(const CameraOptions& opts, const Vector4& intrinsics)
    : PinholeCamera(opts.distortion_coefficients_, intrinsics, opts.resolution_(0), opts.resolution_(1))
{
  // Validate that we have the s parameter
  if (distortion_coefficients_.size() < 1)
  {
    throw std::runtime_error("FOV distortion model requires at least 1 coefficient (s parameter)");
  }
}

void FOVCamera::undistort(std::vector<cv::Point2f>& uv_cv, const bool& normalize)
{
  // Get FOV parameter
  fp s = distortion_coefficients_(0);
  
  // Create camera matrix
  cv::Matx<fp, 3, 3> K_cv;
  K_cv(0, 0) = intrinsics_(0);
  K_cv(1, 1) = intrinsics_(1);
  K_cv(0, 2) = intrinsics_(2);
  K_cv(1, 2) = intrinsics_(3);
  K_cv(2, 2) = 1.0f;

  // Undistort each point
  std::vector<cv::Point2f> uv_normalized;
  uv_normalized.reserve(uv_cv.size());

  for (const auto& uv : uv_cv)
  {
    // Convert to normalized distorted coordinates
    fp x_d = (uv.x - intrinsics_(2)) / intrinsics_(0);
    fp y_d = (uv.y - intrinsics_(3)) / intrinsics_(1);
    
    // Compute distorted radius
    fp r_d = std::sqrt(x_d * x_d + y_d * y_d);
    
    // Undistorted radius
    fp r_u;
    if (std::abs(s) > 1e-8)
    {
      fp tan_half_s = std::tan(s / 2.0);
      r_u = (std::abs(tan_half_s) > 1e-8) ? std::tan(r_d * s) / (2.0 * tan_half_s) : r_d;
    }
    else
    {
      r_u = r_d;
    }
    
    // Compute undistorted normalized coordinates
    fp scale = (r_d > 1e-8) ? (r_u / r_d) : 1.0;
    fp x_u = scale * x_d;
    fp y_u = scale * y_d;
    
    uv_normalized.emplace_back(static_cast<float>(x_u), static_cast<float>(y_u));
  }

  if (normalize)
  {
    uv_cv = std::move(uv_normalized);
  }
  else
  {
    // Convert back to pixel coordinates
    for (size_t i = 0; i < uv_normalized.size(); ++i)
    {
      uv_cv[i].x = uv_normalized[i].x * intrinsics_(0) + intrinsics_(2);
      uv_cv[i].y = uv_normalized[i].y * intrinsics_(1) + intrinsics_(3);
    }
  }
}

void FOVCamera::undistortImage(const cv::Mat& image, cv::Mat& image_undistorted)
{
  // Get FOV parameter
  fp s = distortion_coefficients_(0);
  
  // Create remap matrices
  cv::Mat map_x(image.size(), CV_32FC1);
  cv::Mat map_y(image.size(), CV_32FC1);
  
  // Build lookup table for efficiency
  for (int v = 0; v < image.rows; ++v)
  {
    float* map_x_row = map_x.ptr<float>(v);
    float* map_y_row = map_y.ptr<float>(v);
    
    for (int u = 0; u < image.cols; ++u)
    {
      // Normalized undistorted coordinates
      fp x_u = (u - intrinsics_(2)) / intrinsics_(0);
      fp y_u = (v - intrinsics_(3)) / intrinsics_(1);
      
      // Compute undistorted radius
      fp r_u = std::sqrt(x_u * x_u + y_u * y_u);
      
      // Apply distortion (inverse operation)
      fp r_d;
      if (std::abs(s) > 1e-8)
      {
        r_d = std::atan(2.0 * r_u * std::tan(s / 2.0)) / s;
      }
      else
      {
        r_d = r_u;
      }
      
      // Compute distorted normalized coordinates
      fp scale = (r_u > 1e-8) ? (r_d / r_u) : 1.0;
      fp x_d = scale * x_u;
      fp y_d = scale * y_u;
      
      // Convert to pixel coordinates
      map_x_row[u] = static_cast<float>(x_d * intrinsics_(0) + intrinsics_(2));
      map_y_row[u] = static_cast<float>(y_d * intrinsics_(1) + intrinsics_(3));
    }
  }
  
  // Apply remapping
  cv::remap(image, image_undistorted, map_x, map_y, cv::INTER_LINEAR);
}

}  // namespace msceqf