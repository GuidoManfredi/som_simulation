#pragma once

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "tools.h"

class View
{
  public:
    View () {}
    View (cv::Mat K, double noise, cv::Mat Rwv, cv::Mat twv, cv::Mat points3d, cv::Mat indices, bool kinect);
    void points3dFromMatches (std::vector<cv::DMatch> matches,
                             cv::Mat &points3d);
    void worldPoints3dFromMatches (std::vector<cv::DMatch> matches,
                                    cv::Mat &points3d);
    void points2dFromMatches (std::vector<cv::DMatch> matches,
                             cv::Mat &points2d);
    number_type OpenCVreprojectionError (const cv::Mat Rov, const cv::Mat tov, const cv::Mat points3d, const cv::Mat points2d);
    number_type OpenCVreprojectionError (const cv::Mat points3d, const cv::Mat points2d);

    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;

    cv::Mat points2d_; // for simulation, same as keypoints
    cv::Mat points3d_;

    cv::Mat Rov_, tov_;

    cv::Mat Rwv_, twv_;
    cv::Mat Row_, tow_;
    cv::Mat world_points3d_;

    cv::Mat K_;

    number_type noise_;
    bool initialised;
  private:
    void projectWithNoise(cv::Mat R, cv::Mat t,
                          cv::Mat point3d, cv::Mat &point2d);
};

/*

*/
