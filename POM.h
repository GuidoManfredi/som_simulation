#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "tools.h"
#include "Object.h"

class POM
{
  public:
    void process (Object object, View current);
    void match ( const View current, const std::vector<View> views,
                 std::vector<std::vector<cv::DMatch> > &matches );

    void computeTrainPose (View current, View &train, std::vector<cv::DMatch> matches);
    void computeTrainIntrinsic (View current, View &train, std::vector<cv::DMatch> matches);
    void computeCurrentPose (View &current, View train, std::vector<cv::DMatch> matches);

  private:
    void object2viewFrame (cv::Mat points3d, cv::Mat &Rvo, cv::Mat &tvo);
    void mySolvePnP (cv::Mat points3d, cv::Mat points2d, cv::Mat K, cv::Mat &Rov, cv::Mat &tov);
    void P2Rt (cv::Mat P, cv::Mat &R, cv::Mat &t);
    cv::Mat Rt2P (cv::Mat R, cv::Mat t);
    //void mySolvePnPPoints (cv::Mat points3d, cv::Mat points2d, cv::Mat K, cv::Mat &Rov, cv::Mat &tov);
};
