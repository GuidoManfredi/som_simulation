#pragma once

#include <iostream>

#include "tools.h"
#include "Object.h"

class Simulateur
{
  public:
    Simulateur (cv::Mat K);

    Object generateObject (int N, int K);
    View generateView (cv::Mat points3d, int start, int end, bool kinect, int pose);

  private:
    void specific_pose0 (cv::Mat &R, cv::Mat &t);
    void specific_pose45 (cv::Mat &R, cv::Mat &t);
    void specific_pose315 (cv::Mat &R, cv::Mat &t);
    void random_pose(cv::Mat &R, cv::Mat &t);
    void random_point(number_type & Xw, number_type & Yw, number_type & Zw);

    cv::Mat points3d_;
    std::vector<View> train_views_;
    std::vector<View> first_time_views_;
    std::vector<View> test_loca_views_;
    std::vector<View> test_merge_views_;

    cv::Mat K_;
};



