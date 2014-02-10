#pragma once

#include <opencv2/calib3d/calib3d.hpp>

#include "Simulateur.h"
#include "View.h"

class POM2D
{
  public:
    POM2D();

    void compute (View train, View current);
    void match (View train, View current,
                std::vector<int> &matches);
    void computePoseTrain (cv::Mat train_points2d, cv::Mat current_points3d,
                            View &train, View current);
    void computePoseCurrent (cv::Mat train_points2d, cv::Mat current_points3d,
                             View train, View current);
};

