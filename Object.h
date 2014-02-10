#pragma once

#include "View.h"

class Object
{
  public:
    Object ();

    cv::Mat points3d_; // ground truth
    std::vector<View> views_;
};
