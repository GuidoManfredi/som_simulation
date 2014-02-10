#pragma once

#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#define USE_DOUBLES

#ifdef USE_DOUBLES
typedef double number_type;
typedef cv::Point2d point2d_type;
typedef cv::Point3d point3d_type;
static int matrix_type = CV_64F;
#else
typedef float number_type;
typedef cv::Point2f point2d_type;
typedef cv::Point3f point3d_type;
static int matrix_type = CV_32F;
#endif

void transform (cv::Mat points3d, cv::Mat R, cv::Mat t,
                cv::Mat &points3d_out);

void getBarycentre (cv::Mat points3d,
                    cv::Mat &t);

number_type myRand(number_type min, number_type max);

void inter ( cv::Mat m1, cv::Mat m2,
             std::vector<cv::DMatch> &matches );
