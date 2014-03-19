#pragma once

#include <iostream>
#include <fstream>

#include <cstdlib>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

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

void relative_error(cv::Mat R_true, cv::Mat t_true,
                    cv::Mat R_est, cv::Mat t_est,
                    number_type &R_err, number_type &t_err);

void writeMat (char* filename, cv::Mat m);
void P2Rt (cv::Mat P, cv::Mat &R, cv::Mat &t);
cv::Mat Rt2P (cv::Mat R, cv::Mat t);
cv::Mat Rt2P34 (cv::Mat R, cv::Mat t);
cv::Mat KRt2P (cv::Mat K, cv::Mat R, cv::Mat t);

void relative_error2(cv::Mat Rtrue, cv::Mat ttrue,
                      cv::Mat Rest, cv::Mat test,
                      number_type &rot_err, number_type &transl_err);
void mat2quat(const cv::Mat R, cv::Mat &q);
