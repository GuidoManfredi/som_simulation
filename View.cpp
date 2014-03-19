#include "View.h"

#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;
// t = 1 row and 3 columns
View::View (Mat K, double noise, Mat Rwv, Mat twv, Mat points3d, Mat indices, bool kinect):
K_(K), descriptors_(indices) {
    Rov_ = Mat::eye (3, 3, matrix_type);
    tov_ = Mat::zeros (1, 3, matrix_type);
    points2d_ = Mat(points3d.rows, 2, matrix_type);
    noise_ = noise;

    cout << "KikouView" << endl;
    cout << points3d << endl;
    for ( size_t i = 0; i < points3d.rows; ++i ) {
        Mat point3d = points3d.row(i);
        Mat point2d (1, 2, matrix_type);
        // project 3D points to camera frame then image plan xi = K * (Rwv * Xw + twv)
        projectWithNoise (Rwv, twv, point3d, point2d);
        point2d.copyTo(points2d_.row(i));
    }

    if ( kinect ) {
        points3d_ = Mat(points3d.rows, 3, matrix_type);
        // convert points from world to kinect frame : Xc = P' * Xw
        transform (points3d, Rwv, twv, points3d_);
    }
    // Ground truth
    Rwv.copyTo (Rwv_);
    twv.copyTo (twv_);
    points3d.copyTo (world_points3d_);

    initialised = false;
}

void View::points3dFromMatches (vector<DMatch> matches,
                                Mat &points3d) {
    points3d = Mat (matches.size(), 3, matrix_type);
    for ( size_t i = 0; i < matches.size(); ++i ) {
        int idx = matches[i].queryIdx;
        points3d.at<point3d_type>(i) = points3d_.at<point3d_type>(idx);
    }
}

void View::worldPoints3dFromMatches (vector<DMatch> matches,
                                     Mat &points3d) {
    points3d = Mat (matches.size(), 3, matrix_type);
    for ( size_t i = 0; i < matches.size(); ++i ) {
        int idx = matches[i].queryIdx;
        points3d.at<point3d_type>(i) = world_points3d_.at<point3d_type>(idx);
    }
}

void View::points2dFromMatches (vector<DMatch> matches,
                                Mat &points2d) {
    points2d = Mat (matches.size(), 2, matrix_type);
    for ( size_t i = 0; i < matches.size(); ++i ) {
        int idx = matches[i].trainIdx;
        points2d.at<point2d_type>(i) = points2d_.at<point2d_type>(idx);
    }
}
////////////////////////////////////////////////////////////////////////////////
//  PRIVATE METHODES
////////////////////////////////////////////////////////////////////////////////
void View::projectWithNoise (Mat Rwv, Mat twv,
                             Mat points3d, Mat &points2d) {
    Mat rvec;
    Rodrigues(Rwv, rvec);

    Mat noise_mat = (Mat_<number_type>(1, 2, matrix_type) << myRand(-noise_, +noise_),
                                                             myRand(-noise_, +noise_));
    projectPoints (points3d, rvec, twv, K_, vector<number_type>(), points2d);
    points2d = points2d.reshape (1, 1); // 1 channel, 1 row
    points2d += noise_mat;
}

number_type View::OpenCVreprojectionError (const Mat points3d, const Mat points2d) {
    return OpenCVreprojectionError (Rov_, tov_, points3d, points2d);
}

number_type View::OpenCVreprojectionError (const Mat Rov, const Mat tov, const Mat points3d, const Mat points2d) {
    number_type total_error = 0;
    Mat rvec;
    Rodrigues (Rov, rvec);
    Mat tvec = tov;

    int n = points3d.rows;
    for ( int i = 0; i < n; ++i ) {
        Mat projected_points2d (1, 2, matrix_type);
        projectPoints (points3d.row(i), rvec, tvec, K_, vector<number_type>(), projected_points2d);
        projected_points2d = projected_points2d.reshape (1, 1);
        total_error += norm(projected_points2d, points2d.row(i), NORM_L2);
    }

    return total_error / n;
}
