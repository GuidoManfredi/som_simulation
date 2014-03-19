#include "tools.h"

#include "opencv2/core/core_c.h" // fore L2_NORM

using namespace std;
using namespace cv;

void transform (cv::Mat points3d, cv::Mat R, cv::Mat t,
                cv::Mat &points3d_out) {
    int n = points3d.rows;
    points3d_out = Mat (n, 3, matrix_type);
    for ( int i = 0; i < n; ++i ) {
        // produit de transposé : (R * pt.t() + t.t()).t() == pt * R.t() + t
        Mat res = points3d.row(i) * R.t() + t;
        res.copyTo(points3d_out.row(i));
    }
}

void getBarycentre (Mat points3d,
                    Mat &t) {
    t = Mat::zeros (1, 3, matrix_type);
    number_type n = points3d.rows;
    for ( int i = 0; i < n; ++i ) {
        t += points3d.row(i);
    }
    t /= n;
}


number_type myRand(number_type min, number_type max) {
  //return min + (max - min) * number_type(rand()) / RAND_MAX;
  return min + max  * number_type(rand()) / RAND_MAX;
}

void inter ( Mat query, Mat train,
             vector<DMatch> &matches ) {
    matches.clear();
    for ( int i = 0; i < query.rows; ++i) {
        for ( int j = 0; j < train.rows; ++j) {
            if (query.at<number_type>(i) == train.at<number_type>(j)) {
                DMatch match;
                match.queryIdx = i;
                match.trainIdx = j;
                matches.push_back (match);
            }
        }
    }
}

void writeMat (char* filename, Mat m) {
    ofstream file (filename);
    if (!file.is_open()) {
        cout << "tools.cpp: writeMat: Couldn't open file." << endl;
        return;
    }

    for ( int i = 0; i < m.rows; ++i ) {
        for ( int j = 0; j < m.cols; ++j ) {
            file << m.at<number_type>(i, j) << ",";
        }
        file << endl;
    }
}

void P2Rt (Mat P, Mat &R, Mat &t) {
    R = Mat::eye (3, 3, matrix_type);
    t = Mat::zeros (1, 3, matrix_type);
    R.at<number_type>(0,0) = P.at<number_type>(0,0);
    R.at<number_type>(0,1) = P.at<number_type>(0,1);
    R.at<number_type>(0,2) = P.at<number_type>(0,2);
    t.at<number_type>(0) = P.at<number_type>(0, 3);

    R.at<number_type>(1,0) = P.at<number_type>(1,0);
    R.at<number_type>(1,1) = P.at<number_type>(1,1);
    R.at<number_type>(1,2) = P.at<number_type>(1,2);
    t.at<number_type>(1) = P.at<number_type>(1,3);

    R.at<number_type>(2,0) = P.at<number_type>(2,0);
    R.at<number_type>(2,1) = P.at<number_type>(2,1);
    R.at<number_type>(2,2) = P.at<number_type>(2,2);
    t.at<number_type>(2) = P.at<number_type>(2,3);
}

Mat Rt2P (Mat R, Mat t) {
    Mat P = Mat::zeros(4, 4, matrix_type);
    P.at<number_type>(0,0) = R.at<number_type>(0,0);
    P.at<number_type>(0,1) = R.at<number_type>(0,1);
    P.at<number_type>(0,2) = R.at<number_type>(0,2);
    P.at<number_type>(0,3) = t.at<number_type>(0);

    P.at<number_type>(1,0) = R.at<number_type>(1,0);
    P.at<number_type>(1,1) = R.at<number_type>(1,1);
    P.at<number_type>(1,2) = R.at<number_type>(1,2);
    P.at<number_type>(1,3) = t.at<number_type>(1);

    P.at<number_type>(2,0) = R.at<number_type>(2,0);
    P.at<number_type>(2,1) = R.at<number_type>(2,1);
    P.at<number_type>(2,2) = R.at<number_type>(2,2);
    P.at<number_type>(2,3) = t.at<number_type>(2);

    P.at<number_type>(3,3) = 1;

    return P;
}

Mat Rt2P34 (Mat R, Mat t) {
    Mat P = Mat::zeros(3, 4, matrix_type);
    P.at<number_type>(0,0) = R.at<number_type>(0,0);
    P.at<number_type>(0,1) = R.at<number_type>(0,1);
    P.at<number_type>(0,2) = R.at<number_type>(0,2);
    P.at<number_type>(0,3) = t.at<number_type>(0);

    P.at<number_type>(1,0) = R.at<number_type>(1,0);
    P.at<number_type>(1,1) = R.at<number_type>(1,1);
    P.at<number_type>(1,2) = R.at<number_type>(1,2);
    P.at<number_type>(1,3) = t.at<number_type>(1);

    P.at<number_type>(2,0) = R.at<number_type>(2,0);
    P.at<number_type>(2,1) = R.at<number_type>(2,1);
    P.at<number_type>(2,2) = R.at<number_type>(2,2);
    P.at<number_type>(2,3) = t.at<number_type>(2);

    return P;
}

Mat KRt2P (Mat K, Mat R, Mat t) {
    cv::Mat KR = K * R;
    cv::Mat Kt = K * t.t();
    cv::Mat P = Rt2P (KR, Kt);
    return P;
}
/*
void relative_error2(Mat Rtrue, Mat ttrue,
                      Mat Rest, Mat test,
                      number_type &rot_err, number_type &transl_err) {
  Mat qtrue, qest;

  mat2quat(Rtrue, qtrue);
  mat2quat(Rest, qest);

  double rot_err1 = norm(qtrue - qest)/norm(qest);
  double rot_err2 = norm(qtrue + qest)/norm(qest);
  rot_err = min(rot_err1, rot_err2);

  transl_err = norm (ttrue - test)/norm(test);
}
*/

void relative_error2(Mat Rtrue, Mat ttrue,
                      Mat Rest, Mat test,
                      number_type &rot_err, number_type &transl_err) {
    rot_err = transl_err = 0;
    Mat rtrue, rest;

    Rodrigues (Rtrue, rtrue);
    Rodrigues (Rest, rest);

//    rot_err = norm(rest, rtrue, NORM_L2|NORM_RELATIVE);
//    transl_err = norm(test, ttrue, NORM_L2|NORM_RELATIVE);
    rot_err = norm(rtrue, rest, NORM_L2|NORM_RELATIVE);
    transl_err = norm(ttrue, test, NORM_L2|NORM_RELATIVE);
    //cout << "Error " << rot_err << " " << transl_err << endl;
}

void mat2quat(const Mat R, Mat &q)
{
    q = Mat::zeros (1, 4, matrix_type);
    double tr = R.at<number_type>(0, 0) + R.at<number_type>(1, 1) + R.at<number_type>(2, 2);
    double n4;

    if (tr > 0.0f) {
        q.at<number_type>(0) = R.at<number_type>(1, 2) - R.at<number_type>(2, 1);
        q.at<number_type>(1) = R.at<number_type>(2, 0) - R.at<number_type>(0, 2);
        q.at<number_type>(2) = R.at<number_type>(0, 1) - R.at<number_type>(1, 0);
        q.at<number_type>(3) = tr + 1.0f;
        n4 = q.at<number_type>(3);
    } else if ( (R.at<number_type>(0, 0) > R.at<number_type>(1, 1)) && (R.at<number_type>(0, 0) > R.at<number_type>(2, 2)) ) {
        q.at<number_type>(0) = 1.0f + R.at<number_type>(0, 0) - R.at<number_type>(1, 1) - R.at<number_type>(2, 2);
        q.at<number_type>(1) = R.at<number_type>(1, 0) + R.at<number_type>(0, 1);
        q.at<number_type>(2) = R.at<number_type>(2, 0) + R.at<number_type>(0, 2);
        q.at<number_type>(3) = R.at<number_type>(1, 2) - R.at<number_type>(2, 1);
        n4 = q.at<number_type>(0);
    } else if (R.at<number_type>(1, 1) > R.at<number_type>(2, 2)) {
        q.at<number_type>(0) = R.at<number_type>(1, 0) + R.at<number_type>(0, 1);
        q.at<number_type>(1) = 1.0f + R.at<number_type>(1, 1) - R.at<number_type>(0, 0) - R.at<number_type>(2, 2);
        q.at<number_type>(2) = R.at<number_type>(2, 1) + R.at<number_type>(1, 2);
        q.at<number_type>(3) = R.at<number_type>(2, 0) - R.at<number_type>(0, 2);
        n4 = q.at<number_type>(1);
    } else {
        q.at<number_type>(0) = R.at<number_type>(2, 0) + R.at<number_type>(0, 2);
        q.at<number_type>(1) = R.at<number_type>(2, 1) + R.at<number_type>(1, 2);
        q.at<number_type>(2) = 1.0f + R.at<number_type>(2, 2) - R.at<number_type>(0, 0) - R.at<number_type>(1, 1);
        q.at<number_type>(3) = R.at<number_type>(0, 1) - R.at<number_type>(1, 0);
        n4 = q.at<number_type>(2);
    }
    double scale = 0.5f / double(sqrt(n4));

    q.at<number_type>(0) *= scale;
    q.at<number_type>(1) *= scale;
    q.at<number_type>(2) *= scale;
    q.at<number_type>(3) *= scale;
}
