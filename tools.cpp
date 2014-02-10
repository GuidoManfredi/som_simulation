#include "tools.h"

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
  return min + (max - min) * number_type(rand()) / RAND_MAX;
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
