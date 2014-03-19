#include "Simulateur.h"

using namespace std;
using namespace cv;

Simulateur::Simulateur (Mat K): K_(K) {

}

Object Simulateur::generateObject (int N, int K, double noise) {
    Object object;
    object.points3d_ = Mat::zeros(N, 3, matrix_type);
    for ( int i = 0; i < N; ++i ) {
        random_point (object.points3d_.at<number_type>(i, 0),
                      object.points3d_.at<number_type>(i, 1),
                      object.points3d_.at<number_type>(i, 2));
    }

    int step = N/K;
    object.views_.resize (K);
    for ( int i = 0; i < K; ++i) {
        //View view = generateView (object.points3d_, noise, i * step, (i + 1) * step, false, 0); // only Z translation
        cout << "KikouSimulateur" << endl;
        cout << object.points3d_ << endl;
        View view = generateView (object.points3d_, noise, i * step, (i + 1) * step, false, 3); // random
        object.views_[i] = view;
    }

    return object;
}

View Simulateur::generateView (Mat points3d, double noise, int start, int end, bool kinect, int pose) {
    Mat Rvw, tvw;
    if ( pose == 0 ) {
        specific_pose0 (Rvw, tvw);
    } else if ( pose == 1 ) {
        specific_pose45 (Rvw, tvw);
    } else if ( pose == 2 ) {
        specific_pose315 (Rvw, tvw);
    } else {
        random_pose (Rvw, tvw);
    }

    Mat indices (end - start, 1, matrix_type);
    for ( int j = start; j < end; ++j )
        indices.at<number_type>(j-start) = j;
    Mat view_points3d = points3d (Range(start, end), Range(0, 3));

    Mat Pvw = Rt2P (Rvw, tvw);
    Mat Pwv = Pvw.inv();
    Mat Rwv, twv;
    P2Rt (Pwv, Rwv, twv);
//    cout << K_ << endl;
//    cout << noise << endl;
//    cout << Rwv << endl;
//    cout << twv << endl;
//    cout << view_points3d.size() << endl;
    View view (K_, noise, Rwv, twv, view_points3d, indices, kinect);

    return view;
}
////////////////////////////////////////////////////////////////////////////////
// RANDOM GENERATION PART
////////////////////////////////////////////////////////////////////////////////
void Simulateur::specific_pose0 (Mat &R, Mat &t) {
    R = Mat::eye (3, 3, matrix_type);
    t = Mat::zeros (1, 3, matrix_type);

    const number_type range = 1;
    t.at<number_type>(0) = myRand(0, range * 1);
    t.at<number_type>(1) = myRand(0, range * 1);
    t.at<number_type>(2) = myRand(0, range * 1);
//    t.at<number_type>(0) = 0;
//    t.at<number_type>(1) = 0;
//    t.at<number_type>(2) = 6;
}

void Simulateur::specific_pose45 (Mat &R, Mat &t) {
    R = Mat::eye (3, 3, matrix_type);
    t = Mat::zeros (1, 3, matrix_type);

    R.at<number_type>(0,0) = sqrt(2.0)/2.0;
    R.at<number_type>(0,1) = sqrt(2.0)/2.0;

    R.at<number_type>(1,0) = -sqrt(2.0)/2.0;
    R.at<number_type>(1,1) = sqrt(2.0)/2.0;

    const number_type range = 1;
    t.at<number_type>(0) = myRand(0, range * 1);
    t.at<number_type>(1) = myRand(0, range * 1);
    t.at<number_type>(2) = myRand(0, range * 10);
}

void Simulateur::specific_pose315 (Mat &R, Mat &t) {
    R = Mat::eye (3, 3, matrix_type);
    t = Mat::zeros (1, 3, matrix_type);

    R.at<number_type>(0,0) = sqrt(2.0)/2.0;
    R.at<number_type>(0,1) = -sqrt(2.0)/2.0;

    R.at<number_type>(1,0) = sqrt(2.0)/2.0;
    R.at<number_type>(1,1) = sqrt(2.0)/2.0;

    t.at<number_type>(0) = 0;
    t.at<number_type>(1) = 0;
    t.at<number_type>(2) = 6;
}

void Simulateur::random_pose(Mat &R, Mat &t) {
    R = Mat::eye (3, 3, matrix_type);
    t = Mat::zeros (1, 3, matrix_type);

    const number_type range = 1;

    number_type phi   = myRand(0, range * 3.14159 * 2);
    number_type theta = myRand(0, range * 3.14159);
    number_type psi   = myRand(0, range * 3.14159 * 2);

    R.at<number_type>(0,0) = cos(psi) * cos(phi) - cos(theta) * sin(phi) * sin(psi);
    R.at<number_type>(0,1) = cos(psi) * sin(phi) + cos(theta) * cos(phi) * sin(psi);
    R.at<number_type>(0,2) = sin(psi) * sin(theta);

    R.at<number_type>(1,0) = -sin(psi) * cos(phi) - cos(theta) * sin(phi) * cos(psi);
    R.at<number_type>(1,1) = -sin(psi) * sin(phi) + cos(theta) * cos(phi) * cos(psi);
    R.at<number_type>(1,2) = cos(psi) * sin(theta);

    R.at<number_type>(2,0) = sin(theta) * sin(phi);
    R.at<number_type>(2,1) = -sin(theta) * cos(phi);
    R.at<number_type>(2,2) = cos(theta);

//    t.at<number_type>(0) = myRand(0, range * 1);
//    t.at<number_type>(1) = myRand(0, range * 1);
//    t.at<number_type>(2) = myRand(0, range * 1);
    t.at<number_type>(0) = 0;
    t.at<number_type>(1) = 0;
    t.at<number_type>(2) = 6;
}

void Simulateur::random_point(number_type & Xw, number_type & Yw, number_type & Zw) {
  number_type theta = myRand(0, 3.14159), phi = myRand(0, 2 * 3.14159), R = myRand(0, +2);

  Xw = sin(theta) * sin(phi) * R;
  Yw = -sin(theta) * cos(phi) * R;
  Zw =  cos(theta) * R;
}

////////////////////////////////////////////////////////////////////////////////
//  PRIVATE METHODS
////////////////////////////////////////////////////////////////////////////////
