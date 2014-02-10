#include "POM2D.h"

using namespace std;
using namespace cv;

POM2D::POM2D() {}

void POM2D::compute (View train, View current) {
    /*
    if ( !train.initialised_ )
        computePoseTrain (train, current);
    else
        computePoseCurrent (train, current);
        */
}

match(View train, View current, matches)
convertToCameraFrame (points3d, current)
computeLocalFrame (points3d)
computeTrainPose (View train, View current, matches)

void POM2D::match (View train, View current,
                   vector<int> &matches) {
    inter (train.indices_, current.indices_, matches);
}

void computeLocalFrame (Mat points3d, Mat &R, Mat &t) {
    getBarycentre (points3d, R, t);
}

void POM2D::computePoseTrain (Mat train_points2d, Mat current_points3d,
                              View &train, View current) {
    Mat local_R, local_t;
    // put world 3D points in current camera frame
    Mat points3d_camera;
    transform (current.Rw_, current.tw_, current_points3d, points3d_camera);
    // compute new local frame
    setupLocalFrame (points3d_camera,
                     local_R, local_t);
    // express 3D points in new local frame
    Mat points3d_local;
    transform (local_R, local_t, points3d_camera, points3d_local);
    // solve pnp to get transformation between training image and local frame
    Mat rvec;
    solvePnP (points3d_local, train_points2d, current.K_, vector<double>(), rvec, train.to_);

    Rodrigues (rvec, train.Ro_);
}

void POM2D::computePoseCurrent (Mat train_points2d, Mat current_points3d,
                                View train, View current) {
    Mat points3d_camera;
    transform (current.Rw_, current.tw_, current_points3d, points3d_camera);

    Mat KP = current.K_ * train.Ro_;
    Mat Kt = current.K_ * train.to_;
    cout << train.to_ << endl;
    cout << current.K_ << endl;
    cout << Kt << endl;

    //for ( size_t i = 0; i < train_points2d; )

    Mat rvec;
    solvePnP (points3d_camera, train_points2d, KP, vector<double>(), rvec, current.to_);

    Rodrigues (rvec, current.Ro_);
}
////////////////////////////////////////////////////////////////////////////////
//  PRIVATE METHODES
////////////////////////////////////////////////////////////////////////////////
