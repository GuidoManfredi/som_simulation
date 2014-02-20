#include "POM.h"

#include "opencv2/core/core_c.h" // fore cv::reduce

using namespace std;
using namespace cv;

void POM::process (Object object, View current) {
    vector<vector<DMatch> > matches;
    match ( current, object.views_, matches);
    for ( size_t i = 0; i < matches.size(); ++i ) {
        if ( !object.views_[i].initialised ) {
            computeTrainPose (object.views_[i], current, matches[i]);
            object.views_[i].initialised = true;
        } else {
            computeCurrentPose (object.views_[i], current, matches[i]);
        }
    }
}

void POM::match ( const View current, const std::vector<View> views,
                  std::vector<std::vector<DMatch> > &matches ) {
    for ( size_t i = 0; i < views.size(); ++i ) {
        vector<DMatch> tmp_matches;
        inter (current.descriptors_, views[i].descriptors_, tmp_matches);
        if ( tmp_matches.size() > 0 ) // only if there are matches
            matches.push_back (tmp_matches);
    }
}

void POM::match ( const View current, const View view,
                  std::vector<DMatch> &matches ) {
    inter (current.descriptors_, view.descriptors_, matches);
}

number_type POM::computeTrainPose (View current, View &train, vector<DMatch> matches) {
    if ( matches.size() < 4) {
        cout << "ComputetrainPose : not enought matches (" << matches.size() << ")." << endl;
        return -1;
    }

    Mat points2d, kinect_points3d;
    train.points2dFromMatches (matches, points2d);
    current.points3dFromMatches (matches, kinect_points3d);

    Mat Rov, tov;
    object2viewFrame (kinect_points3d,
                      Rov, tov);
    // transform 3d points from camera to object object frame
    Mat object_points3d;
    Mat Pov = Rt2P (Rov, tov);
    Mat Pvo = Pov.inv();
    Mat Rvo, tvo;
    P2Rt(Pvo, Rvo, tvo);
    transform (kinect_points3d, Rvo, tvo,
               object_points3d);
    //cout << object_points3d << endl;
    //cout << points2d << endl;
    mySolvePnP (object_points3d, points2d, train.K_, train.Rov_, train.tov_);

    number_type reprojection_error = train.OpenCVreprojectionError (object_points3d, points2d);
    //cout << "OpenCV reprojection error train pose: " << reprojection_error << endl;
    return reprojection_error;
}

void POM::computeTrainIntrinsic (View current, View &train, vector<DMatch> matches) {
    Mat points2d, kinect_points3d;
    train.points2dFromMatches (matches, points2d);
    current.points3dFromMatches (matches, kinect_points3d);
}

number_type POM::computeCurrentPose (View &current, View train, vector<DMatch> matches) {
    Mat points2d, camera_points3d;
    train.points2dFromMatches (matches, points2d);
    current.points3dFromMatches (matches, camera_points3d);

    Mat RR, tt;
    mySolvePnP (camera_points3d, points2d, train.K_, RR, tt);
    // convert A = P * (P')-1 to P' with P' = A-1 * P
    Mat PP = Rt2P (RR, tt);
    Mat P = Rt2P (train.Rov_, train.tov_);
    Mat P2 = PP.inv() * P;
    //cout << PP << endl;
    //cout << PP.inv() << endl;
    P2Rt (P2, current.Rov_, current.tov_);

    number_type reprojection_error = current.OpenCVreprojectionError (RR, tt, camera_points3d, points2d);
    //cout << "OpenCV reprojection error train pose: " << reprojection_error << endl;
    return reprojection_error;
}
////////////////////////////////////////////////////////////////////////////////
//  PRIVATE METHODS
////////////////////////////////////////////////////////////////////////////////
void POM::object2viewFrame (Mat points3d, Mat &Rov, Mat &tov) {
    // compute object to points frame transform
    tov = Mat::zeros(1, 3, matrix_type);
    reduce (points3d, tov, 0, CV_REDUCE_AVG);
    Rov = Mat::eye (3, 3, matrix_type); // keep same rotation as current camera
}

void POM::mySolvePnP (Mat p3d, Mat p2d, Mat K, Mat &Rov, Mat &tov) {
    Mat rvec = Mat::zeros (1, 3, matrix_type);
    tov = Mat::zeros (1, 3, matrix_type);
//    p3d.convertTo(p3d, CV_32F);
//    p2d.convertTo(p2d, CV_32F);
//    K.convertTo(K, CV_32F);
//    solvePnPRansac (p3d, p2d, K, vector<float>(), rvec, tov, false, 1000, 8.0);
    solvePnP (p3d, p2d, K, vector<number_type>(), rvec, tov);

    Rodrigues (rvec, Rov);
}

void POM::solveFaugeras (Mat points3d, Mat points2d, Mat &P) {
    P = Mat::eye (4, 4, matrix_type);
}

/*
void POM::mySolvePnPPoints (Mat p3d, Mat p2d, Mat K, Mat &Rov, Mat &tov) {
    Mat rvec = Mat::zeros (1, 3, matrix_type);
    tov = Mat::zeros (1, 3, matrix_type);

    vector<point3d_type> p3d_vec = Mat_<point3d_type> (p3d.reshape(1, p3d.rows));
    vector<point2d_type> p2d_vec = Mat_<point2d_type> (p2d.reshape(1, p2d.rows));
    //solvePnP (p3d_vec, p2d_vec, K, vector<number_type>(), rvec, tov);
    solvePnP (p3d_vec, p2d_vec, K, vector<number_type>(), rvec, tov, CV_EPNP);

    Rodrigues (rvec, Rov);
    Rov.convertTo (Rov, matrix_type);
    tov.convertTo (tov, matrix_type);
}
*/
