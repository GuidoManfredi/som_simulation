#include "Simulateur.h"
#include "POM.h"
#include "POM2.h"

using namespace std;
using namespace cv;

void num_matches_matches_xp (int N, number_type noise,
                             int num_train, int num_current,
                             Mat &R_error, Mat &t_error);

void noise_many_xp (int N);
void noise_xp (int N, int num_matches, number_type noise,
               Mat &R_error, Mat &t_error);
void num_match_many_xp (int N);
void num_match_xp (int N, int num_matches, number_type max_noise,
                    Mat &R_error, Mat &t_error);

void detect_train (View view_train, View &train,
                   number_type &R_err, number_type &t_err);
void detect (View &current, View train,
             number_type &R_err, number_type &t_err);

void testAll ();

Mat K = (Mat_<number_type>(3, 3) << 800,   0, 320,
                                      0, 800, 240,
                                      0,   0,   1);
Simulateur sim (K);
//POM2 pom;
POM pom;

int main() {
    //testAll ();
    Mat Rerr, terr;
    num_matches_matches_xp (1500, 4.0, 5, 25, Rerr, terr);
    //noise_many_xp (100);
    //num_match_many_xp (100);
    return 0;
}

void num_matches_matches_xp (int N, number_type noise,
                             int num_train, int num_current,
                             Mat &R_error, Mat &t_error) {
    R_error = Mat::zeros (num_current, num_train, matrix_type);
    t_error = Mat::zeros (num_current, num_train, matrix_type);

    Object object = sim.generateObject (N, 1, noise);
    View train = sim.generateView (object.points3d_, noise, 0, N, false, 4);

    int step_train = 250 / num_train;
    int step_current = 250 / num_current;
    for (size_t i = 0; i < num_train; ++i) {
        number_type R_train_err, t_train_err;
        int max_train = (i + 1) * step_train;
        //cout << "Max train : " << max_train << endl;
        View view_train = sim.generateView (object.points3d_, noise, 0, max_train, true, 4);
        detect_train (view_train, train, R_train_err, t_train_err);
        for (size_t j = 0; j < num_current; ++j) {
            number_type R_view_err, t_view_err;
            int max_current = (j + 1) * step_current;
            //cout << "Max current : " << max_current << endl;
            View view = sim.generateView (object.points3d_, noise, 0, max_current, true, 4);
            detect (view, train, R_view_err, t_view_err);
            // save results
            R_error.at<number_type>(j, i) = R_view_err;
            t_error.at<number_type>(j, i) = t_view_err;
        }
    }
    //writeMat ("error.txt", R_error);
    writeMat ("error.txt", t_error);
}

void noise_many_xp (int N) {
    int max_noise = 15;
    Mat R_results = Mat::zeros (max_noise, N, matrix_type);
    Mat t_results = Mat::zeros (max_noise, N, matrix_type);
    for ( int i = 0; i < N; ++i ) {
        Mat Rerr, terr;
        noise_xp (1000, 50, max_noise, Rerr, terr);
        Rerr.copyTo(R_results.col(i));
        terr.copyTo(t_results.col(i));
    }
    //writeMat ("error.txt", R_results);
    //writeMat ("error.txt", t_results);
}

void noise_xp (int N, int num_matches, number_type max_noise,
               Mat &R_error, Mat &t_error) {
    R_error = Mat::zeros (max_noise, 1, matrix_type);
    t_error = Mat::zeros (max_noise, 1, matrix_type);

    Object object = sim.generateObject (N, 1, 0);

    for (size_t noise = 0; noise < max_noise; ++noise) {
        number_type R_train_err, t_train_err, R_view_err, t_view_err;

        View train = sim.generateView (object.points3d_, noise, 0, N, false, 0);
        View view_train = sim.generateView (object.points3d_, noise, 0, num_matches, true, 4);
        View view = sim.generateView (object.points3d_, noise, N - num_matches, N, true, 4);

        detect_train (view_train, train, R_train_err, t_train_err);
        detect (view, train, R_view_err, t_view_err);
//        R_error.at<number_type>(noise) = R_train_err;
//        t_error.at<number_type>(noise) = t_train_err;
        R_error.at<number_type>(noise) = R_view_err;
        t_error.at<number_type>(noise) = t_view_err;
    }
}

void num_match_many_xp (int N) {
    int max_num_matches = 200;
    Mat R_results = Mat::zeros (max_num_matches, N, matrix_type);
    Mat t_results = Mat::zeros (max_num_matches, N, matrix_type);
    for ( int i = 0; i < N; ++i ) {
        Mat Rerr, terr;
        num_match_xp (1000, max_num_matches, 4, Rerr, terr);
        Rerr.copyTo(R_results.col(i));
        terr.copyTo(t_results.col(i));
    }
    writeMat ("error.txt", R_results);
    //writeMat ("error.txt", t_results);
}

void num_match_xp (int N, int max_num_matches, number_type noise,
                    Mat &R_error, Mat &t_error) {
    R_error = Mat::zeros (max_num_matches, 1, matrix_type);
    t_error = Mat::zeros (max_num_matches, 1, matrix_type);

    Object object = sim.generateObject (N, 1, 0);

    for (size_t idx = 1; idx < max_num_matches/10; ++idx ) {
        int num_matches = idx * 10;
        number_type R_train_err, t_train_err, R_view_err, t_view_err;

        View train = sim.generateView (object.points3d_, noise, 0, N, false, 0);
        View view_train = sim.generateView (object.points3d_, noise, 0, num_matches, true, 4);
        View view = sim.generateView (object.points3d_, noise, N - num_matches, N, true, 4);

        detect_train (view_train, train, R_train_err, t_train_err);
        detect (view, train, R_view_err, t_view_err);

        R_error.at<number_type>(idx) = R_view_err;
        t_error.at<number_type>(idx) = t_view_err;
    }
}

void detect_train (View view_train, View &train,
                   number_type &R_err, number_type&t_err) {
    // ESTIMATE
    std::vector<cv::DMatch> matches;
    pom.match (view_train, train, matches);
    number_type err = pom.computeTrainPose (view_train, train, matches);
    // GROUND TRUTH
    train.Row_ = view_train.Rwv_.t(); // same as test_view1.Rvw
    getBarycentre (view_train.world_points3d_, train.tow_);
    Mat Rwc = train.Rwv_;
    Mat twc = train.twv_;
    Mat Roc = Rwc * train.Row_;
    Mat toc = train.tow_ + twc;
    // COMPUTE ERROR
    relative_error2 (Roc, toc, train.Rov_, train.tov_,
                     R_err, t_err);
}

void detect (View &current, View train,
             number_type &R_err, number_type &t_err) {
    // ESTIMATE
    std::vector<cv::DMatch> matches;
    pom.match (current, train, matches);
    number_type err = pom.computeCurrentPose (current, train, matches);
    // GROUND TRUTH
    Mat Rwv2 = current.Rwv_;
    Mat twv2 = current.twv_;
    Mat Rov2 = Rwv2 * train.Row_;
    Mat tov2 = twv2 + train.tow_;
    // COMPUTE ERROR
    relative_error2 (Rov2, tov2.t(), current.Rov_, current.tov_, // fucking t dans tous les sens
                     R_err, t_err);
}

void testAll () {
    cout << "Creating test object" << endl;
    int N = 4000;
    int k = 4;
    number_type noise = 20.0;
    Object test_object = sim.generateObject (N, k, noise);
    // Ground truth
    Mat Rwc = test_object.views_[0].Rwv_;
    Mat twc = test_object.views_[0].twv_;
    //cout << test_object.views_[0].twv_ << endl;

    cout << "Creating test view" << endl;
    View test_view1 = sim.generateView (test_object.points3d_, noise, 0, 150, true, 4);
    View test_view2 = sim.generateView (test_object.points3d_, noise, N/k-200, N/k+200, true, 4);

    cout << "Test matching" << endl;
    std::vector<std::vector<cv::DMatch> > matches1, matches2;
    pom.match (test_view1, test_object.views_, matches1);
    pom.match (test_view2, test_object.views_, matches2);
//    cout << "Number of matches 1: " << matches1[0].size() << endl;
//    cout << "Number of matches 2: " << matches2[0].size() << endl;

    cout << "*** Testing train pose computation ***" << endl;
    number_type err1 = pom.computeTrainPose (test_view1, test_object.views_[0], matches1[0]);
    cout << "Reprojection error current pose: " << err1 << endl;
//    cout << test_object.views_[0].Rov_ << endl;
//    cout << test_object.views_[0].tov_ << endl;

    cout << "*** Ground truth ***" << endl;
    Mat Row = test_view1.Rwv_.t(); // same as test_view1.Rvw
    Mat tow;
    getBarycentre (test_view1.world_points3d_, tow);
    Mat Roc = Rwc * Row;
    Mat toc = tow + twc;
//    cout << Roc << endl;
//    cout << toc << endl;
    number_type R_error, t_error;
    relative_error2 (Roc, toc, test_object.views_[0].Rov_, test_object.views_[0].tov_,
                    R_error, t_error);
    cout << R_error << " " << t_error << endl;
/*
    cout << "*** Testing current pose computation ***" << endl;
    number_type err2 = pom.computeCurrentPose (test_view2, test_object.views_[0], matches2[0]);
    cout << "Reprojection error current pose: " << err2 << endl;
//    cout << test_view2.Rov_ << endl;
//    cout << test_view2.tov_ << endl;

    cout << "*** Ground truth ***" << endl;
    Mat Rwv2 = test_view2.Rwv_;
    Mat twv2 = test_view2.twv_;
    Mat Rov2 = Rwv2 * Row;
    Mat tov2 = twv2 + tow;
//    cout << Rov2 << endl;
//    cout << tov2 << endl;
    //cout << tov2.size() << " " << test_view2.tov_ << endl;
    relative_error (Rov2, tov2.t(), test_view2.Rov_, test_view2.tov_,
                    R_error, t_error);
    cout << R_error << " " << t_error << endl;
*/
}
