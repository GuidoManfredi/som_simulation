#include <fstream>

#include "XpCalibrate.h"
#include "Simulateur2.h"

using namespace std;
using namespace cv;

void testFaugeras ();
void testSimulateur2 (int N);

// show that the error becomes too big from some value of noise
// choose this noise as threshold for the RANSAC scheme.
// Use the RANSAC version on various run with various point numbers.
int main() {
    //testSimulateur2 (5);
//    testFaugeras ();

    XpCalibrate xp;
    cv::Mat rotation_estimate, translation_estimate,
            rotation_refine, translation_refine;
    xp.multiXpOnlineTrainNoiseVsLocalisationError (100000, 6,
                                                   0, 0.5, 5,
                                                   rotation_estimate, translation_estimate,
                                                   rotation_refine, translation_refine);

    writeMat ("estimate_rotation_noise_vs_precision.txt", rotation_estimate);
    writeMat ("estimate_translation_noise_vs_precision.txt", translation_estimate);
    writeMat ("refine_rotation_noise_vs_precision.txt", rotation_refine);
    writeMat ("refine_translation_noise_vs_precision.txt", translation_refine);

    return 0;
}

void testFaugeras () {
    Mat K = (Mat_<number_type>(3, 3) << 800,   0, 320,
                                          0, 800, 240,
                                          0,   0,   1);
    double noise = 3;
    Simulateur2 sim (K);
    Mat points2d, points3d, Rtrue, ttrue;
    sim.generatePoints2d3d(6, noise, points2d, points3d, Rtrue, ttrue);
    POM pom;
    Mat Kest, Rest, test;
    //cout << points3d << endl;
    //cout << points2d << endl;
    pom.solveCalibrationMethods (points3d, points2d, Kest, Rest, test, LINEAR3);

    number_type rot_err, transl_err;
    relative_error2(Rtrue, ttrue, Rest, test, rot_err, transl_err);
    cout << rot_err << " " << transl_err << endl;

//    cout << K << endl << Rtrue << endl << ttrue << endl;
//    cout << KRt2P (K, Rtrue, ttrue) << endl;
//    cout << Kest << endl << Rest << endl << test << endl;
//    cout << KRt2P (Kest, Rest, test) << endl;
}

void testSimulateur2 (int N) {
    Mat K = (Mat_<number_type>(3, 3) << 800,   0, 320,
                                          0, 800, 240,
                                          0,   0,   1);
    Simulateur2 sim (K);
    Mat points2d, points3d, R, t;
    sim.generatePoints2d3d(N, 0, points2d, points3d, R, t);
    cout << points3d << endl;
    cout << points2d << endl;
    cout << R << endl;
    cout << t << endl;

    for ( int i = 0; i < N; ++i) {
        Mat p3d = K * R * points3d.row(i).t() + K * t.t();
        p3d /= p3d.at<number_type>(2);
        cout << p3d << endl;
        cout << points2d.row(i) << endl;
    }
}
