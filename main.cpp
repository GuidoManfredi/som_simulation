#include "Simulateur.h"
#include "POM.h"

using namespace std;
using namespace cv;

void testAll ();

int main() {
    testAll ();

    return 0;
}

void testAll () {
    Mat K = (Mat_<number_type>(3, 3) << 800,   0, 320,
                                          0, 800, 240,
                                          0,   0,   1);
    cout << K << endl;
    Simulateur sim (K);
    POM pom;

    cout << "Creating test object" << endl;
    int N = 4000;
    int k = 4;
    Object test_object = sim.generateObject (N, k);
    // Ground truth
    Mat Rwc = test_object.views_[0].Rwv_;
    Mat twc = test_object.views_[0].twv_;
    cout << test_object.views_[0].twv_ << endl;

    cout << "Creating test view" << endl;
    View test_view1 = sim.generateView (test_object.points3d_, 5, N/k, true, 4);
    View test_view2 = sim.generateView (test_object.points3d_, N/k-200, N/k+200, true, 4);
    //View test_view2 = sim.generateView (test_object.points3d_, 0, 6, true);

    cout << "Test matching" << endl;
    std::vector<std::vector<cv::DMatch> > matches1, matches2;
    pom.match (test_view1, test_object.views_, matches1);
    pom.match (test_view2, test_object.views_, matches2);
    cout << "Number of matches 1: " << matches1[0].size() << endl;
    cout << "Number of matches 2: " << matches2[0].size() << endl;
//    for ( size_t i = 0; i < matches2.size(); ++i ) {
//        for ( size_t j = 0; j < matches2[i].size(); ++j ) {
//            cout << matches2[i][j].queryIdx << " " << matches2[i][j].trainIdx << endl;
//        }
//        cout << endl;
//    }

    cout << "*** Testing train pose computation ***" << endl;
    pom.computeTrainPose (test_view1, test_object.views_[0], matches1[0]);
    cout << test_object.views_[0].Rov_ << endl;
    cout << test_object.views_[0].tov_ << endl;

    cout << "*** Ground truth ***" << endl;
    Mat Row = test_view1.Rwv_.t(); // same as test_view1.Rvw
    Mat tow;
    getBarycentre (test_view1.world_points3d_, tow);
    Mat Roc = Rwc * Row;
    Mat toc = tow + twc;
    cout << Roc << endl;
    cout << toc << endl;

    cout << "*** Testing current pose computation ***" << endl;
    pom.computeCurrentPose (test_view2, test_object.views_[0], matches2[0]);
    cout << test_view2.Rov_ << endl;
    cout << test_view2.tov_ << endl;
    cout << "*** Ground truth ***" << endl;
    Mat Rwv2 = test_view2.Rwv_;
    Mat twv2 = test_view2.twv_;
    Mat Rov2 = Rwv2 * Row;
    Mat tov2 = twv2 + tow;
    cout << Rov2 << endl;
    cout << tov2 << endl;
}
