#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    string file1, file2;
    cin >> file1 >> file2;
    clock_t t1 = clock();
    Mat orig1 = imread(file1);
    Mat orig2 = imread(file2);
    // Mat orig1 = imread("1.jpg");
    // Mat orig2 = imread("2.jpg");

    Mat img1 = orig1(Rect(Point(orig1.cols-100, 0), Point(orig1.cols, orig1.rows)));
    Mat img2 = orig2(Rect(Point(0, 0), Point(100, orig2.rows)));

    SiftFeatureDetector detector;
    vector<KeyPoint> kpt1, kpt2;
    detector.detect(img1, kpt1);
    detector.detect(img2, kpt2);

    SiftDescriptorExtractor extractor;
    Mat des1, des2;
    extractor.compute(img1, kpt1, des1);
    extractor.compute(img2, kpt2, des2);

    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match(des1, des2, matches, Mat());

    float THRESHOLD = orig1.rows / 10;
    vector<KeyPoint> npt1, npt2;
    vector<DMatch> new_matches;
    int index = 0, dx = 0, dy = 0;

    for (int i = 0; i < matches.size(); i++){
        if (fabs(kpt1[matches[i].queryIdx].pt.y -
                kpt2[matches[i].trainIdx].pt.y) < THRESHOLD){
            // cout << kpt1[matches[i].queryIdx].pt.y << " "
            // << kpt2[matches[i].trainIdx].pt.y << " "
            // << fabs(kpt1[matches[i].queryIdx].pt.y - kpt2[matches[i].trainIdx].pt.y) <<' '
            // << THRESHOLD << endl;
            npt1.push_back(kpt1[matches[i].queryIdx]);
            npt2.push_back(kpt2[matches[i].trainIdx]);
            dx += kpt2[matches[i].trainIdx].pt.x + (100 - kpt1[matches[i].queryIdx].pt.x);
            dy += kpt2[matches[i].trainIdx].pt.y - kpt1[matches[i].queryIdx].pt.y;
            //cout << "dx: " << dx << "\tdy:" << dy <<endl;

            matches[i].queryIdx = index;
            matches[i].trainIdx = index;
            new_matches.push_back(matches[i]);
            index++;
        }
    }
    dx /= (index-1);
    dy /= (index-1);
    //cout << new_matches.size() << endl;

    cout << "dx: " << dx << "\tdy:" << dy <<endl;


    int nrow = orig1.rows - abs(dy);
    int ncol = orig1.cols + orig2.cols - abs(dx);

    //
    Mat newimg(nrow, ncol, orig1.type());
    Mat roi1, roi2;
    if( dy > 0){
        roi1 = orig1(Rect(Point(0, 0), Point(orig1.cols , orig1.rows - dy)));
        roi2 = orig2(Rect(Point(0, dy), Point(orig2.cols, orig2.rows)));
    }else{
        roi1 = orig1(Rect(Point(0, -dy), Point(orig1.cols, orig1.rows)));
        roi2 = orig2(Rect(Point(0, 0), Point(orig2.cols, orig2.rows + dy)));
    }
    //imshow("roi1", roi1);
    //imshow("roi2", roi2);
    roi2(Rect(Point(0, 0), Point(dx, roi2.rows))) *= 0.5;
    roi2(Rect(Point(0, 0), Point(dx, roi2.rows))) +=
        roi1(Rect(Point(roi1.cols - abs(dx), 0), Point(roi1.cols, roi1.rows))) * 0.5;

    roi1.copyTo(newimg(Rect(Point(0, 0), Point(roi1.cols, roi1.rows))));
    roi2.copyTo(newimg(Rect(Point(roi1.cols - abs(dx) , 0),
            Point(roi1.cols + roi2.cols - abs(dx), roi2.rows))));
    imshow("1", orig1);
    imshow("2", orig2);
    imshow("new", newimg);

    // Mat img_match;
    // drawMatches(img1, npt1, img2, npt2, new_matches, img_match);
    // cout << "cost: " << (clock() - t1)*1000/ CLOCKS_PER_SEC << endl;
    // imshow("match",img_match);
    waitKey(0);

}
