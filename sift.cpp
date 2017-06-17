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
    Mat orig2 = imread(file2);// 原始文件

    // Mat orig1 = imread("1.jpg");
    // Mat orig2 = imread("2.jpg");

    const int area = 100;//重叠部分大小

    // 取左图右侧area部分
    // 取右图左侧area
    Mat img1 = orig1(Rect(Point(orig1.cols-area, 0), Point(orig1.cols, orig1.rows)));
    Mat img2 = orig2(Rect(Point(0, 0), Point(area, orig2.rows)));

    // 定义sift detector
    SiftFeatureDetector detector;
    vector<KeyPoint> kpt1, kpt2;
    // 计算特征点，无关联
    detector.detect(img1, kpt1);
    detector.detect(img2, kpt2);

    //计算description
    SiftDescriptorExtractor extractor;
    Mat des1, des2;
    extractor.compute(img1, kpt1, des1);
    extractor.compute(img2, kpt2, des2);

    // 进行FLANN匹配，结果保存在matches中
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match(des1, des2, matches, Mat());

    // 定义threshold
    float THRESHOLD = orig1.rows / 10;
    // new key point,
    vector<KeyPoint> npt1, npt2;
    vector<DMatch> new_matches;
    int index = 0, dx = 0, dy = 0;

    // 符合要求的加到new_matches里
    for (int i = 0; i < matches.size(); i++){
        if (fabs(kpt1[matches[i].queryIdx].pt.y -
                kpt2[matches[i].trainIdx].pt.y) < THRESHOLD){
            // cout << kpt1[matches[i].queryIdx].pt.y << " "
            // << kpt2[matches[i].trainIdx].pt.y << " "
            // << fabs(kpt1[matches[i].queryIdx].pt.y - kpt2[matches[i].trainIdx].pt.y) <<' '
            // << THRESHOLD << endl;
            npt1.push_back(kpt1[matches[i].queryIdx]);
            npt2.push_back(kpt2[matches[i].trainIdx]);
            dx += kpt2[matches[i].trainIdx].pt.x + (area - kpt1[matches[i].queryIdx].pt.x);
            dy += kpt2[matches[i].trainIdx].pt.y - kpt1[matches[i].queryIdx].pt.y;

            matches[i].queryIdx = index;
            matches[i].trainIdx = index;
            new_matches.push_back(matches[i]);
            index++;
        }
    }
    //dx, dy 为各特征点对的差的均值
    dx /= (index-1);
    dy /= (index-1);

    cout << "dx: " << dx << "\tdy:" << dy <<endl;

    // define the size of result
    int nrow = orig1.rows - abs(dy);
    int ncol = orig1.cols + orig2.cols - abs(dx);

    Mat newimg(nrow, ncol, orig1.type());
    Mat roi1, roi2;

    if( dy > 0){
        roi1 = orig1(Rect(Point(0, 0), Point(orig1.cols , orig1.rows - dy)));
        roi2 = orig2(Rect(Point(0, dy), Point(orig2.cols, orig2.rows)));
    }else{
        roi1 = orig1(Rect(Point(0, -dy), Point(orig1.cols, orig1.rows)));
        roi2 = orig2(Rect(Point(0, 0), Point(orig2.cols, orig2.rows + dy)));
    }

    // 重合部分取平均
    // roi2 减半，加上roi1的一半
    roi2(Rect(Point(0, 0), Point(dx, roi2.rows))) *= 0.5;
    roi2(Rect(Point(0, 0), Point(dx, roi2.rows))) +=
       roi1(Rect(Point(roi1.cols - abs(dx), 0), Point(roi1.cols, roi1.rows))) * 0.5;

    // 将roi1, roi2 合并到newimg
    roi1.copyTo(newimg(Rect(Point(0, 0), Point(roi1.cols, roi1.rows))));
    roi2.copyTo(newimg(Rect(Point(roi1.cols - abs(dx) , 0),
            Point(roi1.cols + roi2.cols - abs(dx), roi2.rows))));
    imshow("1", orig1);
    imshow("2", orig2);
    imshow("new", newimg);

    // Mat img_match;
    // drawMatches(img1, npt1, img2, npt2, new_matches, img_match);
    // 耗时
    cout << "cost: " << (clock() - t1)*1000/ CLOCKS_PER_SEC << endl;
    // imshow("match",img_match);
    waitKey(0);

}
