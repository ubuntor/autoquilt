#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include "PerfectMatching.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Exact_predicates_tag Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, CGAL::Default, Itag> CDT;

using namespace cv;

RNG rng(12345); // ???

int main(int argc, char** argv) {
    Mat image, image_g1, image_g2, image_gdiff, image_threshold;
    std::vector<std::vector<Point>> contours;
    CDT cdt;

    if (argc <= 1) {
        printf("Usage: %s image\n", argv[0]);
        return 1;
    }
    image = imread(argv[1], IMREAD_GRAYSCALE);
    if (!image.data) {
        printf("no image data\n");
        return 1;
    }

    // required edges
    // TODO: actual FDoG
    GaussianBlur(image, image_g1, Size(1, 1), 0, 0);
    GaussianBlur(image, image_g2, Size(3, 3), 0, 0);
    image_gdiff = image_g1 - image_g2;
    threshold(image_gdiff, image_threshold, 127, 255, THRESH_BINARY | THRESH_OTSU);
    findContours(image_threshold, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    // optional edges
    // TODO: CGAL
    for (std::vector<Point> i : contours) {
        for (int j = 0; j < i.size()-1; j++) {
            cdt.insert_constraint(CDT::Point(i[j].x,i[j].y), CDT::Point(i[j+1].x,i[j+1].y));
        }
        cdt.insert_constraint(CDT::Point(i[i.size()-1].x,i[i.size()-1].y), CDT::Point(i[0].x,i[0].y));
    }
    std::cout << cdt.is_valid() << std::endl;



    // ???
      Mat drawing = Mat::zeros( image_threshold.size(), CV_8UC3 );
      for( int i = 0; i< contours.size(); i++ )
         {
           Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
           drawContours( drawing, contours, i, color);
         }
    // ???




    // display
    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", drawing);
    waitKey(0);
    return 0;
}
