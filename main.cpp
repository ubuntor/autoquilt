#include "PerfectMatching.h"
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <boost/graph/adjacency_list.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>

// TODO move this somewhere else
// CGAL triangulation
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_2<int, K> Vb;
typedef CGAL::Constrained_triangulation_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> TDS;
typedef CGAL::Exact_predicates_tag Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, TDS, Itag> CDT;

// Boost graph
struct edge_necessity_tag_t {
    typedef boost::edge_property_tag kind;
};
enum edge_necessity_t { required, optional };

typedef boost::property<edge_necessity_tag_t, edge_necessity_t> EdgeProperty;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                              boost::no_property, EdgeProperty>
    Graph;

using namespace cv;

RNG rng(12345); // ???

int main(int argc, char **argv) {
    // TODO: split this up
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
    threshold(image_gdiff, image_threshold, 127, 255,
              THRESH_BINARY | THRESH_OTSU);
    findContours(image_threshold, contours, RETR_LIST,
                 CHAIN_APPROX_SIMPLE); // ???

    // optional edges
    std::cout << "Contours found" << std::endl;
    for (std::vector<Point> i : contours) {
        for (int j = 0; j < i.size() - 1; j++) {
            cdt.insert_constraint(CDT::Point(i[j].x, i[j].y),
                                  CDT::Point(i[j + 1].x, i[j + 1].y));
        }
        cdt.insert_constraint(CDT::Point(i[i.size() - 1].x, i[i.size() - 1].y),
                              CDT::Point(i[0].x, i[0].y));
    }
    assert(cdt.is_valid());

    int index = 0;
    for (CDT::Finite_vertices_iterator it = cdt.finite_vertices_begin();
         it != cdt.finite_vertices_end(); ++it) {
        std::cout << it->point() << std::endl;
        it->info() = index;
        index++;
    }

    Graph G(cdt.number_of_vertices());

    for (CDT::Finite_edges_iterator eit = cdt.finite_edges_begin();
         eit != cdt.finite_edges_end(); ++eit) {
        CDT::Face_handle f = eit->first;
        int i = eit->second;
        CDT::Vertex_handle v1 = f->vertex(CDT::cw(i));
        CDT::Vertex_handle v2 = f->vertex(CDT::ccw(i));
        std::cout << "(" << *v1 << ") [" << v1->info() << "] to (" << *v2
                  << ") [" << v2->info()
                  << "] constrained: " << cdt.is_constrained(*eit) << std::endl;
        edge_necessity_t necessity =
            cdt.is_constrained(*eit) ? required : optional;
        boost::add_edge(v1->info(), v2->info(), necessity, G);
    }
    std::cout << cdt.number_of_vertices() << " vertices" << std::endl;

    // ???
    Mat drawing = Mat::zeros(image_threshold.size(), CV_8UC3);
    for (int i = 0; i < contours.size(); i++) {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                              rng.uniform(0, 255));
        drawContours(drawing, contours, i, color);
    }
    // ???

    // display
    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", drawing);
    waitKey(0);
    return 0;
}
