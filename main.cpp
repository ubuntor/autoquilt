#include "PerfectMatching.h"
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation.h>
#include <CGAL/Triangulation_vertex_base_with_id_2.h>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <stdio.h>
#include <stdlib.h>

// max radius for bridges
#define MAX_BRIDGE_SIZE 8
// required edge weight
#define ALPHA_E 1
// optional edge weight
#define ALPHA_O 2

// TODO: tweak these
// alignment weight in optional edge weight
#define BETA 0.25
// curvature weight in matching path line graph edge weight
#define GAMMA 2
#define MATCHING_DISTANCE_SCALE 1000

// TODO move this somewhere else
// CGAL triangulation
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_id_2<K> Vb;
typedef CGAL::Constrained_triangulation_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> TDS;
typedef CGAL::Exact_predicates_tag Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, TDS, Itag> CDT;

// Boost graph
struct edge_necessity_tag_t {
    typedef boost::edge_property_tag kind;
};
enum edge_necessity_t { required, optional };

typedef boost::property<
    edge_necessity_tag_t, edge_necessity_t,
    boost::property<boost::edge_weight_t, double, boost::property<boost::edge_index_t, int>>>
    EdgeProperty;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property,
                              EdgeProperty>
    Graph;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property,
                              boost::property<boost::edge_weight_t, double>>
    LineGraph;
typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
typedef boost::graph_traits<Graph>::edge_descriptor Edge;

using namespace cv;

RNG rng(12345); // ???

int main(int argc, char **argv) {
    // TODO: split this up
    Mat image, image_g1, image_g2, image_gdiff, image_threshold;
    std::vector<std::vector<Point>> contours;
    CDT cdt;

    // TODO: actual arg parsing
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
                 CHAIN_APPROX_TC89_KCOS); // ???

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

    Graph g(cdt.number_of_vertices());
    std::vector<Point> positions(cdt.number_of_vertices());

    int id = 0;
    for (CDT::Finite_vertices_iterator it = cdt.finite_vertices_begin();
         it != cdt.finite_vertices_end(); ++it) {
        // std::cout << it->point() << std::endl;
        it->id() = id;
        positions[id] = Point(it->point().x(), it->point().y());
        id++;
    }

    Mat drawing = Mat::zeros(image_threshold.size(), CV_8UC3); // ???

    // convert cdt to boost graph
    for (CDT::Finite_edges_iterator eit = cdt.finite_edges_begin();
         eit != cdt.finite_edges_end(); ++eit) {
        CDT::Face_handle f = eit->first;
        int i = eit->second;
        CDT::Vertex_handle v1 = f->vertex(CDT::cw(i));
        CDT::Vertex_handle v2 = f->vertex(CDT::ccw(i));
        /*std::cout << "(" << *v1 << ") [" << v1->id() << "] to (" << *v2
                  << ") [" << v2->id()
                  << "] constrained: " << cdt.is_constrained(*eit) << std::endl;*/
        boost::add_edge(v1->id(), v2->id(), cdt.is_constrained(*eit) ? required : optional, g);
    }
    std::cout << cdt.number_of_vertices() << " vertices" << std::endl;

    // add optional bridge edges
    typedef boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;
    IndexMap index = boost::get(boost::vertex_index, g);
    typedef boost::graph_traits<Graph>::vertex_iterator vertex_iter;
    vertex_iter vp, vp_end;
    std::queue<boost::graph_traits<Graph>::vertex_descriptor> queue;
    std::vector<std::pair<int, int>> bridge_edges;
    std::vector<int> distance(boost::num_vertices(g));
    for (boost::tie(vp, vp_end) = boost::vertices(g); vp != vp_end; ++vp) {
        if (index[*vp] % 100 == 0) {
            std::cout << "bridging " << index[*vp] << std::endl;
        }
        queue.push(*vp);
        while (!queue.empty()) {
            typename boost::graph_traits<Graph>::vertex_descriptor v = queue.front();
            int cur_distance = distance[v];
            typename boost::graph_traits<Graph>::adjacency_iterator ai;
            typename boost::graph_traits<Graph>::adjacency_iterator ai_end;
            for (boost::tie(ai, ai_end) = boost::adjacent_vertices(v, g); ai != ai_end; ++ai) {
                if (cur_distance < MAX_BRIDGE_SIZE && index[*ai] != index[*vp] && distance[index[*ai]] == 0) {
                    queue.push(*ai);
                    distance[index[*ai]] = cur_distance+1;
                    // don't bridge immediate neighbors, prevent double counting bridge edges
                    if (cur_distance > 1 && index[*vp] < index[*ai]) {
                        bridge_edges.push_back(std::pair<int,int>(index[*vp], index[*ai]));
                    }
                }
            }
            queue.pop();
        }
        std::fill(distance.begin(), distance.end(), 0);
    }
    std::cout << "done bridging" << std::endl;
    for (std::pair<int,int> bridge_edge : bridge_edges) {
        boost::add_edge(bridge_edge.first, bridge_edge.second, optional, g);
        //line(drawing, positions[bridge_edge.first], positions[bridge_edge.second], CV_RGB(0, 0, 255));
    }
    std::cout << "done adding bridge edges: # verts: " << boost::num_vertices(g)
              << " # edges: " << boost::num_edges(g) << std::endl;

    // mst to connect graph

    std::vector<Edge> mst;
    boost::property_map<Graph, edge_necessity_tag_t>::type necessity = boost::get(edge_necessity_tag_t(), g);
    boost::property_map<Graph, boost::edge_weight_t>::type weights = boost::get(boost::edge_weight_t(), g);
    boost::graph_traits<Graph>::edge_iterator ei, ei_end;
    for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ei++) {
        if (boost::get(necessity, *ei) == required) {
            // required edges are free
            boost::put(weights, *ei, 0);
        } else {
            Point v1 = positions[index[source(*ei, g)]];
            Point v2 = positions[index[target(*ei, g)]];
            // TODO flow alignment term from FDoG
            boost::put(weights, *ei, ALPHA_O*hypot(v1.x-v2.x, v1.y-v2.y));
        }
        
    }
    boost::kruskal_minimum_spanning_tree(g, std::back_inserter(mst));
    for (std::vector<Edge>::iterator ei = mst.begin(); ei != mst.end(); ++ei) {
        boost::put(necessity, *ei, required);
    }
    std::cout << "done mst" << std::endl;

    // TODO: optimization? landmark distance estimation
    std::vector<Vertex> odd_verts;
    for (boost::tie(vp, vp_end) = boost::vertices(g); vp != vp_end; ++vp) {
        if (degree(*vp, g) % 2 == 1) {
            odd_verts.push_back(*vp);
        }
    }
    int num_odd = odd_verts.size();
    std::cout << "# odd: " << num_odd << std::endl;

    // minimum weight matching for odd-degree vertices
    // fill in required edge weights
    for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ei++) {
        if (boost::get(necessity, *ei) == required) {
            Point v1 = positions[index[source(*ei, g)]];
            Point v2 = positions[index[target(*ei, g)]];
            boost::put(weights, *ei, ALPHA_E * hypot(v1.x - v2.x, v1.y - v2.y));
        }
    }
    PerfectMatching pm(num_odd, (num_odd - 1) * num_odd / 2); // TODO: what if this overflows?
    std::cout << "calculating odd vert distances" << std::endl;
    for (int i = 0; i < num_odd - 1; i++) {
        if (i % 100 == 0) {
            std::cout << "distance " << i << std::endl;
        }
        std::vector<double> distances(boost::num_vertices(g));
        boost::dijkstra_shortest_paths(g, odd_verts[i],
                                       boost::distance_map(boost::make_iterator_property_map(
                                           distances.begin(), boost::get(boost::vertex_index, g))));
        for (int j = i + 1; j < num_odd; j++) {
            // blossom5 needs integer weights for optimality
            int weight = (int)(MATCHING_DISTANCE_SCALE * distances[odd_verts[j]]);
            // std::cout << odd_verts[i] << " <-> " << odd_verts[j] << ": " << weight << std::endl;
            pm.AddEdge(i, j, weight);
        }
    }
    std::cout << "start matching" << std::endl;
    pm.Solve();
    for (int i = 0; i < num_odd; i++) {
        int match = pm.GetMatch(i);
        // std::cout << i << " <-> " << match << std::endl;
        if (i < match) {
            line(drawing, positions[odd_verts[i]], positions[odd_verts[match]],
                 CV_RGB(255, 255, 0));
        }
    }

    // TODO: convert to line graph and compute shortest paths (add auxiliary start/end vertices with 0-weight edges connecting to clique?)
    typedef boost::property_map<Graph, boost::edge_index_t>::type EdgeIndexMap;
    EdgeIndexMap edge_index = boost::get(boost::edge_index, g);
    LineGraph line_graph(boost::num_edges(g));
    for (boost::tie(vp, vp_end) = boost::vertices(g); vp != vp_end; ++vp) {
        if (index[*vp] % 100 == 0) {
            std::cout << "converting line graph " << index[*vp] << std::endl;
        }
        typename boost::graph_traits<Graph>::out_edge_iterator out_i;
        typename boost::graph_traits<Graph>::out_edge_iterator out_j; // TODO: is this safe?
        typename boost::graph_traits<Graph>::out_edge_iterator out_end;
        for (boost::tie(out_i, out_end) = boost::out_edges(*vp, g); out_i != out_end; ++out_i) {
            for (out_j = out_i + 1; out_j != out_end; out_j++) {
                Vertex v1 = (source(*out_i, g) == *vp) ? target(*out_i, g) : source(*out_i, g);
                Vertex v2 = (source(*out_j, g) == *vp) ? target(*out_j, g) : source(*out_j, g);
                Point p1 = positions[index[v1]];
                Point p2 = positions[index[v2]];
                Point p = positions[index[*vp]];
                double e1_length = hypot(p1.x - p.x, p1.y - p.y);
                double e2_length = hypot(p2.x - p.x, p2.y - p.y);
                // TODO: use an approximation instead?
                double curvature =
                    pow(acos(((p1.x - p.x) * (p2.x - p.x) + (p1.y - p.y) * (p2.y - p.y)) /
                             (e1_length * e2_length)),
                        2) /
                    min(e1_length, e2_length);
                double weight = (boost::get(weights, *out_i) + boost::get(weights, *out_j)) / 2 +
                                GAMMA * curvature;
                boost::add_edge(edge_index[*out_i], edge_index[*out_j], weight, line_graph);
            }
        }
    }
    std::cout << "done with line graph: # verts: " << boost::num_vertices(line_graph)
              << " # edges: " << boost::num_edges(line_graph) << std::endl;

    // TODO: eulerize graph

    // TODO: turn graph into path (Hierholzer's algorithm?): maybe with minimum curvature?

    // TODO: output path: gcode

    for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ei++) {
        if (boost::get(necessity, *ei) == required) {
            line(drawing, positions[index[source(*ei, g)]],
            positions[index[target(*ei, g)]], CV_RGB(0, 255, 0));
        }
    }

    // display
    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", drawing);
    waitKey(0);
    return 0;
}
