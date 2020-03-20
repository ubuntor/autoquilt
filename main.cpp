#include "PerfectMatching.h"
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation.h>
#include <CGAL/Triangulation_vertex_base_with_id_2.h>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/astar_search.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <deque>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <queue>
#include <random>
#include <stdio.h>
#include <stdlib.h>

// max radius for bridges
#define MAX_BRIDGE_RADIUS 8
// required edge weight
#define ALPHA_E 1
// optional edge weight
#define ALPHA_O 2

// TODO: tweak these
// alignment weight in optional edge weight
#define BETA 5
// curvature weight in matching path line graph edge weight
#define GAMMA 2
#define MATCHING_DISTANCE_SCALE 1000

// TODO: make this an option
#define MAX_DIMENSION 10

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
typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
typedef boost::graph_traits<Graph>::edge_descriptor Edge;
typedef boost::graph_traits<Graph>::vertex_iterator vertex_iter;
typedef boost::graph_traits<Graph>::edge_iterator edge_iter;
typedef boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;
typedef boost::property_map<Graph, boost::edge_index_t>::type EdgeIndexMap;
typedef boost::property_map<Graph, edge_necessity_tag_t>::type NecessityMap;
typedef boost::property_map<Graph, boost::edge_weight_t>::type WeightMap;
typedef boost::graph_traits<Graph>::adjacency_iterator adj_iter;
typedef boost::graph_traits<Graph>::out_edge_iterator out_edge_iter;

class graph_heuristic : public boost::astar_heuristic<Graph, double> {
  public:
    graph_heuristic(std::vector<cv::Point> p, Vertex goal) : m_positions(p), m_goal(goal){};
    double operator()(Vertex v) {
        // lower bound for distance is euclidean distance, using required edge weights
        return ALPHA_E * hypot(m_positions[v].x - m_positions[m_goal].x,
                               m_positions[v].y - m_positions[m_goal].y);
    }

  private:
    std::vector<cv::Point> m_positions;
    Vertex m_goal;
};
struct found_goal {}; // exception for termination
// visitor that terminates when we find the goal
class astar_goal_visitor : public boost::default_astar_visitor {
  public:
    astar_goal_visitor(Vertex goal) : m_goal(goal) {}
    template <class Graph> void examine_vertex(Vertex u, Graph &g) {
        if (u == m_goal)
            throw found_goal();
    }

  private:
    Vertex m_goal;
};

void flow_map(cv::Mat &src, cv::Mat &dst_x, cv::Mat &dst_y) {
    cv::Mat img, dx, dy, dxx, dyy, dxy;
    // structure tensor
    src.convertTo(img, CV_32F);
    cv::Sobel(img, dx, CV_32F, 1, 0);
    cv::Sobel(img, dy, CV_32F, 0, 1);
    cv::multiply(dx, dx, dxx);
    cv::multiply(dy, dy, dyy);
    cv::multiply(dx, dy, dxy);
    cv::transform(dxx, dxx, cv::Matx13f(1, 1, 1));
    cv::transform(dyy, dyy, cv::Matx13f(1, 1, 1));
    cv::transform(dxy, dxy, cv::Matx13f(1, 1, 1));
    // smooth structure tensor
    cv::GaussianBlur(dxx, dxx, cv::Size(9, 9), 0, 0);
    cv::GaussianBlur(dyy, dyy, cv::Size(9, 9), 0, 0);
    cv::GaussianBlur(dxy, dxy, cv::Size(9, 9), 0, 0);
    // local edge orientation from eigenvector
    cv::Mat tmp1, tmp2, disc, flowx, flowy, tmpx, tmpy;
    tmp1 = dxx - dyy;
    multiply(tmp1, tmp1, tmp1);
    multiply(dxy, dxy, tmp2);
    cv::sqrt(tmp1 + 4.0 * tmp2, disc);
    tmpx = (dxx - dyy - disc) * 0.5;
    tmpy = dxy;
    // normalize
    cv::Mat tmpxx, tmpyy, mag;
    multiply(tmpx, tmpx, tmpxx);
    multiply(tmpy, tmpy, tmpyy);
    cv::sqrt(tmpxx + tmpyy, mag);
    cv::divide(tmpx, mag, dst_x);
    cv::divide(tmpy, mag, dst_y);
}

void find_eulerian_cycle(Graph &g, const NecessityMap &necessity, std::deque<Edge> &cycle) {
    edge_iter ei, ei_end;
    // Hierholzer's algorithm: get eulerian cycle
    EdgeIndexMap edge_index = boost::get(boost::edge_index, g);
    int num_required = 0;
    for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
        if (boost::get(necessity, *ei) == required) {
            boost::put(edge_index, *ei, num_required);
            num_required++;
        }
    }
    std::vector<bool> edge_used(num_required);

    // find a required edge
    for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
        if (boost::get(necessity, *ei) == required) {
            cycle.push_back(*ei);
            num_required--;
            break;
        }
    }
    while (num_required > 0) {
        Edge e = cycle.front();
        out_edge_iter out_i, out_end;
        bool found = false;
        for (boost::tie(out_i, out_end) = boost::out_edges(boost::target(e, g), g);
             out_i != out_end; ++out_i) {
            if (boost::get(necessity, *out_i) == required && !edge_used[edge_index[*out_i]]) {
                edge_used[edge_index[*out_i]] = true;
                found = true;
                cycle.push_front(*out_i);
                num_required--;
                break;
            }
        }
        if (!found) {
            cycle.push_back(e);
            cycle.pop_front();
        }
    }
}

int output_pat_file(const char *filename, double scale, const std::vector<cv::Point> &positions,
                    const IndexMap &index, const std::deque<Edge> &cycle, const Graph &g) {
    FILE *pat_file = fopen(filename, "w");
    if (pat_file == NULL) {
        std::cout << "could not open output file!" << std::endl;
        return 1;
    }
    cv::Point start = positions[index[boost::source(cycle.front(), g)]];
    fprintf(pat_file, "N1G00X%.3fY%.3f\r\n", start.x * scale, MAX_DIMENSION - start.y * scale);
    int line_number = 1;
    for (Edge e : cycle) {
        Vertex v = boost::target(e, g);
        cv::Point p = positions[index[v]];
        fprintf(pat_file, "N%dG01X%.3fY%.3f\r\n", line_number, p.x * scale,
                MAX_DIMENSION - p.y * scale);
        line_number++;
    }
    fprintf(pat_file, "N%dM02\r\n", line_number);
    fclose(pat_file);
    return 0;
}

cv::RNG rng;

void line_integral_convolution(const cv::Mat &flowx, const cv::Mat &flowy, cv::Mat &out) {
    cv::Mat noise = cv::Mat::zeros(flowx.size(), CV_8UC1);
    rng.fill(noise, cv::RNG::UNIFORM, 0, 256, true);
    cv::GaussianBlur(noise, noise, cv::Size(3, 3), 0, 0);
    noise.convertTo(noise, CV_32F);
    out = cv::Mat::zeros(noise.size(), CV_32F);
    int height = noise.rows;
    int width = noise.cols;
    float epsilon = 0.5;
    int radius = 7;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // evolve forwards
            float sample_x = x;
            float sample_y = y;
            // flowx/flowy only provide orientation, not direction, so we keep track of
            // the previously seen flow vector and follow the closest direction vector
            float prev_flow_x = flowx.at<float>(y, x);
            float prev_flow_y = flowy.at<float>(y, x);
            float total = noise.at<float>(y, x);
            for (int i = 0; i < radius; i++) {
                int cy = (int)sample_y % height;
                if (cy < 0) {
                    cy += height;
                }
                int cx = (int)sample_x % width;
                if (cx < 0) {
                    cx += width;
                }
                float cur_flow_x = flowx.at<float>(cy, cx);
                float cur_flow_y = flowy.at<float>(cy, cx);
                if (prev_flow_x * cur_flow_x + prev_flow_y * cur_flow_y > 0) {
                    sample_x += epsilon * cur_flow_x;
                    sample_y += epsilon * cur_flow_y;
                } else {
                    sample_x -= epsilon * cur_flow_x;
                    sample_y -= epsilon * cur_flow_y;
                }
                cy = (int)sample_y % height;
                if (cy < 0) {
                    cy += height;
                }
                cx = (int)sample_x % width;
                if (cx < 0) {
                    cx += width;
                }
                prev_flow_x = cur_flow_x;
                prev_flow_y = cur_flow_y;
                total += noise.at<float>(cy, cx);
            }
            // evolve backwards
            sample_x = x;
            sample_y = y;
            prev_flow_x = -flowx.at<float>(y, x);
            prev_flow_y = -flowy.at<float>(y, x);
            for (int i = 0; i < radius; i++) {
                int cy = (int)sample_y % height;
                if (cy < 0) {
                    cy += height;
                }
                int cx = (int)sample_x % width;
                if (cx < 0) {
                    cx += width;
                }
                float cur_flow_x = flowx.at<float>(cy, cx);
                float cur_flow_y = flowy.at<float>(cy, cx);
                if (prev_flow_x * cur_flow_x + prev_flow_y * cur_flow_y > 0) {
                    sample_x += epsilon * cur_flow_x;
                    sample_y += epsilon * cur_flow_y;
                } else {
                    sample_x -= epsilon * cur_flow_x;
                    sample_y -= epsilon * cur_flow_y;
                }
                cy = (int)sample_y % height;
                if (cy < 0) {
                    cy += height;
                }
                cx = (int)sample_x % width;
                if (cx < 0) {
                    cx += width;
                }
                prev_flow_x = cur_flow_x;
                prev_flow_y = cur_flow_y;
                total += noise.at<float>(cy, cx);
            }
            // average over 21 samples
            out.at<float>(y, x) = total / (2 * radius + 1);
        }
    }
    out.convertTo(out, CV_8UC1);
}

int main(int argc, char **argv) {
    // TODO: split this up
    cv::Mat image, image_g1, image_g2, image_gdiff, image_threshold, flowx, flowy;
    std::vector<std::vector<cv::Point>> contours;
    CDT cdt;

    // TODO: actual arg parsing
    if (argc <= 2) {
        printf("Usage: %s [image] [output file]\n", argv[0]);
        return 1;
    }
    image = cv::imread(argv[1]);
    if (!image.data) {
        printf("no image data\n");
        return 1;
    }

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", image);
    cv::waitKey(0);

    // required edges
    // TODO: actual FDoG
    flow_map(image, flowx, flowy);

    cv::Mat tmp_lic, tmp_phase;
    line_integral_convolution(flowx, flowy, tmp_lic);
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", tmp_lic);
    cv::waitKey(0);
    cv::phase(flowx, flowy, tmp_phase, true);
    cv::subtract(tmp_phase, 180.0f, tmp_phase, (tmp_phase > 180.0f));
    tmp_phase = tmp_phase * (255.0 / 180.0);
    tmp_phase.convertTo(tmp_phase, CV_8UC1);
    cv::applyColorMap(tmp_phase, tmp_phase, cv::COLORMAP_HSV);
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", tmp_phase);
    cv::waitKey(0);

    // cv::cvtColor(image, image, cv::COLOR_BGR2Lab);
    cv::GaussianBlur(image, image_g1, cv::Size(3, 3), 0, 0);
    cv::GaussianBlur(image, image_g2, cv::Size(5, 5), 0, 0);
    image_gdiff = image_g1 - image_g2;
    // cv::cvtColor(image_gdiff, image_gdiff, cv::COLOR_Lab2BGR);
    cv::cvtColor(image_gdiff, image_gdiff, cv::COLOR_BGR2GRAY);
    cv::threshold(image_gdiff, image_threshold, 127, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", image_threshold);
    cv::waitKey(0);

    cv::ximgproc::thinning(image_threshold, image_threshold);

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", image_threshold);
    cv::waitKey(0);

    cv::findContours(image_threshold, contours, cv::RETR_LIST,
                     cv::CHAIN_APPROX_TC89_KCOS); // ???
    // optional edges
    std::cout << "Contours found" << std::endl;
    for (std::vector<cv::Point> i : contours) {
        for (int j = 0; j < i.size() - 1; j++) {
            cdt.insert_constraint(CDT::Point(i[j].x, i[j].y),
                                  CDT::Point(i[j + 1].x, i[j + 1].y));
        }
        cdt.insert_constraint(CDT::Point(i[i.size() - 1].x, i[i.size() - 1].y),
                              CDT::Point(i[0].x, i[0].y));
    }
    assert(cdt.is_valid());

    Graph g(cdt.number_of_vertices());
    std::vector<cv::Point> positions(cdt.number_of_vertices());

    int id = 0;
    for (CDT::Finite_vertices_iterator it = cdt.finite_vertices_begin();
         it != cdt.finite_vertices_end(); ++it) {
        // std::cout << it->point() << std::endl;
        it->id() = id;
        positions[id] = cv::Point(it->point().x(), it->point().y());
        id++;
    }

    cv::Mat drawing = cv::Mat::zeros(image_threshold.size(), CV_8UC3); // ???

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

    IndexMap index = boost::get(boost::vertex_index, g);
    int num_vertices = boost::num_vertices(g);

    vertex_iter vp, vp_end;
    std::queue<Vertex> queue;
    std::vector<std::pair<int, int>> bridge_edges;
    std::vector<int> distance(num_vertices);
    for (boost::tie(vp, vp_end) = boost::vertices(g); vp != vp_end; ++vp) {
        if (index[*vp] % 100 == 0) {
            std::cout << "bridging " << index[*vp] << std::endl;
        }
        queue.push(*vp);
        while (!queue.empty()) {
            Vertex v = queue.front();
            int cur_distance = distance[v];
            adj_iter ai, ai_end;
            for (boost::tie(ai, ai_end) = boost::adjacent_vertices(v, g); ai != ai_end; ++ai) {
                if (cur_distance < MAX_BRIDGE_RADIUS && index[*ai] != index[*vp] && distance[index[*ai]] == 0) {
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
    std::cout << "done adding bridge edges: # verts: " << num_vertices
              << " # edges: " << boost::num_edges(g) << std::endl;

    // mst to connect graph

    std::vector<Edge> mst;
    NecessityMap necessity = boost::get(edge_necessity_tag_t(), g);
    WeightMap weights = boost::get(boost::edge_weight_t(), g);
    edge_iter ei, ei_end;
    for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
        if (boost::get(necessity, *ei) == required) {
            // required edges are free
            boost::put(weights, *ei, 0);
        } else {
            cv::Point v1 = positions[index[source(*ei, g)]];
            cv::Point v2 = positions[index[target(*ei, g)]];
            double evecx = v1.x - v2.x;
            double evecy = v1.y - v2.y;
            double length = hypot(evecx, evecy);
            double flowx_v1 = flowx.at<float>((int)v1.y, (int)v1.x);
            double flowy_v1 = flowy.at<float>((int)v1.y, (int)v1.x);
            double flowx_v2 = flowx.at<float>((int)v2.y, (int)v2.x);
            double flowy_v2 = flowy.at<float>((int)v2.y, (int)v2.x);
            // added abs since flowx/flowy only provide orientation, not direction
            double alignment_v1 =
                1 -
                exp(-5 * pow(1 - abs(evecx * flowx_v1 / length + evecy * flowy_v1 / length), 2));
            double alignment_v2 =
                1 -
                exp(-5 * pow(1 - abs(evecx * flowx_v2 / length + evecy * flowy_v2 / length), 2));
            double alignment = (alignment_v1 + alignment_v2) * length / 2.0;
            boost::put(weights, *ei, ALPHA_O * (length + BETA * alignment));
        }
    }
    boost::kruskal_minimum_spanning_tree(g, std::back_inserter(mst));
    for (std::vector<Edge>::iterator ei = mst.begin(); ei != mst.end(); ++ei) {
        boost::put(necessity, *ei, required);
    }
    std::cout << "done mst" << std::endl;
    // fill in required edge weights
    for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ei++) {
        if (boost::get(necessity, *ei) == required) {
            cv::Point v1 = positions[index[source(*ei, g)]];
            cv::Point v2 = positions[index[target(*ei, g)]];
            boost::put(weights, *ei, ALPHA_E * hypot(v1.x - v2.x, v1.y - v2.y));
        }
    }

    std::vector<Vertex> odd_verts;
    for (boost::tie(vp, vp_end) = boost::vertices(g); vp != vp_end; ++vp) {
        int required_degree = 0;
        out_edge_iter out_i, out_end;
        for (boost::tie(out_i, out_end) = boost::out_edges(*vp, g); out_i != out_end; ++out_i) {
            if (boost::get(necessity, *out_i) == required) {
                required_degree++;
            }
        }
        if (required_degree % 2 == 1) {
            odd_verts.push_back(*vp);
        }
    }
    int num_odd = odd_verts.size();
    std::cout << "# odd: " << num_odd << std::endl;

    // minimum weight matching for odd-degree vertices
    // TODO: better landmark selection: use centrality instead of random

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, num_vertices - 1);

    int num_landmarks = std::min(100, num_vertices); // TODO: tweak this
    std::cout << "calculating odd vert distances" << std::endl;
    std::vector<Vertex> landmarks;
    std::vector<double> landmark_distances(num_landmarks * num_vertices);
    std::vector<double> distances(num_vertices);
    for (int i = 0; i < num_landmarks; i++) {
        landmarks.push_back(uni(rng));
    }

    for (int i = 0; i < num_landmarks; i++) {
        boost::dijkstra_shortest_paths(g, landmarks[i],
                                       boost::distance_map(boost::make_iterator_property_map(
                                           distances.begin(), boost::get(boost::vertex_index, g))));
        for (boost::tie(vp, vp_end) = boost::vertices(g); vp != vp_end; ++vp) {
            landmark_distances[index[*vp] * num_landmarks + i] = distances[*vp];
        }
    }

    PerfectMatching pm(num_odd, (num_odd - 1) * num_odd / 2); // TODO: what if this overflows?
    for (int i = 0; i < num_odd - 1; i++) {
        if (i % 100 == 0) {
            std::cout << "distance " << i << std::endl;
        }
        for (int j = i + 1; j < num_odd; j++) {
            // use landmark distance upper bound
            double mindist = std::numeric_limits<double>::infinity();
            for (int k = 0; k < num_landmarks; k++) {
                mindist = std::min(mindist,
                                   landmark_distances[index[odd_verts[i]] * num_landmarks + k] +
                                       landmark_distances[index[odd_verts[j]] * num_landmarks + k]);
            }
            // blossom5 needs integer weights for optimality
            int weight = (int)(MATCHING_DISTANCE_SCALE * mindist);
            // std::cout << odd_verts[i] << " <-> " << odd_verts[j] << ": " << weight << std::endl;
            pm.AddEdge(i, j, weight);
        }
    }
    std::cout << "start matching" << std::endl;
    pm.Solve();

    for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
        if (boost::get(necessity, *ei) == required) {
            cv::line(drawing, positions[index[source(*ei, g)]], positions[index[target(*ei, g)]],
                     CV_RGB(0, 255, 0));
        }
    }

    // TODO: use implicit line graph instead so we can add a curvature term
    std::vector<std::pair<int, int>> matching_edges;
    std::vector<Vertex> predecessors(boost::num_vertices(g));
    for (int i = 0; i < num_odd; i++) {
        if (i % 100 == 0) {
            std::cout << "matching path " << i << std::endl;
        }
        int match = pm.GetMatch(i);
        // don't double count matches
        if (i < match) {
            Vertex start = odd_verts[i];
            Vertex goal = odd_verts[match];
            try {
                boost::astar_search(
                    g, start, graph_heuristic(positions, goal),
                    boost::predecessor_map(&predecessors[0]).visitor(astar_goal_visitor(goal)));
            } catch (found_goal gf) {
                Vertex current = goal;
                while (current != start) {
                    Vertex predecessor = predecessors[current];
                    matching_edges.push_back(
                        std::pair<int, int>(index[current], index[predecessor]));
                    current = predecessor;
                }
            }
        }
    }
    for (std::pair<int,int> matching_edge : matching_edges) {
        boost::add_edge(matching_edge.first, matching_edge.second, required, g);
    }

    std::deque<Edge> cycle;
    find_eulerian_cycle(g, necessity, cycle);
    std::cout << "done converting to cycle" << std::endl;

    double scale = MAX_DIMENSION / ((double)std::max(image.cols, image.rows));
    if (output_pat_file(argv[2], scale, positions, index, cycle, g)) {
        return 1;
    }

    cv::Mat drawing_matching = cv::Mat::zeros(image_threshold.size(), CV_8UC3); // ???
    for (std::pair<int,int> matching_edge : matching_edges) {
        cv::line(drawing_matching, positions[matching_edge.first], positions[matching_edge.second],
                 CV_RGB(255, 0, 0));
    }

    drawing = drawing + drawing_matching;
    // display
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", drawing);
    cv::waitKey(0);
    return 0;
}
