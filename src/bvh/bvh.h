#ifndef BVH_H
#define BVH_H

#include "../raytracer/aabb_refactored.h"
#include "../raytracer/hittable.h"
#include "../raytracer/hittable_list.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>

struct bvh_stats {
    long long node_visits = 0;
    long long primitive_tests = 0;
    long long rays_traced = 0;

    void reset() {
        node_visits = 0;
        primitive_tests = 0;
        rays_traced = 0;
    }

    void print() const {
        std::clog << "\n=== BVH Statistics ===\n";
        std::clog << "  Total rays traced: " << rays_traced << "\n";
        if (rays_traced > 0) {
            std::clog << "  Avg node visits/ray: " << (double)node_visits / rays_traced << "\n";
            std::clog << "  Avg primitive tests/ray: " << (double)primitive_tests / rays_traced << "\n";
        }
        std::clog << "======================\n";
    }
};

struct bvh_node {
    aabb bbox;
    int left;
    int right;
    int parent;
    int primitive_id;
    int atomic_counter;

    bvh_node() : left(-1), right(-1), parent(-1), primitive_id(-1), atomic_counter(0) {}

    bool is_leaf() const { return primitive_id >= 0; }
};

class bvh {
  public:
    bvh(std::vector<shared_ptr<hittable>>& objects) : objects(objects), N(objects.size()) {
        auto start_time = std::chrono::high_resolution_clock::now();

        if (N == 0) return;

        nodes.resize(2 * N - 1);
        build_recursive(0, N, -1);

        auto end_time = std::chrono::high_resolution_clock::now();
        construction_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        return hit_recursive(r, ray_t, rec, 0);
    }

    void enable_statistics() const { enable_stats = true; stats.reset(); }
    void disable_statistics() const { enable_stats = false; }
    const bvh_stats& get_stats() const { return stats; }
    void increment_ray_count() const { if (enable_stats) stats.rays_traced++; }
    double get_construction_time() const { return construction_time_ms; }
    const std::vector<bvh_node>& get_nodes() const { return nodes; }
    const std::vector<shared_ptr<hittable>>& get_objects() const { return objects; }
    int get_primitive_count() const { return N; }

    void export_statistics(const std::string& filename, long long total_rays,
                           double avg_box_tests, double avg_prim_tests) const {
        std::ofstream out(filename);
        if (!out.is_open()) {
            std::cerr << "Failed to open: " << filename << std::endl;
            return;
        }
        out << std::fixed << std::setprecision(3);
        out << "total_construction_ms," << construction_time_ms << "\n";
        out << "total_rays," << total_rays << "\n";
        out << "avg_box_tests_per_ray," << avg_box_tests << "\n";
        out << "avg_prim_tests_per_ray," << avg_prim_tests << "\n";
        out.close();
    }

    int get_tree_depth() const { return compute_depth(0); }

  private:
    std::vector<shared_ptr<hittable>>& objects;
    int N = 0;
    int next_node = 0;
    std::vector<bvh_node> nodes;

    mutable bvh_stats stats;
    mutable bool enable_stats = false;
    double construction_time_ms = 0.0;

    int build_recursive(int start, int end, int parent_idx) {
        int node_idx = next_node++;
        bvh_node& node = nodes[node_idx];
        node.parent = parent_idx;

        int count = end - start;

        node.bbox = aabb::empty;
        for (int i = start; i < end; i++) {
            node.bbox = aabb(node.bbox, objects[i]->bounding_box());
        }

        if (count == 1) {
            node.primitive_id = start;
            node.left = -1;
            node.right = -1;
            return node_idx;
        }

        int axis = node.bbox.longest_axis();
        float mid = (node.bbox.axis_min(axis) + node.bbox.axis_max(axis)) / 2.0f;

        auto mid_iter = std::partition(
            objects.begin() + start, objects.begin() + end,
            [axis, mid](const shared_ptr<hittable>& obj) {
                return obj->get_centroid()[axis] < mid;
            }
        );

        int mid_idx = mid_iter - objects.begin();
        if (mid_idx <= start) mid_idx = start + 1;
        if (mid_idx >= end) mid_idx = end - 1;

        node.primitive_id = -1;
        node.left = build_recursive(start, mid_idx, node_idx);
        node.right = build_recursive(mid_idx, end, node_idx);

        return node_idx;
    }

    bool hit_recursive(const ray& r, interval ray_t, hit_record& rec, int node_idx) const {
        if (node_idx < 0 || node_idx >= (int)nodes.size()) return false;

        const bvh_node& node = nodes[node_idx];

        if (!node.bbox.hit(r, ray_t)) return false;

        if (enable_stats) stats.node_visits++;

        if (node.is_leaf()) {
            if (enable_stats) stats.primitive_tests++;
            return objects[node.primitive_id]->hit(r, ray_t, rec);
        }

        bool hit_left = hit_recursive(r, ray_t, rec, node.left);
        bool hit_right = hit_recursive(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec, node.right);
        return hit_left || hit_right;
    }

    int compute_depth(int node_idx) const {
        if (node_idx < 0) return 0;
        const bvh_node& node = nodes[node_idx];
        if (node.is_leaf()) return 1;
        return 1 + std::max(compute_depth(node.left), compute_depth(node.right));
    }
};

#endif
