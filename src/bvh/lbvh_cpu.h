#ifndef LBVH_CPU_H
#define LBVH_CPU_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "morton.h"
#include "bvh.h"
#include "../raytracer/aabb_refactored.h"
#include "../raytracer/hittable.h"
#include "../raytracer/hittable_list.h"

#include <algorithm>
#include <memory>
#include <cstdint>
#include <chrono>

struct lbvh_node {
    aabb bbox;
    std::unique_ptr<lbvh_node> left;    // Left child (nullptr for leaves)
    std::unique_ptr<lbvh_node> right;   // Right child (nullptr for leaves)
    int primitive_id;                   // -1 for internal, >=0 for leaf

    // Constructor for leaf nodes
    lbvh_node(const aabb& bbox, int prim_id)
        : bbox(bbox), left(nullptr), right(nullptr), primitive_id(prim_id) {}

    // Constructor for internal nodes
    lbvh_node(std::unique_ptr<lbvh_node> l, std::unique_ptr<lbvh_node> r)
        : left(std::move(l)), right(std::move(r)), primitive_id(-1) {
        bbox = aabb(left->bbox, right->bbox);  // Merge children bboxes
    }

    bool is_leaf() const { return primitive_id >= 0; }
};

class lbvh_cpu {
  public:
    lbvh_cpu(std::vector<shared_ptr<hittable>>& objects)
        : objects(objects), N(objects.size()) {

        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();

        // Edge case: empty scene
        if (N == 0) {
            root = nullptr;
            return;
        }

        // LBVH setup
        morton_list.resize(N);
        cell_count = 1 << k;  // 2^10 = 1024

        // Compute scene bounding box
        temp_scene_bbox = aabb::empty;
        for (int i = 0; i < N; i++) {
            temp_scene_bbox = aabb(temp_scene_bbox, objects[i]->bounding_box());
        }

        // Compute Morton codes for all primitives
        for (int i = 0; i < N; i++) {
            morton_list[i].morton_code = compute_morton(objects[i]->get_centroid());
            morton_list[i].primitive_id = i;
        }

        // Sort by Morton code
        radix_sort();

        // Build tree recursively
        root = subdivide(0, N);

        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();
        construction_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }

    // Entry point for ray intersection
    bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        if (!root) return false;
        return hit_recursive(r, ray_t, rec, root.get());
    }

    // Statistics methods
    void enable_statistics() const { enable_stats = true; stats.reset(); }
    void disable_statistics() const { enable_stats = false; }
    const bvh_stats& get_stats() const { return stats; }
    void increment_ray_count() const { if (enable_stats) stats.rays_traced++; }

    // Timing methods
    double get_construction_time() const { return construction_time_ms; }

    // Tree visualization
    void print_tree(int max_depth = 10) const {
        std::clog << "\n=== LBVH Tree Structure ===\n";
        std::clog << "Total primitives: " << N << "\n";
        std::clog << "Tree depth limit for display: " << max_depth << "\n";
        std::clog << "\nTree structure:\n";
        print_node_recursive(root.get(), 0, max_depth);
        std::clog << "===========================\n";
    }

    int get_tree_depth() const {
        return compute_depth(root.get());
    }

  private:
    int N = 0; // Size of object list

    // Data members
    std::unique_ptr<lbvh_node> root;
    std::vector<shared_ptr<hittable>>& objects;
    aabb temp_scene_bbox;  // Temporary storage for Morton code computation

    // LBVH Construction Variables
    std::vector<morton_primitive> morton_list;
    uint32_t k = 10; // Default at 10 for 30-bit Morton Codes (used for 2^k X 2^k X 2^k lattice)
    uint32_t cell_count = 0;

    // Statistics tracking (mutable to allow tracking in const hit())
    mutable bvh_stats stats;
    mutable bool enable_stats = false;

    // Construction time tracking
    double construction_time_ms = 0.0;

    // Private recursive hit function
    bool hit_recursive(const ray& r, interval ray_t, hit_record& rec, const lbvh_node* node) const {
        if (!node) return false;

        // Track node visit
        if (enable_stats) stats.node_visits++;

        // Test bounding box
        if (!node->bbox.hit(r, ray_t))
            return false;

        // Is primitive
        if (node->is_leaf()) {
            if (enable_stats) stats.primitive_tests++;
            return objects[node->primitive_id]->hit(r, ray_t, rec);
        }

        // Recurse children
        bool hit_left = hit_recursive(r, ray_t, rec, node->left.get());
        bool hit_right = hit_recursive(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max),
                                       rec, node->right.get());
        return hit_left || hit_right;
    }

    std::unique_ptr<lbvh_node> subdivide(int start, int end) {
        int count = end - start;

        // Check for leaf node
        if (count == 1) {
            int prim_id = morton_list[start].primitive_id;
            aabb prim_bbox = objects[prim_id]->bounding_box();
            return std::make_unique<lbvh_node>(prim_bbox, prim_id);
        }

        if (count < 1) return nullptr;  // Just incase

        // Find Split
        int split_idx = find_split(start, end - 1);

        // Recursively build children
        auto left_child = subdivide(start, split_idx + 1);
        auto right_child = subdivide(split_idx + 1, end);

        // Create internal node 
        return std::make_unique<lbvh_node>(std::move(left_child), std::move(right_child));
    }

    int find_split(int start, int end) {

      uint32_t start_morton = morton_list[start].morton_code;
      uint32_t end_morton = morton_list[end].morton_code;

      if (start_morton == end_morton)
        return (start + end) >> 1;  

      int common_bits = __builtin_clz(start_morton ^ end_morton);

      int step = (end - start);
      int current_split = start;

      do {

        step = (step + 1) >> 1;

        int new_split = current_split + step;

        if (new_split < end) {

          uint32_t split_morton = morton_list[new_split].morton_code;
          int split_msd = __builtin_clz(start_morton ^ split_morton);

          if (split_msd > common_bits)
            current_split = new_split; 
        }
      } while (step > 1);

      return current_split;
    }


    uint32_t compute_morton(point3 barycenter) {
      // Compute cell of barycenter at each axis
      float x_cell = cell_count * (barycenter[0] - temp_scene_bbox.x.min) / temp_scene_bbox.x.size();
      float y_cell = cell_count * (barycenter[1] - temp_scene_bbox.y.min) / temp_scene_bbox.y.size();
      float z_cell = cell_count * (barycenter[2] - temp_scene_bbox.z.min) / temp_scene_bbox.z.size();

      uint32_t x = std::min(static_cast<uint32_t>(x_cell), cell_count - 1);
      uint32_t y = std::min(static_cast<uint32_t>(y_cell), cell_count - 1);
      uint32_t z = std::min(static_cast<uint32_t>(z_cell), cell_count - 1);

      uint32_t morton_code = 0;
      for (int i = 0; i < k; i++){
        morton_code |= ( (x >> i) & 1 ) << (3 * i);
        morton_code |= ( (y >> i) & 1 ) << (3 * i + 1);
        morton_code |= ( (z >> i) & 1 ) << (3 * i + 2);
      }

      return morton_code;

    }

    void radix_sort() {
      // Process each bit from LSB to MSB
      for (int bit = 0; bit < 30; bit++) {
          std::vector<morton_primitive> zero_bucket;
          std::vector<morton_primitive> one_bucket;

          for (const auto& prim : morton_list) {
              if (((prim.morton_code >> bit) & 1) == 0) {
                  zero_bucket.push_back(prim);
              } else {
                  one_bucket.push_back(prim);
              }
          }

          morton_list.clear();
          morton_list.insert(morton_list.end(), zero_bucket.begin(), zero_bucket.end());
          morton_list.insert(morton_list.end(), one_bucket.begin(), one_bucket.end());
      }
    }

    // Helper method to print tree structure recursively
    void print_node_recursive(const lbvh_node* node, int depth, int max_depth) const {
        if (!node || depth > max_depth) return;

        // Indentation
        for (int i = 0; i < depth; i++) std::clog << "  ";

        // Node info
        if (node->is_leaf()) {
            std::clog << "LEAF [prim_id=" << node->primitive_id << "]";
        } else {
            std::clog << "INTERNAL";
        }

        // Print bounding box info
        std::clog << " bbox=[(" << node->bbox.x.min << "," << node->bbox.y.min << "," << node->bbox.z.min << ")";
        std::clog << " - (" << node->bbox.x.max << "," << node->bbox.y.max << "," << node->bbox.z.max << ")]";
        std::clog << "\n";

        // Recurse to children if internal node
        if (!node->is_leaf()) {
            print_node_recursive(node->left.get(), depth + 1, max_depth);
            print_node_recursive(node->right.get(), depth + 1, max_depth);
        }
    }

    // Helper method to compute tree depth
    int compute_depth(const lbvh_node* node) const {
        if (!node) return 0;
        if (node->is_leaf()) return 1;

        int left_depth = compute_depth(node->left.get());
        int right_depth = compute_depth(node->right.get());

        return 1 + std::max(left_depth, right_depth);
    }

};


#endif
