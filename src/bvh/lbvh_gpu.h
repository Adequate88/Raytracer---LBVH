#ifndef LBVH_GPU_H
#define LBVH_GPU_H
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
#include "../raytracer/aabb.h"
#include "../raytracer/hittable.h"
#include "../raytracer/hittable_list.h"

#include <algorithm>
#include <memory>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <queue>

struct int2 {
  int x, y;

  int2() : x(0), y(0) {}  // Default constructor
  int2(int x, int y) : x(x), y(y) {}
};

struct lbvh_node {
    aabb bbox;
    lbvh_node* left;
    lbvh_node* right;
    lbvh_node* parent;
    int primitive_id;  // -1 for internal nodes, >=0 for leaf nodes

    // Default constructor
    lbvh_node()
        : left(nullptr), right(nullptr), parent(nullptr), primitive_id(-1) {}

    // Leaf constructor
    lbvh_node(const aabb& bbox, int prim_id)
        : bbox(bbox), left(nullptr), right(nullptr), parent(nullptr), primitive_id(prim_id) {}

    // Internal constructor
    lbvh_node(lbvh_node* l, lbvh_node* r)
        : left(l), right(r), parent(nullptr), primitive_id(-1) {
        bbox = aabb(left->bbox, right->bbox);
    }

    bool is_leaf() const { return primitive_id >= 0; }
};

class lbvh_gpu {
  public:
    lbvh_gpu(std::vector<shared_ptr<hittable>>& objects)
        : objects(objects), N(objects.size()) {

        // Pre-allocate arrays
        internal_nodes.resize(N - 1);  // N-1 internal nodes
        leaf_nodes.resize(N);           // N leaf nodes

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

        // Initialize leaf nodes with sorted primitives
        for (int i = 0; i < N; i++) {
            morton_primitive morton_object = morton_list[i];
            leaf_nodes[i].bbox = objects[morton_object.primitive_id]->bounding_box();
            leaf_nodes[i].primitive_id = morton_object.primitive_id;
            leaf_nodes[i].parent = nullptr;  // Will be set during tree construction
        }

        // Build tree recursively
        root = subdivide();

        bottom_up_bbox_build();

        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();
        construction_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }

    // Entry point for ray intersection
    bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        if (!root) return false;
        return hit_recursive(r, ray_t, rec, root);
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
        print_node_recursive(root, 0, max_depth);
        std::clog << "===========================\n";
    }

    int get_tree_depth() const {
        return compute_depth(root);
    }

  private:
    int N = 0; // Size of object list

    // Data members
    lbvh_node* root;                         // Pointer to root (points into internal_nodes)
    std::vector<lbvh_node> internal_nodes;   // N-1 internal nodes
    std::vector<lbvh_node> leaf_nodes;       // N leaf nodes

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

        // Is primitive (leaf node)
        if (node->is_leaf()) {
            if (enable_stats) stats.primitive_tests++;
            return objects[node->primitive_id]->hit(r, ray_t, rec);
        }

        // Recurse children (internal node)
        bool hit_left = hit_recursive(r, ray_t, rec, node->left);
        bool hit_right = hit_recursive(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max),
                                       rec, node->right);
        return hit_left || hit_right;
    }

    lbvh_node* subdivide() {      

      for (int i = 0; i < N-1; i++) {

        int2 range = find_range(i);
        int start = range.x;
        int end = range.y;
        // Get split idx
        int split = find_split(start, end);

        // Assign children
        lbvh_node* left_node;
        if (split == start) { // Is leaf
          left_node = &leaf_nodes[split];
        } else {
          left_node = &internal_nodes[split];
        }

        lbvh_node* right_node;
        if (split + 1 == end) { // Is leaf
          right_node = &leaf_nodes[split + 1];
        } else {
          right_node = &internal_nodes[split + 1];
        }

        internal_nodes[i].left = left_node;
        internal_nodes[i].right = right_node;
        left_node->parent = &internal_nodes[i];
        right_node->parent = &internal_nodes[i];

      }
      return &internal_nodes[0];
    }

    void bottom_up_bbox_build() {
      std::queue<lbvh_node*> q;

      for (const auto& leaf : leaf_nodes) {
        if (leaf.parent != nullptr)
          q.push(leaf.parent);
      }

      while (!q.empty()) {
        lbvh_node* current_node = q.front();
        q.pop();

        // Skip if already processed (bbox has been set to non-empty)
        if (current_node->bbox.x.min <= current_node->bbox.x.max)
          continue;

        lbvh_node* left_child = current_node->left;
        lbvh_node* right_child = current_node->right;

        // Check if both children have valid bboxes before computing parent bbox
        bool left_ready = (left_child->bbox.x.min <= left_child->bbox.x.max);
        bool right_ready = (right_child->bbox.x.min <= right_child->bbox.x.max);

        if (!left_ready || !right_ready) {
          // One or both children not ready yet, push back to queue and try later
          q.push(current_node);
          continue;
        }

        current_node->bbox = aabb(left_child->bbox, right_child->bbox);

        if (current_node->parent != nullptr)
          q.push(current_node->parent);
      }
    }

    int2 find_range(int idx) {
      // TODO implement algorithm from Karras
      // Find direction using neighbors
      
      int dir = (delta(idx, idx + 1) - delta(idx, idx - 1)) > 0 ? 1 : -1; // If positive return 1, otherwise -1


      // Get minimum bound at i - d
      int min_bound = delta(idx, idx - dir);
     
      
      // Get Lmax bound of range
      int l_max = 2; // Initial max bound;
      while (delta(idx, idx + dir*l_max) > min_bound)
        l_max *= 2; 
      

      // Binary search for end range
      int l = 0; // Find distance to end range 

      for (int t = l_max / 2; t >= 1; t /= 2) {
                
        if (delta(idx, idx + (t + l)*dir) > min_bound) {
          l += t;
        }
      }
      
      int final_point = idx + l*dir;
      
      int2 range;
      if ( final_point > idx) {
        range = int2(idx, final_point);
      } else {
        range = int2(final_point, idx);
      }

      return range;
    }

    int delta(int i, int j) {
      if (j < 0 || j >= N) return -1;

      uint32_t code_i = morton_list[i].morton_code;
      uint32_t code_j = morton_list[j].morton_code;

      return __builtin_clz(code_i ^ code_j);
    }
    
    // TODO move this function somewhere smarter in codebase
    int sign(int x) {
      if (x > 0) return 1;
      if (x < 0) return -1;
      return 0;
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
            print_node_recursive(node->left, depth + 1, max_depth);
            print_node_recursive(node->right, depth + 1, max_depth);
        }
    }

    // Helper method to compute tree depth
    int compute_depth(const lbvh_node* node) const {
        if (!node) return 0;
        if (node->is_leaf()) return 1;

        int left_depth = compute_depth(node->left);
        int right_depth = compute_depth(node->right);

        return 1 + std::max(left_depth, right_depth);
    }
};

#endif
