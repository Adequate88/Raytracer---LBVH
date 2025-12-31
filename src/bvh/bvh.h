#ifndef BVH_H
#define BVH_H
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

#include "../raytracer/aabb_refactored.h"
#include "../raytracer/hittable.h"
#include "../raytracer/hittable_list.h"

#include <algorithm>


// BVH statistics for performance analysis
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
  int leftFirst, primCount;
};

class bvh {
  public:
    bvh(std::vector<shared_ptr<hittable>>& objects) : objects(objects), N(objects.size()){
      // Initialize relevant vectors
      primitive_indices.resize(N);
      for (int i = 0; i < N ; i++) {
        primitive_indices[i] = i;
      }

      pool.resize(N*2 - 1);
      //
      // Edge case checks
      if (N == 0) {
        pool[0].primCount = 0;          
        pool[0].leftFirst = 0;          
        pool[0].bbox = aabb::empty; 
        return;
      }

      if (N == 1) {
        pool[0].primCount = 1;
        pool[0].leftFirst = 0;
        pool[0].bbox = objects[0]->bounding_box();
        return;
      } 

      // Set head node values
      pool[0].leftFirst = 1;
      pool[0].primCount = 0;
      
      // Create head node bbox
      create_bbox(0, 0, N);

      // Begin recursion
      subdivide(0, 0, N);
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec, int node_idx = 0) const {

      const bvh_node &node = pool[node_idx];

      // Track node visit
      if (enable_stats) stats.node_visits++;

      if (!node.bbox.hit(r, ray_t))
          return false;

      if (node.primCount > 0) {
        bool hit_anything = false;
        for (int i = 0; i < node.primCount; i++) {
          // Track primitive intersection test
          if (enable_stats) stats.primitive_tests++;

          if (objects[primitive_indices[node.leftFirst + i]]->hit(r, ray_t, rec)) {
            ray_t.max = rec.t;
            hit_anything = true;
          }
        }
        return hit_anything;
      }

      bool hit_left = hit(r, ray_t, rec, node.leftFirst);
      bool hit_right = hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec, node.leftFirst + 1);
      return hit_left || hit_right;
    }

    // Statistics methods
    void enable_statistics() const { enable_stats = true; stats.reset(); }
    void disable_statistics() const { enable_stats = false; }
    const bvh_stats& get_stats() const { return stats; }
    void reset_stats() const { stats.reset(); }
    void increment_ray_count() const { if (enable_stats) stats.rays_traced++; }

    // BVH tree visualization
    void print_tree(int max_depth = 10) const {
        std::clog << "\n=== BVH Tree Structure ===\n";
        std::clog << "Total nodes: " << nodes_used << "\n";
        std::clog << "Total primitives: " << N << "\n";
        std::clog << "Tree depth limit for display: " << max_depth << "\n";
        std::clog << "\nTree structure:\n";
        print_node(0, 0, max_depth);
        std::clog << "==========================\n";
    }

    int get_tree_depth() const {
        return compute_depth(0);
    }
    
  private:
    int N = 0; // Size of object list
    int nodes_used = 1;

    // Vectors
    std::vector<bvh_node> pool;
    std::vector<int> primitive_indices;
    std::vector<shared_ptr<hittable>>& objects;

    // Statistics tracking (mutable to allow tracking in const hit())
    mutable bvh_stats stats;
    mutable bool enable_stats = false;

    void create_bbox(int node_idx, int objects_start, int objects_end){
      if (objects_start >= objects_end) {
        pool[node_idx].bbox = aabb::empty;
        return;
      }
      // TODO there might be a faster way to do this
      for (int i = objects_start; i < objects_end; i++){
        pool[node_idx].bbox = aabb(pool[node_idx].bbox, objects[primitive_indices[i]]->bounding_box());
      }
    }

    void subdivide(int node_idx, int objects_start, int objects_end){
      int count = objects_end - objects_start;
      bvh_node &node = pool[node_idx];
      
      if (count == 1) {
        node.primCount = 1;
        node.leftFirst = objects_start;
        node.bbox = objects[primitive_indices[objects_start]]->bounding_box();
        return;
      } else if (count < 1) { return; }

      // Get notable values
      int axis = node.bbox.longest_axis();
      float mid_point = (node.bbox.axis_min(axis) + node.bbox.axis_max(axis)) / 2.0;

      // Define while loop counters
      int i = objects_start;
      int j = objects_end - 1;
      
      // Loop until objects are sorted
      while (i <= j) {
        if (objects[primitive_indices[i]]->get_centroid()[axis] < mid_point) {
          i++;
        } else {
          std::swap(primitive_indices[i], primitive_indices[j]);
          j--;
        }
      } // At this point i should equal mid point

      // Degenerate check
      if (i <= objects_start) i = objects_start + 1;
      if (i >= objects_end) i = objects_end - 1;

      // Assigning node values
      node.primCount = 0;
      node.leftFirst = nodes_used;
      nodes_used += 2;

      // Create bbox of 2 child nodes
      create_bbox(node.leftFirst, objects_start, i);
      create_bbox(node.leftFirst + 1, i , objects_end);
      
      // Recursion onto children
      subdivide(node.leftFirst, objects_start, i);
      subdivide(node.leftFirst + 1, i , objects_end);
    }

    // Helper method to print a node recursively
    void print_node(int node_idx, int depth, int max_depth) const {
        if (depth > max_depth) {
            for (int i = 0; i < depth; i++) std::clog << "  ";
            std::clog << "... (max depth reached)\n";
            return;
        }

        const bvh_node& node = pool[node_idx];

        // Print indentation
        for (int i = 0; i < depth; i++) std::clog << "  ";

        std::clog << "Node[" << node_idx << "] ";

        if (node.primCount > 0) {
            // Leaf node
            std::clog << "LEAF: " << node.primCount << " primitive(s)";
            std::clog << " [indices " << node.leftFirst << "-" << (node.leftFirst + node.primCount - 1) << "]";
        } else {
            // Internal node
            std::clog << "INTERNAL: children at [" << node.leftFirst << ", " << (node.leftFirst + 1) << "]";
        }

        // Print bounding box info
        std::clog << " bbox=[(" << node.bbox.min_x << "," << node.bbox.min_y << "," << node.bbox.min_z << ")";
        std::clog << " - (" << node.bbox.max_x << "," << node.bbox.max_y << "," << node.bbox.max_z << ")]";
        std::clog << "\n";

        // Recurse to children if internal node
        if (node.primCount == 0) {
            print_node(node.leftFirst, depth + 1, max_depth);
            print_node(node.leftFirst + 1, depth + 1, max_depth);
        }
    }

    // Helper method to compute tree depth
    int compute_depth(int node_idx) const {
        const bvh_node& node = pool[node_idx];

        if (node.primCount > 0) {
            return 1;  // Leaf node
        }

        int left_depth = compute_depth(node.leftFirst);
        int right_depth = compute_depth(node.leftFirst + 1);

        return 1 + std::max(left_depth, right_depth);
    }
};


#endif
