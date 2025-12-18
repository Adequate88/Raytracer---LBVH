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
#include "../raytracer/aabb.h"
#include "../raytracer/hittable.h"
#include "../raytracer/hittable_list.h"

#include <algorithm>

class lbvh_cpu {
  public:
    lbvh_cpu(std::vector<shared_ptr<hittable>>& objects) : objects(objects), N(objects.size()){
      
      // General BVH requirements
      primitive_indices.resize(N);
      for (int i = 0; i < N ; i++) {
        primitive_indices[i] = i;
      }

      pool.resize(N*2 - 1);

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
      
      // LBVH Additions
      morton_list.resize(N);
      cell_count = 1 << k;

      // Using enclosing bbox, create morton code for all primitives
      // TODO Maybe iterating through primitives twice back to back is not the most efficient
      for ( int i = 0 ; i < N ; i++) {
        morton_list[i].morton_code = compute_morton(objects[i]->get_centroid());
        morton_list[i].primitive_id = i;
      }

      radix_sort();

      create_sort_split_lists();
      
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

  private:
    int N = 0; // Size of object list
    int nodes_used = 1;

    // Vectors
    std::vector<bvh_node> pool;
    std::vector<int> primitive_indices;
    std::vector<shared_ptr<hittable>>& objects;
    
    // LBVH Construction Variables
    std::vector<morton_primitive> morton_list;
    int k = 10; // Default at 10 for 30-bit Morton Codes (used for 2^k X 2^k X 2^k lattice)
    int cell_count = 0;
    std::vector<std::pair<int, int>> split_list;

    void create_bbox(int node_idx, int objects_start, int objects_end, int depth){
      if (objects_start >= objects_end) {
        pool[node_idx].bbox = aabb::empty;
        return;
      }
      //
      // TODO there might be a faster way to do this
      for (int i = objects_start; i < objects_end; i++){
        pool[node_idx].bbox = aabb(pool[node_idx].bbox, objects[primitive_indices[i]]->bounding_box());
      }
    }

    void subdivide(int node_idx, int start_idx, int end_idx){
      bvh_node &node = pool[node_idx];

    }

    uint32_t compute_morton(point3 barycenter) {
      // Compute cell of barycenter at each axis
      float x_cell = cell_count * (barycenter[0] - pool[0].bbox.x.min) / pool[0].bbox.x.size();
      float y_cell = cell_count * (barycenter[1] - pool[0].bbox.y.min) / pool[0].bbox.y.size();
      float z_cell = cell_count * (barycenter[2] - pool[0].bbox.z.min) / pool[0].bbox.z.size();

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

    void create_sort_split_lists() {
      for (int i = 0; i < morton_list.size() - 1; i++) {
        uint32_t morton_code_diffs = morton_list[i].morton_code ^ morton_list[i+1].morton_code;
        if (morton_code_diffs != 0) {
          // Find the most significant differing bit
          int first_diff_pos = 31 - __builtin_clz(morton_code_diffs);
          split_list.push_back({i, level});
        }
      }

      int max_level = 3*k;

      // Create buckets for each level
      std::vector<std::vector<std::pair<int, int>>> buckets(max_level);

      // Distribute splits into buckets
      for (const auto& split : split_list) {
        buckets[split.second].push_back(split);
      }

      split_list.clear();
      for (int level = max_level - 1; level >= 0; level--) {
        split_list.insert(split_list.end(),
        buckets[level].begin(),
        buckets[level].end());
      }

  }
    
};


#endif
