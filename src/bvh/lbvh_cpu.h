#pragma once

#include "bvh_utils.h"
#include "aabb.h"
#include "hittable.h"
#include "hittable_list.h"
#include "types.h"

struct Range { int start, end; };

struct lbvh_karras_node {
    AABB bbox;
    int left;
    int right;
    int parent;
    int primitive_id;
    int atomic_counter;

    lbvh_karras_node()
        : left(-1), right(-1), parent(-1), primitive_id(-1), atomic_counter(0) {}

    lbvh_karras_node(const AABB& bbox, int prim_id)
        : bbox(bbox), left(-1), right(-1), parent(-1), primitive_id(prim_id), atomic_counter(0) {}

    lbvh_karras_node(int l, int r)
        : left(l), right(r), parent(-1), primitive_id(-1), atomic_counter(0) {}

    bool is_leaf() const { return primitive_id >= 0; }
};

class LBVHCpu {
  public:
    LBVHCpu(std::vector<shared_ptr<hittable>>& objects);

    bool hit(const ray& r, interval ray_t, hit_record& rec) const;
    

    void increment_ray_count() const;
    double get_construction_time() const;
    double get_morton_time() const;
    double get_sort_time() const;
    double get_hierarchy_time() const;
    double get_bbox_time() const;
    int get_tree_depth() const;
    const std::vector<lbvh_karras_node>& get_nodes() const;
    const std::vector<shared_ptr<hittable>>& get_objects() const;
    int get_primitive_count() const;

  private:
    std::vector<shared_ptr<hittable>>& objects;
    int N = 0;

    std::vector<lbvh_karras_node> nodes;
    std::vector<AABB> primitive_bboxes;

    AABB temp_scene_bbox;

    std::vector<morton_primitive> morton_list;
    uint32_t k = 10;
    uint32_t cell_count = 0;


    double construction_time_ms = 0.0;
    double morton_time_ms = 0.0;
    double sort_time_ms = 0.0;
    double hierarchy_time_ms = 0.0;
    double bbox_time_ms = 0.0;

    inline int clz(uint32_t x) const;

    
    inline int delta(int i, int j) const;
    
    inline Range find_range(int idx) const;

    inline int find_split(int start, int end) const;

    void create_hierarchy_for_node(int node_idx);

    void build_bboxes_from_leaf(int leaf_idx);

    bool hit_recursive(const ray& r, interval ray_t, hit_record& rec, int node_idx) const;

    uint32_t compute_morton(point3 barycenter);

    void radix_sort();

    int compute_depth(int node_idx) const;
};

