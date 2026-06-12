#include "lbvh_cpu.h"
#include "metrics_macros.h"

// MSVC version of __builtin_clz
#ifdef _MSC_VER
#include <intrin.h>
static inline int __builtin_clz(unsigned int x) {
    unsigned long index;
    _BitScanReverse(&index, x);
    return 31 - (int)index;
}
#endif

LBVHCpu::LBVHCpu(std::vector<shared_ptr<hittable>>& objects) : objects(objects), N(objects.size()) 
{
  init();   
}

void LBVHCpu::init()
{
    nodes.resize(2*N - 1);
    if (N == 0) return;

    METRIC_START_TIME("BVH_CONSTRUCTION");

    morton_list.resize(N);
    cell_count = 1 << k;

    init_scene_bbox();

    METRIC_START_TIME("MORTON_TIME");
    compute_mortons();
    METRIC_END_TIME("MORTON_TIME");

    METRIC_START_TIME("RADIX_SORT"); 
    radix_sort();
    METRIC_END_TIME("RADIX_SORT");

    METRIC_START_TIME("HIERARCHY");
    build_hierarchy();
    METRIC_END_TIME("HIERARCHY");

    METRIC_START_TIME("BUILD_BBOX");
    build_bboxes();
    METRIC_END_TIME("BUILD_BBOX");

    METRIC_END_TIME("BVH_CONSTRUCTION");
}

void LBVHCpu::init_scene_bbox()
{
    temp_scene_bbox = AABB::empty;
    for (int i = 0; i < N; i++) {
        temp_scene_bbox = AABB(temp_scene_bbox, objects[i]->bounding_box());
        primitive_bboxes.push_back(objects[i]->bounding_box());
    }
}

void LBVHCpu::compute_mortons()
{
    for (int i = 0; i < N; i++) {
        morton_list[i].morton_code = compute_morton(objects[i]->get_centroid());
        morton_list[i].primitive_id = i;
    }
}

void LBVHCpu::build_hierarchy()
{
  for (int i = 0; i < N; i++) {
        int leaf_idx = N - 1 + i;
        int prim_id = morton_list[i].primitive_id;
        nodes[leaf_idx].bbox = primitive_bboxes[prim_id];
        nodes[leaf_idx].primitive_id = prim_id;
        nodes[leaf_idx].atomic_counter = 0;
        nodes[leaf_idx].parent = -1;
        nodes[leaf_idx].left = -1;
        nodes[leaf_idx].right = -1;
    }
    for (int node_idx = 0; node_idx < N - 1; node_idx++) {
        create_hierarchy_for_node(node_idx);
    }
}

void LBVHCpu::build_bboxes()
{
    for (int leaf_idx = 0; leaf_idx < N; leaf_idx++) {
        build_bboxes_from_leaf(leaf_idx);
    }
}

bool LBVHCpu::hit(const ray& r, interval ray_t, hit_record& rec) const
{
  return hit_recursive(r, ray_t, rec, 0);
}

int LBVHCpu::get_tree_depth() const { return compute_depth(0); }
const std::vector<lbvh_karras_node>& LBVHCpu::get_nodes() const { return nodes; }
const std::vector<shared_ptr<hittable>>& LBVHCpu::get_objects() const { return objects; }
int LBVHCpu::get_primitive_count() const { return N; }

inline int LBVHCpu::clz(uint32_t x) const {
        if (x == 0) return 32;
        return __builtin_clz(x);
    }

inline int LBVHCpu::delta(int i, int j) const {
    if (j < 0 || j >= N) return -1;

    uint32_t code_i = morton_list[i].morton_code;
    uint32_t code_j = morton_list[j].morton_code;

    if (code_i == code_j) {
        return 32 + clz(i ^ j);
    }

    return clz(code_i ^ code_j);
}

inline Range LBVHCpu::find_range(int idx) const {
    int dir = (delta(idx, idx + 1) - delta(idx, idx - 1)) > 0 ? 1 : -1;
    int min_bound = delta(idx, idx - dir);

    int l_max = 2;
    while (delta(idx, idx + dir * l_max) > min_bound)
        l_max *= 2;

    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2) {
        if (delta(idx, idx + (t + l) * dir) > min_bound) {
            l += t;
        }
    }

    int final_point = idx + l * dir;

    Range range;
    if (final_point > idx) {
        range.start = idx;
        range.end = final_point;
    } else {
        range.start = final_point;
        range.end = idx;
    }

    return range;
}

inline int LBVHCpu::find_split(int start, int end) const {
    uint32_t start_morton = morton_list[start].morton_code;
    uint32_t end_morton = morton_list[end].morton_code;
    bool use_tiebreaker = (start_morton == end_morton);
    int common_bits = use_tiebreaker ? (32 + clz(start ^ end)) : clz(start_morton ^ end_morton);

    int step = (end - start);
    int current_split = start;

    do {
        step = (step + 1) >> 1;
        int new_split = current_split + step;

        if (new_split < end) {
            int split_msd;
            if (use_tiebreaker) {
                split_msd = 32 + clz(start ^ new_split);
            } else {
                uint32_t split_morton = morton_list[new_split].morton_code;
                split_msd = clz(start_morton ^ split_morton);
            }

            if (split_msd > common_bits)
                current_split = new_split;
        }
    } while (step > 1);

    return current_split;
}

void LBVHCpu::create_hierarchy_for_node(int node_idx) {
    Range range = find_range(node_idx);
    int start = range.start;
    int end = range.end;
    int split = find_split(start, end);

    int left_node = (split == start) ? split + N - 1 : split;
    int right_node = (split + 1 == end) ? split + N : split + 1;

    nodes[node_idx].left = left_node;
    nodes[node_idx].right = right_node;
    nodes[node_idx].primitive_id = -1;
    nodes[node_idx].bbox = AABB::empty;
    nodes[node_idx].atomic_counter = 0;
    nodes[left_node].parent = node_idx;
    nodes[right_node].parent = node_idx;
}

void LBVHCpu::build_bboxes_from_leaf(int leaf_idx) {
    int current_idx = N - 1 + leaf_idx;
    int parent_idx = nodes[current_idx].parent;

    while (parent_idx != -1) {
        int old_count = nodes[parent_idx].atomic_counter;
        nodes[parent_idx].atomic_counter++;

        if (old_count == 0) break;

        int left_idx = nodes[parent_idx].left;
        int right_idx = nodes[parent_idx].right;
        nodes[parent_idx].bbox = AABB(nodes[left_idx].bbox, nodes[right_idx].bbox);

        current_idx = parent_idx;
        parent_idx = nodes[current_idx].parent;
    }
}

bool LBVHCpu::hit_recursive(const ray& r, interval ray_t, hit_record& rec, int node_idx) const {
    if (node_idx >= 2*N - 1 || node_idx < 0) return false;
    //TODO NodesVisisted++

    const lbvh_karras_node& node = nodes[node_idx];
    if (!node.bbox.hit(r, ray_t)) return false;

    if (node.is_leaf()) {
        // TODO PRIMITIVES VISITED ++
        return objects[node.primitive_id]->hit(r, ray_t, rec);
    }

    bool hit_left = hit_recursive(r, ray_t, rec, node.left);
    bool hit_right = hit_recursive(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max),
                                   rec, node.right);
    return hit_left || hit_right;
}

uint32_t LBVHCpu::compute_morton(point3 barycenter) {
    float bx = static_cast<float>(barycenter[0]);
    float by = static_cast<float>(barycenter[1]);
    float bz = static_cast<float>(barycenter[2]);

    float x_size = temp_scene_bbox.max_x - temp_scene_bbox.min_x;
    float y_size = temp_scene_bbox.max_y - temp_scene_bbox.min_y;
    float z_size = temp_scene_bbox.max_z - temp_scene_bbox.min_z;

    float x_cell = static_cast<float>(cell_count) * (bx - temp_scene_bbox.min_x) / x_size;
    float y_cell = static_cast<float>(cell_count) * (by - temp_scene_bbox.min_y) / y_size;
    float z_cell = static_cast<float>(cell_count) * (bz - temp_scene_bbox.min_z) / z_size;

    uint32_t x = std::min(static_cast<uint32_t>(x_cell), cell_count - 1);
    uint32_t y = std::min(static_cast<uint32_t>(y_cell), cell_count - 1);
    uint32_t z = std::min(static_cast<uint32_t>(z_cell), cell_count - 1);

    uint32_t morton_code = 0;
    for (uint32_t i = 0; i < k; i++) {
        morton_code |= ((x >> i) & 1) << (3 * i);
        morton_code |= ((y >> i) & 1) << (3 * i + 1);
        morton_code |= ((z >> i) & 1) << (3 * i + 2);
    }

    return morton_code;
}

void LBVHCpu::radix_sort() {
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

int LBVHCpu::compute_depth(int node_idx) const {
    if (node_idx < 0) return 0;

    const lbvh_karras_node& node = nodes[node_idx];
    if (node.is_leaf()) return 1;

    int left_depth = compute_depth(node.left);
    int right_depth = compute_depth(node.right);

    return 1 + std::max(left_depth, right_depth);
}
