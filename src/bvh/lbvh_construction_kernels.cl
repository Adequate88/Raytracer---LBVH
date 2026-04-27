inline int delta(int i, int j, __global const morton_primitive* morton_list, int N) {
    if (j < 0 || j >= N) return -1;
    uint code_i = morton_list[i].morton_code;
    uint code_j = morton_list[j].morton_code;
    if (code_i == code_j) return 32 + clz(i ^ j);
    return clz(code_i ^ code_j);
}

inline int2 find_range(int idx, __global const morton_primitive* morton_list, int N) {
    int dir = (delta(idx, idx + 1, morton_list, N) - delta(idx, idx - 1, morton_list, N)) > 0 ? 1 : -1;
    int min_bound = delta(idx, idx - dir, morton_list, N);

    int l_max = 2;
    while (delta(idx, idx + dir*l_max, morton_list, N) > min_bound)
        l_max *= 2;

    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2) {
        if (delta(idx, idx + (t + l)*dir, morton_list, N) > min_bound)
            l += t;
    }

    int final_point = idx + l*dir;
    int2 range;
    range.x = (final_point > idx) ? idx : final_point;
    range.y = (final_point > idx) ? final_point : idx;
    return range;
}

inline int find_split(int start, int end, __global const morton_primitive* morton_list) {
    uint start_morton = morton_list[start].morton_code;
    uint end_morton = morton_list[end].morton_code;
    bool use_tiebreaker = (start_morton == end_morton);
    int common_bits = use_tiebreaker ? (32 + clz(start ^ end)) : clz(start_morton ^ end_morton);

    int step = (end - start);
    int current_split = start;

    do {
        step = (step + 1) >> 1;
        int new_split = current_split + step;
        if (new_split < end) {
            int split_msd = use_tiebreaker ? (32 + clz(start ^ new_split)) : clz(start_morton ^ morton_list[new_split].morton_code);
            if (split_msd > common_bits)
                current_split = new_split;
        }
    } while (step > 1);

    return current_split;
}

aabb create_bbox(aabb box0, aabb box1) {
    aabb new_bbox;
    new_bbox.min_x = min(box0.min_x, box1.min_x);
    new_bbox.max_x = max(box0.max_x, box1.max_x);
    new_bbox.min_y = min(box0.min_y, box1.min_y);
    new_bbox.max_y = max(box0.max_y, box1.max_y);
    new_bbox.min_z = min(box0.min_z, box1.min_z);
    new_bbox.max_z = max(box0.max_z, box1.max_z);
    return new_bbox;
}

__kernel void compute_morton(
    __global const point3* barycenter,
    const aabb global_bbox,
    const int N,
    const uint cell_count,
    const uint k,
    __global morton_primitive* morton_list
) {
    int i = get_global_id(0);
    if (i >= N) return;

    float x_cell = cell_count * (barycenter[i].x - global_bbox.min_x) / (global_bbox.max_x - global_bbox.min_x);
    float y_cell = cell_count * (barycenter[i].y - global_bbox.min_y) / (global_bbox.max_y - global_bbox.min_y);
    float z_cell = cell_count * (barycenter[i].z - global_bbox.min_z) / (global_bbox.max_z - global_bbox.min_z);

    uint x = min((uint)x_cell, cell_count - 1);
    uint y = min((uint)y_cell, cell_count - 1);
    uint z = min((uint)z_cell, cell_count - 1);

    uint code = 0;
    for (int j = 0; j < k; j++) {
        code |= ((x >> j) & 1) << (3 * j);
        code |= ((y >> j) & 1) << (3 * j + 1);
        code |= ((z >> j) & 1) << (3 * j + 2);
    }

    morton_list[i].morton_code = code;
    morton_list[i].primitive_id = i;
}

__kernel void init_leaf_nodes(
    __global const morton_primitive* morton_list,
    __global const aabb* primitive_bboxes,
    int N,
    __global lbvh_node* nodes
) {
    int i = get_global_id(0);
    if (i >= N) return;

    int leaf_idx = N - 1 + i;
    int prim_id = morton_list[i].primitive_id;

    nodes[leaf_idx].bbox = primitive_bboxes[prim_id];
    nodes[leaf_idx].primitive_id = prim_id;
    nodes[leaf_idx].atomic_counter = 0;
    nodes[leaf_idx].parent = -1;
    nodes[leaf_idx].left = -1;
    nodes[leaf_idx].right = -1;
}

__kernel void create_hierarchy(
    __global const morton_primitive* morton_list,
    int N,
    __global lbvh_node* nodes
) {
    int node_idx = get_global_id(0);
    if (node_idx >= N - 1) return;

    if (node_idx == 0) {
        nodes[node_idx].parent = -1;
    }
    nodes[node_idx].left = -1;
    nodes[node_idx].right = -1;

    int2 range = find_range(node_idx, morton_list, N);
    int split = find_split(range.x, range.y, morton_list);

    int left_node = (split == range.x) ? (split + N - 1) : split;
    int right_node = (split + 1 == range.y) ? (split + N) : (split + 1);

    nodes[node_idx].left = left_node;
    nodes[node_idx].right = right_node;
    nodes[node_idx].primitive_id = -1;

    nodes[node_idx].bbox.min_x = INFINITY;
    nodes[node_idx].bbox.max_x = -INFINITY;
    nodes[node_idx].bbox.min_y = INFINITY;
    nodes[node_idx].bbox.max_y = -INFINITY;
    nodes[node_idx].bbox.min_z = INFINITY;
    nodes[node_idx].bbox.max_z = -INFINITY;
    nodes[node_idx].atomic_counter = 0;

    atomic_xchg(&nodes[left_node].parent, node_idx);
    atomic_xchg(&nodes[right_node].parent, node_idx);
}

__kernel void build_bboxes(__global lbvh_node* nodes, int N) {
    int leaf_idx = get_global_id(0);
    if (leaf_idx >= N) return;

    int current_idx = N - 1 + leaf_idx;
    int parent_idx = nodes[current_idx].parent;

    while (parent_idx != -1) {
        int old_count = atomic_inc(&nodes[parent_idx].atomic_counter);
        if (old_count == 0) break;  // First visitor waits

        int left_idx = nodes[parent_idx].left;
        int right_idx = nodes[parent_idx].right;
        nodes[parent_idx].bbox = create_bbox(nodes[left_idx].bbox, nodes[right_idx].bbox);
        mem_fence(CLK_GLOBAL_MEM_FENCE);

        current_idx = parent_idx;
        parent_idx = nodes[current_idx].parent;
    }
}

__kernel void create_histogram(
    __global const morton_primitive* morton_list,
    int N,
    int pass_idx,
    __global uint* histogram
) {
    int tid = get_local_id(0);
    int grp = get_group_id(0);
    int items = get_local_size(0);
    int groups = get_num_groups(0);

    __local uint loc_histo[16 * 256];

    for (int digit = 0; digit < 16; digit++)
        loc_histo[digit * items + tid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < 4; i++) {
        int idx = tid * 4 + grp * 1024 + i;
        if (idx >= N) continue;
        uint digit = (morton_list[idx].morton_code >> (pass_idx * 4)) & 15;
        loc_histo[digit * items + tid]++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int digit = 0; digit < 16; digit++)
        histogram[items * (digit * groups + grp) + tid] = loc_histo[digit * items + tid];
}

__kernel void prefix_sum(
    __global uint* histogram,
    __global uint* scanned_gram,
    __global uint* glob_sum,
    int num_blocks,
    int N
) {
    int tid = get_local_id(0);
    int bid = get_group_id(0);
    int id = get_global_id(0);
    int n = 512;
    __local uint temp[512];
    int offset = 1;

    temp[tid*2] = (id*2 < N) ? histogram[id*2] : 0;
    temp[tid*2 + 1] = (id*2 + 1 < N) ? histogram[id*2 + 1] : 0;

    for (int d = n >> 1; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < d) {
            int left = offset * (2 * tid + 1) - 1;
            int right = offset * (2 * tid + 2) - 1;
            temp[right] += temp[left];
        }
        offset *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 255) {
        glob_sum[bid] = temp[tid*2 + 1];
        temp[tid*2 + 1] = 0;
    }

    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < d) {
            int left = offset * (2 * tid + 1) - 1;
            int right = offset * (2 * tid + 2) - 1;
            uint t = temp[left];
            temp[left] = temp[right];
            temp[right] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (id*2 < N) scanned_gram[id*2] = temp[tid*2];
    if (id*2 + 1 < N) scanned_gram[id*2 + 1] = temp[tid*2 + 1];
}

__kernel void scatter(
    __global const morton_primitive* morton_list,
    __global const uint* scanned_gram,
    int N,
    int pass_idx,
    __global morton_primitive* output_list
) {
    int tid = get_local_id(0);
    int grp = get_group_id(0);
    int items = get_local_size(0);
    int groups = get_num_groups(0);

    __local uint loc_histo[16 * 256];

    for (int digit = 0; digit < 16; digit++)
        loc_histo[digit * items + tid] = scanned_gram[items * (digit * groups + grp) + tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < 4; i++) {
        int idx = tid * 4 + grp * 1024 + i;
        if (idx >= N) continue;

        morton_primitive elem = morton_list[idx];
        uint digit = (elem.morton_code >> (pass_idx * 4)) & 15;
        uint pos = loc_histo[digit * items + tid];
        loc_histo[digit * items + tid] = pos + 1;
        output_list[pos] = elem;
    }
}

__kernel void scan_glob_sum(__global uint* glob_sum, __local uint* temp, int n) {
    int tid = get_local_id(0);

    temp[tid] = (tid < n) ? glob_sum[tid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = 1; offset < get_local_size(0); offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < get_local_size(0))
            temp[idx] += temp[idx - offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0)
        temp[get_local_size(0) - 1] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = get_local_size(0) >> 1; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < get_local_size(0)) {
            uint t = temp[idx - offset];
            temp[idx - offset] = temp[idx];
            temp[idx] += t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid < n)
        glob_sum[tid] = temp[tid];
}

__kernel void add_glob_sums(__global uint* glob_sum, __global uint* scanned_gram, int N) {
    int id = get_global_id(0);
    int bid = get_group_id(0);
    if (id*2 < N) scanned_gram[id*2] += glob_sum[bid];
    if (id*2 + 1 < N) scanned_gram[id*2 + 1] += glob_sum[bid];
}
