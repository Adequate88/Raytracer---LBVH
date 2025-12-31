// Structs 

typedef struct {
  float x, y, z;
} point3;

typedef struct {
  float min_x, min_y, min_z;
  float max_x, max_y, max_z;
} aabb;

typedef struct {
  uint morton_code;
  int primitive_id;
} morton_primitive;

// Note: int2 is a built-in OpenCL vector type, no need to define it

typedef struct {
    aabb bbox;
    int left;
    int right;
    int parent;
    int primitive_id; 
} lbvh_node;

// Helper Functions

inline int delta(int i, int j,__global const morton_primitive* morton_list, int N) {
  if (j < 0 || j >= N) return -1;

  uint code_i = morton_list[i].morton_code;
  uint code_j = morton_list[j].morton_code;

  return clz(code_i ^ code_j);
}

inline int2 find_range(int idx, __global const morton_primitive* morton_list, int N) {
  int dir = (delta(idx, idx + 1, morton_list, N) - delta(idx, idx - 1, morton_list, N)) > 0 ? 1 : -1; // If positive return 1, otherwise -1

  // Get minimum bound at i - d
  int min_bound = delta(idx, idx - dir, morton_list, N);
      
  // Get Lmax bound of range
  int l_max = 2; // Initial max bound;
  while (delta(idx, idx + dir*l_max, morton_list, N) > min_bound) 
    l_max *= 2; 
      

  // Binary search for end range
  int l = 0; // Find distance to end range 

  for (int t = l_max / 2; t >= 1; t /= 2) {
    if (delta(idx, idx + (t + l)*dir, morton_list, N) > min_bound) {
      l += t;
    }
  }
      
  int final_point = idx + l*dir;
      
  int2 range;
  if ( final_point > idx) {
    range.x = idx;
    range.y = final_point;
  } else {
    range.x = final_point;
    range.y = idx;
  }

  return range;
  
}


inline int find_split(int start, int end, __global const morton_primitive* morton_list) {
  uint start_morton = morton_list[start].morton_code;
  uint end_morton = morton_list[end].morton_code;

  if (start_morton == end_morton)
    return (start + end) >> 1;

  int common_bits = clz(start_morton ^ end_morton);

  int step = (end - start);
  int current_split = start;

  do {
    step = (step + 1) >> 1;
    int new_split = current_split + step;

    if (new_split < end) {
      uint split_morton = morton_list[new_split].morton_code;
      int split_msd = clz(start_morton ^ split_morton);

      if (split_msd > common_bits)
        current_split = new_split;
      }
    } while (step > 1);

    return current_split;
}

// Kernels

__kernel void compute_morton(
  __global const point3* barycenter,
  const aabb global_bbox,
  const int N,
  const uint cell_count,
  const uint k,
  __global morton_primitive* morton_list
  ) {

  int i = get_global_id(0);
  
  if (i >= N)
    return;

  // Compute cell of barycenter at each axis
  float x_cell = cell_count * (barycenter[i].x - global_bbox.min_x) / ( global_bbox.max_x - global_bbox.min_x );
  float y_cell = cell_count * (barycenter[i].y - global_bbox.min_y) / ( global_bbox.max_y - global_bbox.min_y );
  float z_cell = cell_count * (barycenter[i].z - global_bbox.min_z) / ( global_bbox.max_z - global_bbox.min_z );

  uint x = min((uint)x_cell, cell_count - 1);
  uint y = min((uint)y_cell, cell_count - 1);
  uint z = min((uint)z_cell, cell_count - 1);

  uint code = 0;
  for (int j = 0; j < k; j++){
    code |= ( (x >> j) & 1 ) << (3 * j);
    code |= ( (y >> j) & 1 ) << (3 * j + 1);
    code |= ( (z >> j) & 1 ) << (3 * j + 2);
  }

  morton_list[i].morton_code = code;
  morton_list[i].primitive_id = i;
  
}

__kernel void init_leaf_nodes (
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
  if (node_idx >= N -1) return;

  int2 range = find_range(node_idx, morton_list, N);
  int start = range.x;
  int end = range.y;
  // Get split idx
  int split = find_split(start, end, morton_list);

  // Assign children
  int left_node;
  if (split == start) { // Is leaf
    left_node = split + N - 1;
  } else {
    left_node = split;
  }

  int right_node;
  if (split + 1 == end) { // Is leaf
    right_node = split + N;
  } else {
    right_node = split + 1 ;
  }

  nodes[node_idx].left = left_node;
  nodes[node_idx].right = right_node;
  nodes[node_idx].primitive_id = -1;  // Internal nodes have no primitive

  // Initialize bbox to empty (min > max) so bottom_up_bbox_build knows it needs processing
  nodes[node_idx].bbox.min_x = INFINITY;
  nodes[node_idx].bbox.max_x = -INFINITY;
  nodes[node_idx].bbox.min_y = INFINITY;
  nodes[node_idx].bbox.max_y = -INFINITY;
  nodes[node_idx].bbox.min_z = INFINITY;
  nodes[node_idx].bbox.max_z = -INFINITY;

  nodes[left_node].parent = node_idx;
  nodes[right_node].parent = node_idx;
  
}
