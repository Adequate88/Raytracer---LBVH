inline int delta(int i, int j,__global const morton_primitive* morton_list, int N) {
  if (j < 0 || j >= N) return -1;

  uint code_i = morton_list[i].morton_code;
  uint code_j = morton_list[j].morton_code;

  if (code_i == code_j) {
    return 32 + clz(i ^ j);
  }

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
        uint split_morton = morton_list[new_split].morton_code;
        split_msd = clz(start_morton ^ split_morton);
      }

      if (split_msd > common_bits)
        current_split = new_split;
    }
  } while (step > 1);

  return current_split;
}

aabb create_bbox (aabb box0, aabb box1) {
  aabb new_bbox;
  new_bbox.min_x = min(box0.min_x, box1.min_x);
  new_bbox.max_x = max(box0.max_x, box1.max_x);
  new_bbox.min_y = min(box0.min_y, box1.min_y); 
  new_bbox.max_y = max(box0.max_y, box1.max_y); 
  new_bbox.min_z = min(box0.min_z, box1.min_z); 
  new_bbox.max_z = max(box0.max_z, box1.max_z); 
  return new_bbox;
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
  if (node_idx >= N -1) return;

  // Initialize this internal node's parent to -1 (will be set by parent's thread)
  nodes[node_idx].parent = -1;
  nodes[node_idx].left = -1;
  nodes[node_idx].right = -1;

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

  // Initialize atomic counter for bbox construction
  nodes[node_idx].atomic_counter = 0;

  // Use atomic operations to safely set parent pointers (prevents race conditions)
  atomic_xchg(&nodes[left_node].parent, node_idx);
  atomic_xchg(&nodes[right_node].parent, node_idx);
  
}

// Parallel bottom-up bounding box construction (Karras 2012)
// Each thread starts from a leaf and walks up the tree using parent pointers.
// Atomic counters track arrivals: first thread waits, second thread processes the node.
__kernel void build_bboxes(
  __global lbvh_node* nodes,
  int N  // Number of primitives (leaf nodes)
) {
  // Get leaf index for this thread (one thread per leaf)
  int leaf_idx = get_global_id(0);
  if (leaf_idx >= N) return;

  // Convert to actual node index (leaves are stored at indices N-1 to 2N-2)
  int current_idx = N - 1 + leaf_idx;

  // Get parent of this leaf node
  int parent_idx = nodes[current_idx].parent;

  // Walk up the tree until we can't proceed
  while (parent_idx != -1) {
    // Atomically increment parent's counter and get old value
    // old_count == 0: first child arrived
    // old_count == 1: second child arrived (both children ready!)
    int old_count = atomic_inc(&nodes[parent_idx].atomic_counter);

    if (old_count == 0) {
      // First child arrived - parent not ready yet
      // Terminate this thread's upward traversal
      break;
    }

    // Second child arrived (old_count == 1) - both children are done!
    // Now we can safely compute this parent's bounding box

    int left_idx = nodes[parent_idx].left;
    int right_idx = nodes[parent_idx].right;

    aabb left_bbox = nodes[left_idx].bbox;
    aabb right_bbox = nodes[right_idx].bbox;

    // Compute parent's bbox as union of children's bboxes
    nodes[parent_idx].bbox = create_bbox(left_bbox, right_bbox);

    // Move up to grandparent and continue
    current_idx = parent_idx;
    parent_idx = nodes[current_idx].parent;
  }
}

