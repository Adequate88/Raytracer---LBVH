//==============================================================================================
// Material Functions
//==============================================================================================

// Get emitted color from material (lights only)
inline point3 material_emitted(int mat_idx, __global const gpu_material* materials) {
  if (mat_idx < 0) {
    point3 black = {0.0f, 0.0f, 0.0f};
    return black;
  }

  gpu_material mat = materials[mat_idx];

  if (mat.type == MAT_LIGHT) {
    return mat.albedo;  // Lights emit their albedo color
  }

  point3 black = {0.0f, 0.0f, 0.0f};
  return black;
}

// Lambertian (diffuse) material scatter
inline bool scatter_lambertian(ray r_in, hit_record rec, gpu_material mat,
                                uint* rng_state, point3* attenuation, ray* scattered) {
  // Random diffuse direction
  point3 scatter_direction = vec3_add(rec.normal, random_unit_vector(rng_state));

  // Catch degenerate scatter direction
  if (near_zero(scatter_direction)) {
    scatter_direction = rec.normal;
  }

  // Scattered ray
  scattered->origin = rec.p;
  scattered->direction = scatter_direction;

  // Attenuation is albedo color
  *attenuation = mat.albedo;

  return true;
}

// Metal (reflective) material scatter
inline bool scatter_metal(ray r_in, hit_record rec, gpu_material mat,
                          uint* rng_state, point3* attenuation, ray* scattered) {
  // Reflect the ray
  point3 reflected = reflect(normalize(r_in.direction), rec.normal);

  // Add fuzz (roughness)
  point3 fuzz_offset = vec3_scale(random_unit_vector(rng_state), mat.fuzz);
  reflected = vec3_add(reflected, fuzz_offset);

  // Scattered ray
  scattered->origin = rec.p;
  scattered->direction = reflected;

  // Attenuation is albedo color
  *attenuation = mat.albedo;

  // Only scatter if reflected ray is above surface
  return (dot(scattered->direction, rec.normal) > 0);
}

// Light material (absorbs rays, doesn't scatter)
inline bool scatter_light(ray r_in, hit_record rec, gpu_material mat,
                          uint* rng_state, point3* attenuation, ray* scattered) {
  // Lights don't scatter - they absorb rays
  return false;
}

// Unified material scatter dispatch
inline bool scatter_material(ray r_in, hit_record rec,
                             __global const gpu_material* materials,
                             uint* rng_state, point3* attenuation, ray* scattered) {
  if (rec.mat_idx < 0) {
    return false;  // No material
  }

  gpu_material mat = materials[rec.mat_idx];

  switch (mat.type) {
    case MAT_LAMBERTIAN:
      return scatter_lambertian(r_in, rec, mat, rng_state, attenuation, scattered);

    case MAT_METAL:
      return scatter_metal(r_in, rec, mat, rng_state, attenuation, scattered);

    case MAT_LIGHT:
      return scatter_light(r_in, rec, mat, rng_state, attenuation, scattered);

    default:
      return false;  // Unknown material type
  }
}

// Ray-sphere intersection (adapted from sphere.h)
inline bool sphere_intersection(ray r, float t_min, float t_max, hit_record* rec, sphere _sphere) {
  point3 oc;
  oc.x = _sphere.center.x - r.origin.x;
  oc.y = _sphere.center.y - r.origin.y;
  oc.z = _sphere.center.z - r.origin.z;

  float a = length_squared(r.direction);
  float h = dot(r.direction, oc);
  float c = length_squared(oc) - _sphere.radius * _sphere.radius;

  float discriminant = h*h - a*c;
  if (discriminant < 0)
    return false;

  float sqrtd = sqrt(discriminant);
  float root = (h - sqrtd) / a;
  if (root >= t_max || root <= t_min) {
    root = (h + sqrtd) / a;
    if (root >= t_max || root <= t_min)
      return false;
  }

  rec->t = root;
  rec->p = ray_at(r, rec->t);

  point3 outward_normal;
  outward_normal.x = (rec->p.x - _sphere.center.x) / _sphere.radius;
  outward_normal.y = (rec->p.y - _sphere.center.y) / _sphere.radius;
  outward_normal.z = (rec->p.z - _sphere.center.z) / _sphere.radius;

  rec->front_face = dot(r.direction, outward_normal) < 0;

  if (rec->front_face) {
    rec->normal = outward_normal;
  } else {
    rec->normal.x = -outward_normal.x;
    rec->normal.y = -outward_normal.y;
    rec->normal.z = -outward_normal.z;
  }

  rec->mat_idx = _sphere.mat_idx;

  return true;
}

inline bool aabb_hit(aabb box, ray r, float t_min, float t_max) {
  for (int axis = 0; axis < 3; axis++) {
    float ray_orig, ray_dir, box_min, box_max;

    if (axis == 0) {        
      ray_orig = r.origin.x;
      ray_dir = r.direction.x;
      box_min = box.min_x;
      box_max = box.max_x;
    } else if (axis == 1) { 
      ray_orig = r.origin.y;
      ray_dir = r.direction.y;
      box_min = box.min_y;
      box_max = box.max_y;
    } else {               
      ray_orig = r.origin.z;
      ray_dir = r.direction.z;
      box_min = box.min_z;
      box_max = box.max_z;
    }

    float adinv = 1.0f / ray_dir; 
    float t0 = (box_min - ray_orig) * adinv;
    float t1 = (box_max - ray_orig) * adinv;

    if (t0 < t1) {
      if (t0 > t_min) t_min = t0;
      if (t1 < t_max) t_max = t1;
    } else {
      if (t1 > t_min) t_min = t1;
      if (t0 < t_max) t_max = t0;
    }

    if (t_max <= t_min)
      return false;
  }
  return true;
}

//==============================================================================================
// Forward declaration for ray_color
//==============================================================================================

inline bool bvh_traverse(
  ray r, float t_min, float t_max,
  hit_record* rec,
  __global const lbvh_node* nodes,
  __global const sphere* spheres,
  __global const int* leaf_prim_indices,
  int N
);

//==============================================================================================
// Iterative Ray Tracing (Path Tracing with Multiple Bounces)
//==============================================================================================

inline point3 ray_color(
  ray initial_ray, int max_depth, point3 background,
  __global const lbvh_node* nodes,
  __global const sphere* spheres,
  __global const int* leaf_prim_indices,
  __global const gpu_material* materials,
  int N, uint* rng_state
) {
  point3 accumulated_color = {0.0f, 0.0f, 0.0f};
  point3 accumulated_attenuation = {1.0f, 1.0f, 1.0f};
  ray current_ray = initial_ray;

  for (int depth = 0; depth < max_depth; depth++) {
    hit_record rec;

    // Trace ray through BVH
    if (!bvh_traverse(current_ray, 0.001f, 1e38f, &rec, nodes, spheres, leaf_prim_indices, N)) {
      // No hit - return background color weighted by accumulated attenuation
      point3 bg_contribution = vec3_mul(accumulated_attenuation, background);
      accumulated_color = vec3_add(accumulated_color, bg_contribution);
      break;
    }

    // Add emission from hit surface (lights)
    point3 emission = material_emitted(rec.mat_idx, materials);
    point3 emission_contribution = vec3_mul(accumulated_attenuation, emission);
    accumulated_color = vec3_add(accumulated_color, emission_contribution);

    // Try to scatter ray
    point3 attenuation;
    ray scattered;
    if (!scatter_material(current_ray, rec, materials, rng_state, &attenuation, &scattered)) {
      // Ray absorbed - no more bounces
      break;
    }

    // Update accumulated attenuation and continue with scattered ray
    accumulated_attenuation = vec3_mul(accumulated_attenuation, attenuation);
    current_ray = scattered;
  }

  return accumulated_color;
}

inline bool bvh_traverse(
  ray r, float t_min, float t_max,
  hit_record* rec,
  __global const lbvh_node* nodes,
  __global const sphere* spheres,
  __global const int* leaf_prim_indices,
  int N
  ) {

  int stack[128];  // Increased from 64 for deep trees (depth 44+)
  int stack_ptr = 0;

  stack[stack_ptr++] = 0;

  bool hit_anything = false;
  float closest_t = t_max;

  while (stack_ptr > 0) {
    int node_idx = stack[--stack_ptr];
    lbvh_node node = nodes[node_idx];

    if (!aabb_hit(node.bbox, r, t_min, closest_t))
      continue;

    if (node.primitive_id >= 0) {
      hit_record temp_rec;
      int sphere_idx = leaf_prim_indices[node.primitive_id];

      if (sphere_idx >= 0 && sphere_intersection(r, t_min, closest_t, &temp_rec, spheres[sphere_idx])) {
        if (temp_rec.t < closest_t) {
          closest_t = temp_rec.t;
          *rec = temp_rec;
          rec->prim_idx = sphere_idx;
          rec->prim_type = 0;
          hit_anything = true;
        }
      }
    } else {
      if (node.left >= 0 && node.left < 2*N-1 && stack_ptr < 128) {
        stack[stack_ptr++] = node.left;
      }
      if (node.right >= 0 && node.right < 2*N-1 && stack_ptr < 128) {
        stack[stack_ptr++] = node.right;
      }
    }
  }
  return hit_anything;
}

__kernel void render_primary_rays(
    point3 camera_center,
    point3 pixel00_loc,
    point3 pixel_delta_u,
    point3 pixel_delta_v,
    int image_width,
    int image_height,
    point3 background,
    __global const lbvh_node* nodes,
    __global const sphere* spheres,
    __global const int* leaf_prim_indices,
    int N,
    __global point3* output_pixels
) {
    int pixel_x = get_global_id(0);
    int pixel_y = get_global_id(1);

    if (pixel_x >= image_width || pixel_y >= image_height)
        return;

    point3 pixel_center;
    pixel_center.x = pixel00_loc.x + pixel_x * pixel_delta_u.x + pixel_y * pixel_delta_v.x;
    pixel_center.y = pixel00_loc.y + pixel_x * pixel_delta_u.y + pixel_y * pixel_delta_v.y;
    pixel_center.z = pixel00_loc.z + pixel_x * pixel_delta_u.z + pixel_y * pixel_delta_v.z;

    ray r;
    r.origin = camera_center;
    r.direction.x = pixel_center.x - camera_center.x;
    r.direction.y = pixel_center.y - camera_center.y;
    r.direction.z = pixel_center.z - camera_center.z;

    hit_record rec;
    point3 color;

    if (bvh_traverse(r, 0.001f, 1e38f, &rec, nodes, spheres, leaf_prim_indices, N)) {
        color.x = 0.5f * (rec.normal.x + 1.0f);
        color.y = 0.5f * (rec.normal.y + 1.0f);
        color.z = 0.5f * (rec.normal.z + 1.0f);
    } else {
        color.x = background.x;
        color.y = background.y;
        color.z = background.z;
    }

    int pixel_idx = pixel_y * image_width + pixel_x;
    output_pixels[pixel_idx] = color;
}

//==============================================================================================
// Phase 2: Full Path Tracing with Multi-Sampling and Defocus Blur
//==============================================================================================

__kernel void render_with_sampling(
    // Camera parameters (7)
    point3 camera_center,
    point3 pixel00_loc,
    point3 pixel_delta_u,
    point3 pixel_delta_v,
    point3 defocus_disk_u,
    point3 defocus_disk_v,
    float defocus_angle,

    // Image parameters (5)
    int image_width,
    int image_height,
    int samples_per_pixel,
    int max_depth,
    point3 background,

    // Scene parameters (5)
    __global const lbvh_node* nodes,
    __global const sphere* spheres,
    __global const int* leaf_prim_indices,
    __global const gpu_material* materials,
    int N,

    // Output (1)
    __global point3* output_pixels
) {
    int pixel_x = get_global_id(0);
    int pixel_y = get_global_id(1);

    if (pixel_x >= image_width || pixel_y >= image_height)
        return;

    point3 pixel_color = {0.0f, 0.0f, 0.0f};

    // Multi-sampling loop
    for (int sample = 0; sample < samples_per_pixel; sample++) {
        // Initialize RNG for this sample
        uint rng_state = seed_rng(pixel_x, pixel_y, sample, image_width);

        // Pixel jitter for anti-aliasing
        float jitter_x = random_float(&rng_state) - 0.5f;
        float jitter_y = random_float(&rng_state) - 0.5f;

        // Compute pixel sample location
        point3 pixel_sample;
        pixel_sample.x = pixel00_loc.x + (pixel_x + jitter_x) * pixel_delta_u.x
                                       + (pixel_y + jitter_y) * pixel_delta_v.x;
        pixel_sample.y = pixel00_loc.y + (pixel_x + jitter_x) * pixel_delta_u.y
                                       + (pixel_y + jitter_y) * pixel_delta_v.y;
        pixel_sample.z = pixel00_loc.z + (pixel_x + jitter_x) * pixel_delta_u.z
                                       + (pixel_y + jitter_y) * pixel_delta_v.z;

        // Ray origin (with optional defocus blur)
        ray r;
        if (defocus_angle <= 0.0f) {
            // No defocus blur - ray from camera center
            r.origin = camera_center;
        } else {
            // Defocus blur - randomize origin on defocus disk
            point3 disk_sample = random_in_unit_disk(&rng_state);
            point3 offset;
            offset.x = defocus_disk_u.x * disk_sample.x + defocus_disk_v.x * disk_sample.y;
            offset.y = defocus_disk_u.y * disk_sample.x + defocus_disk_v.y * disk_sample.y;
            offset.z = defocus_disk_u.z * disk_sample.x + defocus_disk_v.z * disk_sample.y;

            r.origin = vec3_add(camera_center, offset);
        }

        // Ray direction points to pixel sample
        r.direction = vec3_sub(pixel_sample, r.origin);

        // Trace ray with bouncing and materials
        point3 sample_color = ray_color(r, max_depth, background,
                                        nodes, spheres, leaf_prim_indices, materials,
                                        N, &rng_state);

        // Accumulate sample
        pixel_color = vec3_add(pixel_color, sample_color);
    }

    // Average samples
    float scale = 1.0f / samples_per_pixel;
    pixel_color = vec3_scale(pixel_color, scale);

    // Write to output
    int pixel_idx = pixel_y * image_width + pixel_x;
    output_pixels[pixel_idx] = pixel_color;
}

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

