#include "triangle.glsl"

// DUPLICATED CODE BLOCK

struct aabb {
  vec4 corner1;
  vec4 corner2;
};

struct node {
  aabb bbox;
  int left;
  int right;
  int parent;
  int primitive_id;
  int atomic_counter; // 52 bytes
  int pad_to_64[3]; // Vec4 has alignment of 16 bytes, so node has alignment of 16 bytes, so to ensure correct reads we need sizeof(node) % 16 == 0.
};

bool aabb_hit(vec3 ray_origin, vec3 adinv, float t_curr_min,
  vec3 aabb_min, vec3 aabb_max) {

  vec3 t0 = (aabb_min - ray_origin) * adinv;
  vec3 t1 = (aabb_max - ray_origin) * adinv;

  vec3 t_min_vec = min(t0, t1);
  vec3 t_max_vec = max(t0, t1);

  float t_min = max(max(t_min_vec.x, t_min_vec.y), t_min_vec.z);
  float t_max = min(min(t_max_vec.x, t_max_vec.y), t_max_vec.z);

  return (t_min <= t_max) && (t_max > 0.0) && (t_min < t_curr_min);
}

layout(binding = 1, std430) readonly buffer SceneBuffer {
  triangle triangles[];
} scene;

// END OF DUPLICATED CODE BLOCK

const int BVH_STACK_SIZE = 32;

// layout(push_constant) uniform Camera {} cam;

layout(binding = 2, std430) buffer BvhBuffer {
  node nodes[];
} bvh;

bool bvh_hit(vec3 ray_origin, vec3 ray_dir, out hit_record rec) {
  int stack[BVH_STACK_SIZE];
  int stack_ptr = 0;
  stack[stack_ptr++] = -1;
  stack[stack_ptr] = 0; // Root node

  rec.t = 1e30;
  bool found_hit = false;

  vec3 ray_dir_inv = 1.0f / ray_dir;

  // Traverse
  do {
    // temporarily loading current node into this register, to avoid loading it twice
    node child_l = bvh.nodes[stack[stack_ptr]];

    int curr_left_idx = child_l.left;
    int curr_right_idx = child_l.right;

    child_l = bvh.nodes[curr_left_idx];
    node child_r = bvh.nodes[curr_right_idx];

    // Process left child
    bool hit = aabb_hit(ray_origin, ray_dir_inv, rec.t, 
      min(child_l.bbox.corner1, child_l.bbox.corner2).xyz,  // child min
      max(child_l.bbox.corner1, child_l.bbox.corner2).xyz); // child max
    bool traverse_l = (hit && child_l.primitive_id < 0);
    if (hit && child_l.primitive_id >= 0) {
      hit_record temp_rec;
      if (intersect_triangle(ray_origin, ray_dir, rec.t,
          scene.triangles[child_l.primitive_id], temp_rec)) {
        rec = temp_rec;
        found_hit = true;
      }
    }

    // Process right child
    hit = aabb_hit(ray_origin, ray_dir_inv, rec.t, 
      min(child_r.bbox.corner1, child_r.bbox.corner2).xyz,  // child min
      max(child_r.bbox.corner1, child_r.bbox.corner2).xyz); // child max
    bool traverse_r = (hit && child_r.primitive_id < 0);
    if (hit && child_r.primitive_id >= 0) {
      hit_record temp_rec;
      if (intersect_triangle(ray_origin, ray_dir, rec.t,
          scene.triangles[child_r.primitive_id], temp_rec)) {
        rec = temp_rec;
        found_hit = true;
      }
    }
    
    if (!traverse_l && !traverse_r) {
      stack_ptr--;
    }
    else {
      stack[stack_ptr] = (traverse_l) ? curr_left_idx : curr_right_idx;
      if (traverse_l && traverse_r) {
        stack_ptr++;
        stack[stack_ptr] = curr_right_idx;
      }
    }
  }
  while (stack_ptr >= 0 && stack[stack_ptr] != -1);

  return found_hit;
}
