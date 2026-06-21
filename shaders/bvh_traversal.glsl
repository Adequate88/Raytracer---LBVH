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

bool aabb_hit(vec3 ray_origin, vec3 ray_dir, float t_curr_min,
  vec3 aabb_min, vec3 aabb_max) {
  vec3 adinv = 1.0f / ray_dir; // TODO: replace with precomputed argument

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

const int BVH_STACK_SIZE = 64;

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

  // Traverse
  do {
    node curr_node = bvh.nodes[stack[stack_ptr]];

    node child_l = bvh.nodes[curr_node.left];
    node child_r = bvh.nodes[curr_node.right];

    // Intersection with bounding boxes of nodes
    vec3 child_l_min = min(child_l.bbox.corner1, child_l.bbox.corner2).xyz;
    vec3 child_l_max = max(child_l.bbox.corner1, child_l.bbox.corner2).xyz;
    bool hit_l = aabb_hit(
        ray_origin, ray_dir, rec.t,
        child_l_min, child_l_max);
    vec3 child_r_min = min(child_r.bbox.corner1, child_r.bbox.corner2).xyz;
    vec3 child_r_max = max(child_r.bbox.corner1, child_r.bbox.corner2).xyz;
    bool hit_r = aabb_hit(
        ray_origin, ray_dir, rec.t,
        child_r_min, child_r_max);

    // If these are leaf nodes, intersect with their triangles
    if (hit_l && child_l.primitive_id >= 0) {
      hit_record temp_rec;
      if (intersect_triangle(ray_origin, ray_dir, rec.t,
          scene.triangles[child_l.primitive_id], temp_rec)) {
        rec = temp_rec;
        found_hit = true;
      }
    }
    if (hit_r && child_r.primitive_id >= 0) {
      hit_record temp_rec;
      if (intersect_triangle(ray_origin, ray_dir, rec.t,
          scene.triangles[child_r.primitive_id], temp_rec)) {
        rec = temp_rec;
        found_hit = true;
      }
    }

    // If these are not leaf nodes, traverse the nodes
    bool traverse_l = (hit_l && child_l.primitive_id < 0);
    bool traverse_r = (hit_r && child_r.primitive_id < 0);

    if (!traverse_l && !traverse_r) {
      stack_ptr--;
    }
    else {
      stack[stack_ptr] = (traverse_l) ? curr_node.left : curr_node.right;
      if (traverse_l && traverse_r) {
        stack_ptr++;
        stack[stack_ptr] = curr_node.right;
      }
    }
  }
  while (stack_ptr >= 0 && stack[stack_ptr] != -1);

  return found_hit;
}
