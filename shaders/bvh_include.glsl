#include "triangle.glsl"

struct node {
  float min_x, min_y, min_z;
  float max_x, max_y, max_z;
  int left;
  int right;
  int parent;
  int primitive_idx;
};

layout(local_size_x = 64) in;

// layout(push_constant) uniform Camera {} cam;
//
layout(binding = 0, std430) readonly buffer SceneBuffer {
  triangle triangles[];
} scene;

layout(binding = 1, std430) buffer BvhBuffer {
  node nodes[];
} bvh;

bool aabb_hit(vec3 ray_origin, vec3 ray_dir, float t_curr_min,
              vec3 aabb_min, vec3 aabb_max) {
  vec3 adinv = 1.0f / ray_dir; // TODO: replace with precomputed argument

  vec3 t0 = (aabb_min - ray_origin) * adinv;
  vec3 t1 = (aabb_max - ray_origin) * adinv;

  vec3 t_min_vec = min(t0, t1);
  vec3 t_max_vec = max(t0, t1);

  float t_min = min(min(t_min_vec.x, t_min_vec.y), t_min_vec.z);
  float t_max = max(max(t_max_vec.x, t_max_vec.y), t_max_vec.z);

  return (t_min <= t_max) && (t_max > 0.0) && (t_min < t_curr_min);
}