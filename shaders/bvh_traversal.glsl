#include "triangle.glsl"

// DUPLICATED CODE BLOCK

layout(binding = 8, std430) readonly buffer SceneBuffer {
  triangle triangles[];
} scene;

layout(binding = 11, std430) readonly buffer corner1xBuffer {
  float corner1x[];
};
layout(binding = 12, std430) readonly buffer corner1yBuffer {
  float corner1y[];
};
layout(binding = 13, std430) readonly buffer corner1zBuffer {
  float corner1z[];
};
layout(binding = 14, std430) readonly buffer corner2xBuffer {
  float corner2x[];
};
layout(binding = 15, std430) readonly buffer corner2yBuffer {
  float corner2y[];
};
layout(binding = 16, std430) readonly buffer corner2zBuffer {
  float corner2z[];
};
layout(binding = 17, std430) readonly buffer leftChildsBuffer {
  int leftChilds[];
};
layout(binding = 18, std430) readonly buffer rightChildsBuffer {
  int rightChilds[];
};
layout(binding = 19, std430) readonly buffer primIndicesBuffer {
  uint primIndices[];
};

bool aabb_hit(vec3 ray_origin, vec3 adinv, float t_curr_min,
  int node_idx) {
  vec3 aabb_min = vec3(corner1x[node_idx], corner1y[node_idx], corner1z[node_idx]);
  vec3 aabb_max = vec3(corner2x[node_idx], corner2y[node_idx], corner2z[node_idx]);

  vec3 t0 = (aabb_min - ray_origin) * adinv;
  vec3 t1 = (aabb_max - ray_origin) * adinv;

  vec3 t_min_vec = min(t0, t1);
  vec3 t_max_vec = max(t0, t1);

  float t_min = max(max(t_min_vec.x, t_min_vec.y), t_min_vec.z);
  float t_max = min(min(t_max_vec.x, t_max_vec.y), t_max_vec.z);

  return (t_min <= t_max) && (t_max > 0.0) && (t_min < t_curr_min);
}

// END OF DUPLICATED CODE BLOCK

const int BVH_STACK_SIZE = 32;

// layout(push_constant) uniform Camera {} cam;

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

    int curr_left_idx = leftChilds[stack[stack_ptr]];
    int curr_right_idx = rightChilds[stack[stack_ptr]];

    // Process left child
    bool hit = aabb_hit(ray_origin, ray_dir_inv, rec.t,
        curr_left_idx); // child max
    //
    bool traverse_l = (hit && leftChilds[curr_left_idx] >= 0);
    if (hit && leftChilds[curr_left_idx] < 0) {
      hit_record temp_rec;
      if (intersect_triangle(ray_origin, ray_dir, rec.t,
          scene.triangles[primIndices[curr_left_idx + 1 - primitiveCount]], temp_rec)) {
        rec = temp_rec;
        found_hit = true;
      }
    }

    // Process right child
    hit = aabb_hit(ray_origin, ray_dir_inv, rec.t,
        curr_right_idx); // child max
    bool traverse_r = (hit && leftChilds[curr_right_idx] >= 0);
    if (hit && leftChilds[curr_right_idx] < 0) {
      hit_record temp_rec;
      if (intersect_triangle(ray_origin, ray_dir, rec.t,
          scene.triangles[primIndices[curr_right_idx + 1 - primitiveCount]], temp_rec)) {
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
