#include "triangle.glsl"

#define FLT_MAX 3.4e38
#define LOCAL_SIZE gl_WorkGroupSize.x

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

layout(local_size_x = 256) in;

layout(binding = 0, std430) readonly buffer SceneBuffer {
  triangle triangles[];
} scene;

layout(binding = 1, std430) coherent buffer BvhBuffer {
  node nodes[];
};

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

layout(binding = 2, std430) buffer primBboxesBuffer {
  aabb primBBoxes[];
};

layout(binding = 3, std430) buffer mortonCodesBuffer {
  uint mortonCodes[];
};

layout(binding = 4, std430) buffer primIndicesBuffer {
  uint primIndices[];
};

layout(binding = 5, std430) buffer worldBboxBuffer {
  aabb worldBbox;
};

layout(binding = 6, std430) buffer histogramBuffer {
  uint histogram[];
};

layout(binding = 7, std430) buffer scannedGramBuffer {
  uint scanned_gram[];
};

layout(binding = 8, std430) buffer globSumBuffer {
  uint glob_sum[];
};

layout(binding = 9, std430) buffer outputMortonCodesBuffer {
  uint outputMortonCodes[];
};

layout(binding = 10, std430) buffer outputPrimIndicesBuffer {
  uint outputPrimIndices[];
};

layout(push_constant) uniform PushConstants {
  int primitiveCount;
  int pass_idx;
  int num_blocks;
  int histogram_size;
};
