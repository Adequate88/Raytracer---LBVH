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
