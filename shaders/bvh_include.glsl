#include "triangle.glsl"

#define FLT_MAX 3.4e38
#define LOCAL_SIZE gl_WorkGroupSize.x

layout(local_size_x = 256) in;

layout(binding = 0, std430) readonly buffer SceneBuffer {
  triangle triangles[];
} scene;

layout(binding = 3, std430) buffer mortonCodesBuffer {
  uint mortonCodes[];
};

layout(binding = 4, std430) buffer primIndicesBuffer {
  uint primIndices[];
};

layout(binding = 5, std430) buffer worldBboxBuffer {
  float corner1x;
  float corner1y;
  float corner1z;
  float corner2x;
  float corner2y;
  float corner2z;
} worldBbox;

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

// SoA node and bbox

layout(binding = 11, std430) buffer corner1xBuffer {
  float corner1x[];
};

layout(binding = 12, std430) buffer corner1yBuffer {
  float corner1y[];
};

layout(binding = 13, std430) buffer corner1zBuffer {
  float corner1z[];
};

layout(binding = 14, std430) buffer corner2xBuffer {
  float corner2x[];
};

layout(binding = 15, std430) buffer corner2yBuffer {
  float corner2y[];
};

layout(binding = 16, std430) buffer corner2zBuffer {
  float corner2z[];
};

layout(binding = 17, std430) buffer leftChildsBuffer {
  int leftChilds[];
};

layout(binding = 18, std430) buffer rightChildsBuffer {
  int rightChilds[];
};

layout(binding = 19, std430) buffer parentsBuffer {
  int parents[];
};

layout(binding = 20, std430) buffer atomicCounterBuffer {
  int atomicCounters[];
};

// Sorted-order destination arrays for the leaf-bbox reorder pass.
layout(binding = 21, std430) buffer corner1xOutBuffer {
  float corner1x_out[];
};

layout(binding = 22, std430) buffer corner1yOutBuffer {
  float corner1y_out[];
};

layout(binding = 23, std430) buffer corner1zOutBuffer {
  float corner1z_out[];
};

layout(binding = 24, std430) buffer corner2xOutBuffer {
  float corner2x_out[];
};

layout(binding = 25, std430) buffer corner2yOutBuffer {
  float corner2y_out[];
};

layout(binding = 26, std430) buffer corner2zOutBuffer {
  float corner2z_out[];
};

layout(push_constant) uniform PushConstants {
  int primitiveCount;
  int pass_idx;
  int num_blocks;
  int histogram_size;
};
