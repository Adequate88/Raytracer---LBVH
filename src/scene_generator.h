#pragma once

#include "scenes.h"
#include "triangle.h"
#include "vec3.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

inline void push_gpu_triangle(std::vector<triangle_new> &world, const vec3 &a,
                              const vec3 &b, const vec3 &c, const vec3 &na,
                              const vec3 &nb, const vec3 &nc) {
  triangle_new t{};
  vec3_to_float4(a, t.v0);
  vec3_to_float4(b, t.v1);
  vec3_to_float4(c, t.v2);
  vec3_to_float4(na, t.n0);
  vec3_to_float4(nb, t.n1);
  vec3_to_float4(nc, t.n2);
  t.mat_index = 0;
  world.push_back(t);
}

// Procedurally generate a UV sphere with approximately `target_triangles`
// triangles, for benchmarking BVH traversal vs. brute force as triangle count
// scales. The actual count (printed) is 2*res*res for an integer resolution res
// chosen to be nearest to the request.
inline scene_config load_sphere(int image_width, int image_height, int samples,
                                int target_triangles) {
  scene_config config;
  config.name = "sphere";

  const double pi = 3.14159265358979323846;
  const vec3 center(0, 1, 0);
  const double radius = 1.0;

  int res = std::max(2, (int)std::lround(std::sqrt(target_triangles / 2.0)));
  int stacks = res, slices = res;

  auto point_on_sphere = [&](double phi, double theta) {
    return vec3(std::sin(phi) * std::cos(theta), std::cos(phi),
                std::sin(phi) * std::sin(theta));
  };

  config.world.reserve((size_t)2 * stacks * slices);
  for (int i = 0; i < stacks; i++) {
    double phi0 = pi * i / stacks, phi1 = pi * (i + 1) / stacks;
    for (int j = 0; j < slices; j++) {
      double th0 = 2.0 * pi * j / slices, th1 = 2.0 * pi * (j + 1) / slices;

      vec3 n00 = point_on_sphere(phi0, th0), n01 = point_on_sphere(phi0, th1);
      vec3 n10 = point_on_sphere(phi1, th0), n11 = point_on_sphere(phi1, th1);

      vec3 p00 = center + radius * n00, p01 = center + radius * n01;
      vec3 p10 = center + radius * n10, p11 = center + radius * n11;

      push_gpu_triangle(config.world, p00, p10, p11, n00, n10, n11);
      push_gpu_triangle(config.world, p00, p11, p01, n00, n11, n01);
    }
  }

  config.cam.image_width = image_width;
  config.cam.image_height = image_height;
  config.cam.samples_per_pixel = samples;
  config.cam.max_depth = 10;
  config.cam.background = color(0.7, 0.8, 1.00);
  config.cam.vfov = 30;
  config.cam.lookfrom = point3(-2, 3, 6);
  config.cam.lookat = point3(-0.3, 1.0, 0);
  config.cam.vup = vec3(0, 1, 0);
  config.cam.defocus_angle = 0;
  config.cam.focus_dist = 3.0;

  fmt::println("Generated sphere: {} triangles ({} stacks x {} slices)",
               config.world.size(), stacks, slices);

  return config;
}
