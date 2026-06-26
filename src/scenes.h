#pragma once

#include "camera.h"
#include "hittable_list.h"
#include "mesh.h"
#include "metrics_macros.h"
#include "quad.h"
#include "sphere.h"
#include "triangle.h"
#include "types.h"
#include <fmt/base.h>

const int SAMPLES = 10;
const std::string MODELS_DIR = "models/";

struct scene_config {
  std::vector<triangle_new> world;
  camera cam;
  std::string name;
};

inline std::vector<triangle_new> convert_for_gpu(const hittable_list &mesh) {
  std::vector<triangle_new> new_mesh;
  new_mesh.reserve(mesh.objects.size());
  for (const auto &obj : mesh.objects) {
    auto tri = std::dynamic_pointer_cast<triangle>(obj);
    if (!tri) {
      fmt::println("Non-triangle exists, beware");
      continue;
    }
    triangle_new gpu_tri{};
    vec3_to_float4(tri->get_v0(), gpu_tri.v0);
    vec3_to_float4(tri->get_v1(), gpu_tri.v1);
    vec3_to_float4(tri->get_v2(), gpu_tri.v2);

    vec3_to_float4(tri->get_n0(), gpu_tri.n0);
    vec3_to_float4(tri->get_n1(), gpu_tri.n1);
    vec3_to_float4(tri->get_n2(), gpu_tri.n2);

    vec2_to_float2(tri->get_uv0(), gpu_tri.uv0);
    vec2_to_float2(tri->get_uv1(), gpu_tri.uv1);
    vec2_to_float2(tri->get_uv2(), gpu_tri.uv2);

    gpu_tri.mat_index = 0; // TODO ADD PROPER MATERIAL INDEXING

    new_mesh.push_back(gpu_tri);
  }

  return new_mesh;
}
scene_config load_conference(int image_width, int image_height, int samples) {
  scene_config config;
  config.name = "conference";

  auto mat = make_shared<lambertian>(color(0.4, 0.6, 0.4));
  auto mesh_triangles =
      mesh::load_obj(MODELS_DIR + "conference.obj", mat, true, 10.0);

  config.world = convert_for_gpu(*mesh_triangles);

  // Camera setup for bunny
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

  return config;
}

scene_config load_bunny(int image_width, int image_height, int samples) {
  scene_config config;
  config.name = "bunny";

  auto mat = make_shared<lambertian>(color(0.4, 0.6, 0.4));
  auto mesh_triangles =
      mesh::load_obj(MODELS_DIR + "bunny.obj", mat, true, 1.0);

  config.world = convert_for_gpu(*mesh_triangles);

  // Camera setup for bunny
  config.cam.image_width = image_width;
  config.cam.image_height = image_height;
  config.cam.samples_per_pixel = 10;
  config.cam.max_depth = 20;
  config.cam.background = color(0.7, 0.8, 1.00);
  config.cam.vfov = 30;
  config.cam.lookfrom = point3(-2, 3, 6);
  config.cam.lookat = point3(-0.3, 1.0, 0);
  config.cam.vup = vec3(0, 1, 0);
  config.cam.defocus_angle = 0;
  config.cam.focus_dist = 3.0;

  return config;
}

scene_config load_teapot(int image_width, int image_height, int samples) {
  scene_config config;
  config.name = "teapot";

  auto mat = make_shared<lambertian>(color(0.4, 0.6, 0.4));
  auto mesh_triangles =
      mesh::load_obj(MODELS_DIR + "teapot.obj", mat, true, 0.5);

  config.world = convert_for_gpu(*mesh_triangles);

  // Camera setup for bunny
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

  return config;
}
