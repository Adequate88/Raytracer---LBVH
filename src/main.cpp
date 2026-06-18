// =============================================================================
// TEMPORARY benchmark driver (NOT part of the project).
// Runs the ORIGINAL "OpenCL BVH + CPU raytracer" pipeline that already lives in
// this repo (LBVH = lbvh_gpu.cpp, camera::render = camera.cpp) on the SAME scene
// and SAME window size as the current Vulkan run: teapot.obj, 600x600, 20 spp.
//
// It emits EXACTLY the same 13 metric rows the Vulkan/main version emits (same
// names, same definitions, no extras), and writes the rendered image so it can
// be compared visually against the Vulkan output. Lives on its own branch
// (original-opencl-benchmark); outputs are written OUTSIDE the repo.
// =============================================================================
#include "camera.h"   // camera::render(LBVH&)  -> "Total Rendering Time"
#include "lbvh_gpu.h" // LBVH (OpenCL build)    -> kernel-group metrics + getters
#include "material.h"
#include "mesh.h"
#include "metrics_macros.h"
#include "types.h"
#include <cstdint>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
  // --- Scene: identical to scenes.h:load_teapot(600, 600, 20) ---------------
  auto mat = make_shared<lambertian>(color(0.4, 0.6, 0.4));
  auto mesh_triangles = mesh::load_obj("models/teapot.obj", mat, true, 0.5);
  auto &objects = mesh_triangles->objects;

  camera cam;
  cam.image_width = 600;
  cam.image_height = 600;
  cam.samples_per_pixel = 20;
  cam.max_depth = 10;
  cam.background = color(0.7, 0.8, 1.00);
  cam.vfov = 30;
  cam.lookfrom = point3(-2, 3, 6);
  cam.lookat = point3(-0.3, 1.0, 0);
  cam.vup = vec3(0, 1, 0);
  cam.defocus_angle = 0;
  cam.focus_dist = 3.0;
  cam.initialize();

  // --- Single build: capture the single-shot init metrics -------------------
  // (Vulkan emits these once, without stddev: Vulkan Device Init, BVH
  // Initialization, Total BVH Construction Time.)
  LBVH bvh(objects);
  METRIC_SET_VALUE("Vulkan Device Init",
                   static_cast<float>(bvh.get_device_init_ms()));
  METRIC_SET_VALUE("BVH Initialization",
                   static_cast<float>(bvh.get_bvh_init_ms()));
  METRIC_SET_VALUE("Total BVH Construction Time",
                   static_cast<float>(bvh.get_construction_ms()));

  // --- Save one clean render to PNG for visual comparison -------------------
  // camera::render appends to img.data (BGRA), so clear first to keep exactly
  // one frame, then swap B<->R for stb's RGBA expectation.
  cam.img.data.clear();
  cam.render(bvh);
  {
    std::vector<uint8_t> rgba(cam.img.data.size());
    for (size_t p = 0; p + 3 < cam.img.data.size(); p += 4) {
      rgba[p + 0] = cam.img.data[p + 2]; // R
      rgba[p + 1] = cam.img.data[p + 1]; // G
      rgba[p + 2] = cam.img.data[p + 0]; // B
      rgba[p + 3] = cam.img.data[p + 3]; // A
    }
    stbi_write_png("/home/growlithe/Portfolio/original_opencl_cpu_render.png",
                   cam.image_width, cam.image_height, 4, rgba.data(),
                   cam.image_width * 4);
  }

  METRIC_SET_VALUE("Ray Count", static_cast<float>(600 * 600 * 20));

  // --- Sampled CPU render: "Total Rendering Time" + "Rays traced per second" -
  // (Vulkan samples the render; fewer iterations here since each CPU frame is
  // ~50 s. "Total Rendering Time" is emitted by camera::render.)
  METRIC_BENCHMARK(3, 1, {
    cam.render(bvh);
    METRIC_SET_VALUE("Rays traced per second",
                     METRIC_READ("Ray Count") /
                         (METRIC_READ("Total Rendering Time") * ms_scale));
  });

  // --- Sampled BVH construction: the 6 device kernel-group metrics ----------
  // (Mirrors the Vulkan METRIC_BENCHMARK over bvh.build(); the LBVH ctor emits
  // the grouped kernel times, so each rebuild produces one sample.)
  METRIC_BENCHMARK(50, 5, { LBVH sample_bvh(objects); });

  // Total Run Time, same formula as the Vulkan main.
  METRIC_SET_VALUE("Total Run Time",
                   METRIC_READ("Vulkan Device Init") +
                       METRIC_READ("Total BVH Construction Time") +
                       METRIC_READ("Total Rendering Time"));

  METRIC_EXPORT("/home/growlithe/Portfolio/original_opencl_cpu_metrics.csv");
  return 0;
}
