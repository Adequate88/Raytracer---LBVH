#pragma once

#include "hittable.h"
#include "lbvh_gpu.h"
#include "material.h"

struct image {
  int width, height;
  std::vector<uint8_t> data;
};

struct camera_push_constants {
  float center[4]; // 16 bytes
  float pixel00_loc[4];
  float pixel_delta_u[4];
  float pixel_delta_v[4]; // 16 * 4 = 64 bytes, perfect for push constant

  // depth can be hardcoded for now
  // defocus is an additional item we can add later
};

static_assert(sizeof(camera_push_constants) ==
              64); // Currently hardcoded in pipeline to 64, so make sure to fix
                   // if struct is changed

class camera {
public:
  double aspect_ratio = 1.0;
  int image_width = 1600;
  int image_height = 1080;

  int samples_per_pixel = 10;
  int max_depth = 10;
  color background;

  image img;

  double vfov = 90;
  point3 lookfrom = point3(0, 0, 0);
  point3 lookat = point3(0, 0, -1);
  vec3 vup = vec3(0, 1, 0);

  double defocus_angle = 0;
  double focus_dist = 10;

  camera_push_constants gpu_constants;

  void initialize();
  void render(LBVH &world);

  // DIAGNOSTIC (temporary): compare the BVH against a brute-force linear scan
  // over every triangle, one deterministic ray per pixel. Reports pixels where
  // brute force hits geometry but the BVH misses it (or returns a different
  // nearest t) -- i.e. holes caused by culled/empty subtrees.
  int diagnose_bvh(LBVH &world,
                   const std::vector<shared_ptr<hittable>> &objects) const;

private:
  double pixel_samples_scale;
  point3 center;
  point3 pixel00_loc;
  vec3 pixel_delta_u;
  vec3 pixel_delta_v;
  vec3 u, v, w;
  vec3 defocus_disk_u;
  vec3 defocus_disk_v;

  ray get_ray(int i, int j) const;

  vec3 sample_square() const;

  vec3 sample_disk(double radius) const;

  point3 defocus_disk_sample() const;

  color ray_color(const ray &r, int depth, const LBVH &world) const;
};
