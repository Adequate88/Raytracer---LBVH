#pragma once

#include "hittable.h"
#include "lbvh_gpu.h"
#include "material.h"

struct image {
  int width, height;
  std::vector<uint8_t> data;
};

class camera {
public:
  double aspect_ratio = 1.0;
  int image_width = 100;
  int image_height = int(image_width / aspect_ratio);

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

  void render(LBVH &world);

private:
  double pixel_samples_scale;
  point3 center;
  point3 pixel00_loc;
  vec3 pixel_delta_u;
  vec3 pixel_delta_v;
  vec3 u, v, w;
  vec3 defocus_disk_u;
  vec3 defocus_disk_v;

  void initialize();

  ray get_ray(int i, int j) const;

  vec3 sample_square() const;

  vec3 sample_disk(double radius) const;

  point3 defocus_disk_sample() const;

  color ray_color(const ray &r, int depth, const LBVH &world) const;
};
