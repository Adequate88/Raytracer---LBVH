#include "camera.h"
#include "metrics_macros.h"
#include "vec3.h"
#include <iostream>

void camera::render(LBVH &world) {
  METRIC_START_TIME("Total Rendering Time");
  for (int j = 0; j < image_height; j++) {
    for (int i = 0; i < image_width; i++) {
      color pixel_color(0, 0, 0);
      for (int sample = 0; sample < samples_per_pixel; sample++) {
        ray r = get_ray(i, j);
        pixel_color += ray_color(r, max_depth, world);
      }
      write_color(img.data, pixel_samples_scale * pixel_color);
    }
  }
  METRIC_END_TIME("Total Rendering Time");
}

int camera::diagnose_bvh(
    LBVH &world,
    const std::vector<shared_ptr<hittable>> &objects) const {
  // (a) Structural check: any internal node left with an inverted/empty bbox
  // (min > max) gets culled by AABB::hit, silently dropping its whole subtree.
  const auto &nodes = world.get_nodes();
  int inverted = 0;
  for (const auto &n : nodes) {
    if (n.bbox.min_x > n.bbox.max_x || n.bbox.min_y > n.bbox.max_y ||
        n.bbox.min_z > n.bbox.max_z)
      inverted++;
  }
  std::clog << "[diag] inverted/empty node bboxes: " << inverted << " / "
            << nodes.size() << "\n";

  // (b) Ground-truth check: brute-force nearest-hit vs BVH nearest-hit, one
  // deterministic center ray per pixel.
  int bf_hits = 0, mismatches = 0;
  int min_i = image_width, max_i = -1, min_j = image_height, max_j = -1;
  for (int j = 0; j < image_height; j++) {
    for (int i = 0; i < image_width; i++) {
      auto pixel_center =
          pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
      ray r(center, pixel_center - center, 0.0);

      hit_record bf;
      bool bf_hit = false;
      double closest = infinity;
      for (const auto &o : objects) {
        hit_record tmp;
        if (o->hit(r, interval(0.001, closest), tmp)) {
          bf_hit = true;
          closest = tmp.t;
          bf = tmp;
        }
      }

      hit_record bv;
      bool bv_hit = world.hit(r, interval(0.001, infinity), bv);

      if (bf_hit) {
        bf_hits++;
        if (!bv_hit || std::fabs(bv.t - closest) > 1e-4) {
          mismatches++;
          min_i = std::min(min_i, i);
          max_i = std::max(max_i, i);
          min_j = std::min(min_j, j);
          max_j = std::max(max_j, j);
        }
      }
    }
  }

  std::clog << "[diag] pixels where brute-force hit geometry: " << bf_hits
            << "\n";
  std::clog << "[diag] pixels where BVH missed/differed: " << mismatches
            << "\n";
  if (mismatches > 0) {
    std::clog << "[diag] mismatch bounding box (image coords): x[" << min_i
              << "," << max_i << "] y[" << min_j << "," << max_j << "]\n";
  }
  return mismatches;
}

void camera::initialize() {

  aspect_ratio = double(image_width) / image_height;

  img.width = image_width;
  img.height = image_height;
  img.data.reserve(img.width * img.height * 4);

  pixel_samples_scale = 1.0 / samples_per_pixel;

  center = lookfrom;

  auto theta = degrees_to_radians(vfov);
  auto h = std::tan(theta / 2);
  auto viewport_height = 2 * h * focus_dist;
  auto viewport_width = viewport_height * (double(image_width) / image_height);

  w = unit_vector(lookfrom - lookat);
  u = unit_vector(cross(vup, w));
  v = cross(w, u);

  vec3 viewport_u = viewport_width * u;
  vec3 viewport_v = viewport_height * -v;

  pixel_delta_u = viewport_u / image_width;
  pixel_delta_v = viewport_v / image_height;

  auto viewport_upper_left =
      center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
  pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

  auto defocus_radius =
      focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
  defocus_disk_u = u * defocus_radius;
  defocus_disk_v = v * defocus_radius;

  gpu_constants = {};
  vec3_to_float4(center, gpu_constants.center);
  vec3_to_float4(pixel00_loc, gpu_constants.pixel00_loc);
  vec3_to_float4(pixel_delta_u, gpu_constants.pixel_delta_u);
  vec3_to_float4(pixel_delta_v, gpu_constants.pixel_delta_v);
}

ray camera::get_ray(int i, int j) const {
  // Construct a camera ray originating from the defocus disk and directed at a
  // randomly sampled point around the pixel location i, j.

  auto offset = sample_square();
  auto pixel_sample = pixel00_loc + ((i + offset.x()) * pixel_delta_u) +
                      ((j + offset.y()) * pixel_delta_v);

  auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
  auto ray_direction = pixel_sample - ray_origin;
  auto ray_time = random_double();

  return ray(ray_origin, ray_direction, ray_time);
}

vec3 camera::sample_square() const {
  // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit
  // square.
  return vec3(random_double() - 0.5, random_double() - 0.5, 0);
}

vec3 camera::sample_disk(double radius) const {
  // Returns a random point in the unit (radius 0.5) disk centered at the
  // origin.
  return radius * random_in_unit_disk();
}

point3 camera::defocus_disk_sample() const {
  // Returns a random point in the camera defocus disk.
  auto p = random_in_unit_disk();
  return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
}

color camera::ray_color(const ray &r, int depth, const LBVH &world) const {
  // If we've exceeded the ray bounce limit, no more light is gathered.
  if (depth <= 0)
    return color(0, 0, 0);

  hit_record rec;

  // If the ray hits nothing, return the background color.
  if (!world.hit(r, interval(0.001, infinity), rec))
    return background;

  ray scattered;
  color attenuation;
  color color_from_emission = rec.mat->emitted(rec.u, rec.v, rec.p);

  if (!rec.mat->scatter(r, rec, attenuation, scattered))
    return color_from_emission;

  color color_from_scatter =
      attenuation * ray_color(scattered, depth - 1, world);

  return color_from_emission + color_from_scatter;
}
