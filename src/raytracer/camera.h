#pragma once

#include "hittable.h"
#include "material.h"
#include "lbvh_gpu.h"

class camera {
  public:
    double aspect_ratio      = 1.0;
    int    image_width       = 100;
    int    samples_per_pixel = 10;
    int    max_depth         = 10;
    color  background;

    double vfov     = 90;
    point3 lookfrom = point3(0,0,0);
    point3 lookat   = point3(0,0,-1);
    vec3   vup      = vec3(0,1,0);

    double defocus_angle = 0;
    double focus_dist = 10;

    void render(LBVH& world);


  private:
    int    image_height;
    double pixel_samples_scale;
    point3 center;
    point3 pixel00_loc;
    vec3   pixel_delta_u;
    vec3   pixel_delta_v;
    vec3   u, v, w;
    vec3   defocus_disk_u;
    vec3   defocus_disk_v;



    void initialize();

    ray get_ray(int i, int j) const;

    vec3 sample_square() const;

    vec3 sample_disk(double radius) const;

    point3 defocus_disk_sample() const;

    template<typename BVH>
    color ray_color(const ray& r, int depth, const BVH& world) const;
};

