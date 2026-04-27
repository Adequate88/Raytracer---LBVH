#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "material.h"
#include "gpu_renderer.h"

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

    std::string scene_name_for_stats = "";

    template<typename BVH>
    void render(BVH& world) {
        initialize();

        camera_params cam;
        cam.width = image_width;
        cam.height = image_height;
        cam.samples = samples_per_pixel;
        cam.max_depth = max_depth;
        cam.center_x = center.x();
        cam.center_y = center.y();
        cam.center_z = center.z();
        cam.pixel00_x = pixel00_loc.x();
        cam.pixel00_y = pixel00_loc.y();
        cam.pixel00_z = pixel00_loc.z();
        cam.delta_u_x = pixel_delta_u.x();
        cam.delta_u_y = pixel_delta_u.y();
        cam.delta_u_z = pixel_delta_u.z();
        cam.delta_v_x = pixel_delta_v.x();
        cam.delta_v_y = pixel_delta_v.y();
        cam.delta_v_z = pixel_delta_v.z();
        cam.defocus_u_x = defocus_disk_u.x();
        cam.defocus_u_y = defocus_disk_u.y();
        cam.defocus_u_z = defocus_disk_u.z();
        cam.defocus_v_x = defocus_disk_v.x();
        cam.defocus_v_y = defocus_disk_v.y();
        cam.defocus_v_z = defocus_disk_v.z();
        cam.defocus_angle = defocus_angle;
        cam.bg_r = background.x();
        cam.bg_g = background.y();
        cam.bg_b = background.z();

        gpu_renderer<BVH> renderer(world);
        std::vector<color> pixels;
        renderer.render(cam, pixels);

        last_total_rays = renderer.get_total_rays();
        last_avg_box_tests = renderer.get_avg_box_tests();
        last_avg_prim_tests = renderer.get_avg_prim_tests();

        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
        for (const auto& pixel : pixels) {
            write_color(std::cout, pixel);
        }

        std::clog << "\rDone.\n";
    }

    long long get_total_rays() const { return last_total_rays; }
    double get_avg_box_tests() const { return last_avg_box_tests; }
    double get_avg_prim_tests() const { return last_avg_prim_tests; }

    int get_image_height() const { return image_height; }
    point3 get_center() const { return center; }
    point3 get_pixel00_loc() const { return pixel00_loc; }
    vec3 get_pixel_delta_u() const { return pixel_delta_u; }
    vec3 get_pixel_delta_v() const { return pixel_delta_v; }
    int get_samples_per_pixel() const { return samples_per_pixel; }
    int get_max_depth() const { return max_depth; }
    vec3 get_defocus_disk_u() const { return defocus_disk_u; }
    vec3 get_defocus_disk_v() const { return defocus_disk_v; }
    double get_defocus_angle() const { return defocus_angle; }

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

    long long last_total_rays = 0;
    double last_avg_box_tests = 0.0;
    double last_avg_prim_tests = 0.0;

    void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = lookfrom;

        auto theta = degrees_to_radians(vfov);
        auto h = std::tan(theta/2);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width = viewport_height * (double(image_width)/image_height);

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        vec3 viewport_u = viewport_width * u;
        vec3 viewport_v = viewport_height * -v;

        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }
};

#endif
