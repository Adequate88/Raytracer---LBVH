//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

// Temporarily disable GPU rendering to test CPU triangle
// #define USE_GPU_RENDERING

#include "rtweekend.h"

#include "../bvh/lbvh_gpu.h"
#include "camera.h"
#include "constant_medium.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "mesh.h"
#include "quad.h"
#include "sphere.h"
#include "texture.h"
#include "triangle.h"

#include <chrono>


void bouncing_spheres() {
    hittable_list world;

    auto checker = make_shared<checker_texture>(0.32, color(.2, .3, .1), color(.9, .9, .9));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    auto center2 = center + vec3(0, random_double(0,.5), 0);
                    world.add(make_shared<sphere>(center, center2, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0.70, 0.80, 1.00);

    cam.vfov     = 20;
    cam.lookfrom = point3(13,2,3);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0.6;
    cam.focus_dist    = 10.0;

    lbvh_gpu world_bvh(world.objects);
    cam.render(world_bvh);
}


void checkered_spheres() {
    hittable_list world;

    auto checker = make_shared<checker_texture>(0.32, color(.2, .3, .1), color(.9, .9, .9));

    world.add(make_shared<sphere>(point3(0,-10, 0), 10, make_shared<lambertian>(checker)));
    world.add(make_shared<sphere>(point3(0, 10, 0), 10, make_shared<lambertian>(checker)));

    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0.70, 0.80, 1.00);

    cam.vfov     = 20;
    cam.lookfrom = point3(13,2,3);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;

    lbvh_gpu world_bvh(world.objects);
    cam.render(world_bvh);
}


void earth() {
    hittable_list world;

    auto earth_texture = make_shared<image_texture>("earthmap.jpg");
    auto earth_surface = make_shared<lambertian>(earth_texture);
    auto globe = make_shared<sphere>(point3(0,0,0), 2, earth_surface);
    world.add(globe);

    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0.70, 0.80, 1.00);

    cam.vfov     = 20;
    cam.lookfrom = point3(0,0,12);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;

    lbvh_gpu world_bvh(world.objects);
    cam.render(world_bvh);
}


void perlin_spheres() {
    hittable_list world;

    auto pertext = make_shared<noise_texture>(4);
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(pertext)));
    world.add(make_shared<sphere>(point3(0,2,0), 2, make_shared<lambertian>(pertext)));

    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0.70, 0.80, 1.00);

    cam.vfov     = 20;
    cam.lookfrom = point3(13,2,3);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;

    lbvh_gpu world_bvh(world.objects);

    world_bvh.enable_statistics();

    cam.render(world_bvh);

    // Print BVH statistics
    world_bvh.get_stats().print();
}


void quads() {
    hittable_list world;

    // Materials
    auto left_red     = make_shared<lambertian>(color(1.0, 0.2, 0.2));
    auto back_green   = make_shared<lambertian>(color(0.2, 1.0, 0.2));
    auto right_blue   = make_shared<lambertian>(color(0.2, 0.2, 1.0));
    auto upper_orange = make_shared<lambertian>(color(1.0, 0.5, 0.0));
    auto lower_teal   = make_shared<lambertian>(color(0.2, 0.8, 0.8));

    // Quads
    world.add(make_shared<quad>(point3(-3,-2, 5), vec3(0, 0,-4), vec3(0, 4, 0), left_red));
    world.add(make_shared<quad>(point3(-2,-2, 0), vec3(4, 0, 0), vec3(0, 4, 0), back_green));
    world.add(make_shared<quad>(point3( 3,-2, 1), vec3(0, 0, 4), vec3(0, 4, 0), right_blue));
    world.add(make_shared<quad>(point3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4), upper_orange));
    world.add(make_shared<quad>(point3(-2,-3, 5), vec3(4, 0, 0), vec3(0, 0,-4), lower_teal));

    camera cam;

    cam.aspect_ratio      = 1.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0.70, 0.80, 1.00);

    cam.vfov     = 80;
    cam.lookfrom = point3(0,0,9);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;

    lbvh_gpu world_bvh(world.objects);
    cam.render(world_bvh);
}


void simple_light() {
    hittable_list world;

    auto pertext = make_shared<noise_texture>(4);
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(pertext)));
    world.add(make_shared<sphere>(point3(0,2,0), 2, make_shared<lambertian>(pertext)));

    auto difflight = make_shared<diffuse_light>(color(4,4,4));
    world.add(make_shared<sphere>(point3(0,7,0), 2, difflight));
    world.add(make_shared<quad>(point3(3,1,-2), vec3(2,0,0), vec3(0,2,0), difflight));

    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0,0,0);

    cam.vfov     = 20;
    cam.lookfrom = point3(26,3,6);
    cam.lookat   = point3(0,2,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;

    lbvh_gpu world_bvh(world.objects);
    cam.render(world_bvh);
}


void cornell_box() {
    hittable_list world;

    auto red   = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto light = make_shared<diffuse_light>(color(15, 15, 15));

    world.add(make_shared<quad>(point3(555,0,0), vec3(0,555,0), vec3(0,0,555), green));
    world.add(make_shared<quad>(point3(0,0,0), vec3(0,555,0), vec3(0,0,555), red));
    world.add(make_shared<quad>(point3(343, 554, 332), vec3(-130,0,0), vec3(0,0,-105), light));
    world.add(make_shared<quad>(point3(0,0,0), vec3(555,0,0), vec3(0,0,555), white));
    world.add(make_shared<quad>(point3(555,555,555), vec3(-555,0,0), vec3(0,0,-555), white));
    world.add(make_shared<quad>(point3(0,0,555), vec3(555,0,0), vec3(0,555,0), white));

    shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), white);
    box1 = make_shared<rotate_y>(box1, 15);
    box1 = make_shared<translate>(box1, vec3(265,0,295));
    world.add(box1);

    shared_ptr<hittable> box2 = box(point3(0,0,0), point3(165,165,165), white);
    box2 = make_shared<rotate_y>(box2, -18);
    box2 = make_shared<translate>(box2, vec3(130,0,65));
    world.add(box2);

    camera cam;

    cam.aspect_ratio      = 1.0;
    cam.image_width       = 100;
    cam.samples_per_pixel = 200;
    cam.max_depth         = 50;
    cam.background        = color(0,0,0);

    cam.vfov     = 40;
    cam.lookfrom = point3(278, 278, -800);
    cam.lookat   = point3(278, 278, 0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;

    lbvh_gpu world_bvh(world.objects);

    // Print BVH tree structure
    world_bvh.print_tree(15);
    std::clog << "\nTree depth: " << world_bvh.get_tree_depth() << "\n\n";

    // Enable BVH statistics tracking
    world_bvh.enable_statistics();

    // Start render timing
    auto render_start = std::chrono::high_resolution_clock::now();

    cam.render(world_bvh);

    // End render timing
    auto render_end = std::chrono::high_resolution_clock::now();
    double render_time_ms = std::chrono::duration<double, std::milli>(render_end - render_start).count();

    // Print timing results
    std::clog << "\n=== Performance ===\n";
    std::clog << "BVH construction time: " << world_bvh.get_construction_time() << " ms\n";
    std::clog << "  - OpenCL initialization: " << world_bvh.get_opencl_init_time() << " ms\n";
    std::clog << "  - Morton code computation: " << world_bvh.get_morton_time() << " ms\n";
    std::clog << "  - Radix sort: " << world_bvh.get_sort_time() << " ms\n";
    std::clog << "  - Hierarchy construction: " << world_bvh.get_hierarchy_time() << " ms\n";
    std::clog << "  - Bounding box propagation: " << world_bvh.get_bbox_time() << " ms\n";
    std::clog << "Rendering time: " << render_time_ms << " ms\n";
    std::clog << "Total time: " << (world_bvh.get_construction_time() + render_time_ms) << " ms\n";

    // Print BVH statistics
    world_bvh.get_stats().print();
}


void cornell_smoke() {
    hittable_list world;

    auto red   = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto light = make_shared<diffuse_light>(color(7, 7, 7));

    world.add(make_shared<quad>(point3(555,0,0), vec3(0,555,0), vec3(0,0,555), green));
    world.add(make_shared<quad>(point3(0,0,0), vec3(0,555,0), vec3(0,0,555), red));
    world.add(make_shared<quad>(point3(113,554,127), vec3(330,0,0), vec3(0,0,305), light));
    world.add(make_shared<quad>(point3(0,555,0), vec3(555,0,0), vec3(0,0,555), white));
    world.add(make_shared<quad>(point3(0,0,0), vec3(555,0,0), vec3(0,0,555), white));
    world.add(make_shared<quad>(point3(0,0,555), vec3(555,0,0), vec3(0,555,0), white));

    shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), white);
    box1 = make_shared<rotate_y>(box1, 15);
    box1 = make_shared<translate>(box1, vec3(265,0,295));

    shared_ptr<hittable> box2 = box(point3(0,0,0), point3(165,165,165), white);
    box2 = make_shared<rotate_y>(box2, -18);
    box2 = make_shared<translate>(box2, vec3(130,0,65));

    world.add(make_shared<constant_medium>(box1, 0.01, color(0,0,0)));
    world.add(make_shared<constant_medium>(box2, 0.01, color(1,1,1)));

    camera cam;

    cam.aspect_ratio      = 1.0;
    cam.image_width       = 600;
    cam.samples_per_pixel = 200;
    cam.max_depth         = 50;
    cam.background        = color(0,0,0);

    cam.vfov     = 40;
    cam.lookfrom = point3(278, 278, -800);
    cam.lookat   = point3(278, 278, 0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;

    lbvh_gpu world_bvh(world.objects);
    cam.render(world_bvh);
}


void final_scene(int image_width, int samples_per_pixel, int max_depth) {
    hittable_list boxes1;
    auto ground = make_shared<lambertian>(color(0.48, 0.83, 0.53));

    int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0;
            auto x0 = -1000.0 + i*w;
            auto z0 = -1000.0 + j*w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = random_double(1,101);
            auto z1 = z0 + w;

            boxes1.add(box(point3(x0,y0,z0), point3(x1,y1,z1), ground));
        }
    }

    hittable_list world;

    // Add all boxes from boxes1 to world
    for (auto& box : boxes1.objects) {
        world.add(box);
    }

    auto light = make_shared<diffuse_light>(color(7, 7, 7));
    world.add(make_shared<quad>(point3(123,554,147), vec3(300,0,0), vec3(0,0,265), light));

    auto center1 = point3(400, 400, 200);
    auto center2 = center1 + vec3(30,0,0);
    auto sphere_material = make_shared<lambertian>(color(0.7, 0.3, 0.1));
    world.add(make_shared<sphere>(center1, center2, 50, sphere_material));

    world.add(make_shared<sphere>(point3(260, 150, 45), 50, make_shared<dielectric>(1.5)));
    world.add(make_shared<sphere>(
        point3(0, 150, 145), 50, make_shared<metal>(color(0.8, 0.8, 0.9), 1.0)
    ));

    auto boundary = make_shared<sphere>(point3(360,150,145), 70, make_shared<dielectric>(1.5));
    world.add(boundary);
    world.add(make_shared<constant_medium>(boundary, 0.2, color(0.2, 0.4, 0.9)));
    boundary = make_shared<sphere>(point3(0,0,0), 5000, make_shared<dielectric>(1.5));
    world.add(make_shared<constant_medium>(boundary, .0001, color(1,1,1)));

    auto emat = make_shared<lambertian>(make_shared<image_texture>("earthmap.jpg"));
    world.add(make_shared<sphere>(point3(400,200,400), 100, emat));
    auto pertext = make_shared<noise_texture>(0.2);
    world.add(make_shared<sphere>(point3(220,280,300), 80, make_shared<lambertian>(pertext)));

    auto boxes2_list = make_shared<hittable_list>();
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    int ns = 1000;
    for (int j = 0; j < ns; j++) {
        boxes2_list->add(make_shared<sphere>(point3::random(0,165), 10, white));
    }

    world.add(make_shared<translate>(
        make_shared<rotate_y>(
            boxes2_list, 15),
            vec3(-100,270,395)
        )
    );

    camera cam;

    cam.aspect_ratio      = 1.0;
    cam.image_width       = image_width;
    cam.samples_per_pixel = samples_per_pixel;
    cam.max_depth         = max_depth;
    cam.background        = color(0,0,0);

    cam.vfov     = 40;
    cam.lookfrom = point3(478, 278, -600);
    cam.lookat   = point3(278, 278, 0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;

    lbvh_gpu world_bvh(world.objects);
    cam.render(world_bvh);
}


void bvh_stress_test() {
    hittable_list world;

    std::clog << "\n=== BVH Stress Test Scene ===\n";
    std::clog << "Creating randomly scattered spheres...\n";

    // Create many random spheres - compact for Phase 2 testing
    int num_spheres = 5;

    for (int i = 0; i < num_spheres; i++) {
        auto choose_mat = random_double();
        point3 center(random_double(-25, 25), random_double(0.5, 10), random_double(-20, 20));
        auto radius = random_double(0.5, 5);

        shared_ptr<material> sphere_material;

        if (choose_mat < 0.3) {
            // diffuse (Lambertian)
            auto albedo = color::random() * color::random();
            sphere_material = make_shared<lambertian>(albedo);
        } else if (choose_mat < 0.9) {
            // metal (reflective)
            auto albedo = color::random(0.5, 1);
            auto fuzz = random_double(0, 0.5);
            sphere_material = make_shared<metal>(albedo, fuzz);
        } else {
            // light (emissive) - testing Phase 2 lights
            auto emit = color::random(1.0, 3.0);
            sphere_material = make_shared<diffuse_light>(emit);
        }

        world.add(make_shared<sphere>(center, radius, sphere_material));
    }

    std::clog << "Total primitives created: " << world.objects.size() << "\n";
    std::clog << "==============================\n\n";

    // Add ground
    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    // Add main light above scene
    auto light = make_shared<diffuse_light>(color(8, 8, 8));
    world.add(make_shared<sphere>(point3(0, 20, 0), 5, light));

    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 600;  // Higher resolution
    cam.samples_per_pixel = 1000;  // More samples for quality
    cam.max_depth         = 50;    // Test multi-bounce
    cam.background        = color(0.05, 0.05, 0.10);  // Dark blue background

    cam.vfov     = 40;
    cam.lookfrom = point3(26, 12, 8);
    cam.lookat   = point3(0, 5, 0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;

    // Start BVH construction timing
    std::clog << "\n=== BVH Construction (GPU-Accelerated) ===\n";
    auto bvh_start = std::chrono::high_resolution_clock::now();

    lbvh_gpu world_bvh(world.objects);

    auto bvh_end = std::chrono::high_resolution_clock::now();
    double bvh_time_ms = std::chrono::duration<double, std::milli>(bvh_end - bvh_start).count();

    std::clog << "\n=== Phase-by-Phase Breakdown ===\n";
    std::clog << "Total BVH construction: " << bvh_time_ms << " ms\n";
    std::clog << "  - OpenCL initialization: " << world_bvh.get_opencl_init_time() << " ms\n";
    std::clog << "  - Morton code computation (GPU): " << world_bvh.get_morton_time() << " ms\n";
    std::clog << "  - Radix sort (CPU): " << world_bvh.get_sort_time() << " ms\n";
    std::clog << "  - Hierarchy construction (GPU): " << world_bvh.get_hierarchy_time() << " ms\n";
    std::clog << "  - Bounding box propagation (CPU): " << world_bvh.get_bbox_time() << " ms\n";

    double algorithmic_time = world_bvh.get_morton_time() + world_bvh.get_sort_time() +
                             world_bvh.get_hierarchy_time() + world_bvh.get_bbox_time();
    std::clog << "\nAlgorithmic work (excluding OpenCL init): " << algorithmic_time << " ms\n";
    std::clog << "Tree depth: " << world_bvh.get_tree_depth() << "\n";
    std::clog << "======================================\n\n";

    // Enable statistics
    world_bvh.enable_statistics();

    // Render (minimal resolution for speed)
    std::clog << "Starting render...\n";
    auto render_start = std::chrono::high_resolution_clock::now();
    cam.render(world_bvh);
    auto render_end = std::chrono::high_resolution_clock::now();
    double render_time_ms = std::chrono::duration<double, std::milli>(render_end - render_start).count();

    std::clog << "\n=== Final Performance Summary ===\n";
    std::clog << "BVH construction: " << bvh_time_ms << " ms\n";
    std::clog << "Rendering time: " << render_time_ms << " ms\n";
    std::clog << "Total time: " << (bvh_time_ms + render_time_ms) << " ms\n";
    std::clog << "=================================\n";

    // Print BVH statistics
    world_bvh.get_stats().print();
}


void single_triangle_test() {
    hittable_list world;

    // Single triangle (red)
    auto tri_mat = make_shared<lambertian>(color(0.8, 0.2, 0.2));
    auto tri = make_shared<triangle>(
        point3(-1, 0, -1),
        point3( 1, 0, -1),
        point3( 0, 2, -1),
        tri_mat
    );
    world.add(tri);

    // Ground sphere
    auto ground_mat = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_mat));

    // Light source
    auto light = make_shared<diffuse_light>(color(4, 4, 4));
    world.add(make_shared<sphere>(point3(3, 5, 3), 1.5, light));

    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 600;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0.7, 0.8, 1.0);

    cam.vfov     = 40;
    cam.lookfrom = point3(0, 2, 5);
    cam.lookat   = point3(0, 1, -1);
    cam.vup      = vec3(0, 1, 0);

    cam.defocus_angle = 0;

    lbvh_gpu world_bvh(world.objects);
    cam.render(world_bvh);
}


void mesh_test_scene() {
    hittable_list world;

    // Load cube mesh
    auto mesh_mat = make_shared<lambertian>(color(0.7, 0.3, 0.2));
    auto mesh_triangles = mesh::load_obj("models/cube.obj", mesh_mat, true, 1.0);

    // Add triangles directly (GPU renderer doesn't support translate wrapper yet)
    for (auto& tri : mesh_triangles->objects) {
        world.add(tri);
    }

    // Add ground plane
    auto ground_mat = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_mat));

    // Add light source
    auto light = make_shared<diffuse_light>(color(4, 4, 4));
    world.add(make_shared<sphere>(point3(3, 5, 3), 1.5, light));

    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 600;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0.7, 0.8, 1.0);

    cam.vfov     = 40;
    cam.lookfrom = point3(4, 3, 6);
    cam.lookat   = point3(0, 1, 0);
    cam.vup      = vec3(0, 1, 0);

    cam.defocus_angle = 0;

    lbvh_gpu world_bvh(world.objects);
    cam.render(world_bvh);
}


void dragon_scene() {
    hittable_list world;

    // Load Stanford Dragon (high resolution)
    auto dragon_mat = make_shared<metal>(color(0.8, 0.6, 0.2), 0.1);
    std::clog << "Loading Stanford Dragon..." << std::endl;
    auto dragon_triangles = mesh::load_obj("models/dragon.obj", dragon_mat, true, 1.0);

    // Add triangles directly (GPU renderer doesn't support translate wrapper yet)
    for (auto& tri : dragon_triangles->objects) {
        world.add(tri);
    }


    // Add multiple light sources for better illumination
    auto light1 = make_shared<diffuse_light>(color(4, 4, 4));
    world.add(make_shared<sphere>(point3(5, 8, 5), 2.0, light1));

    auto light2 = make_shared<diffuse_light>(color(2, 2, 2));
    world.add(make_shared<sphere>(point3(-5, 6, -3), 1.5, light2));

    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 100;
    cam.samples_per_pixel = 10;
    cam.max_depth         = 5;
    cam.background        = color(0.02, 0.02, 0.02);  // Dark background

    cam.vfov     = 40;  // Wider FOV for closer view
    cam.lookfrom = point3(-1, 0.3, -1);  // Much closer, slightly above
    cam.lookat   = point3(0, 0, 0);  // Looking at dragon's center
    cam.vup      = vec3(0, 1, 0);

    cam.defocus_angle = 0.0;  // Slight depth of field for cinematic look

    // Start BVH construction timing
    std::clog << "\n=== BVH Construction (GPU-Accelerated) ===\n";
    auto bvh_start = std::chrono::high_resolution_clock::now();

    lbvh_gpu world_bvh(world.objects);

    auto bvh_end = std::chrono::high_resolution_clock::now();
    double bvh_time_ms = std::chrono::duration<double, std::milli>(bvh_end - bvh_start).count();

    std::clog << "\n=== Phase-by-Phase Breakdown ===\n";
    std::clog << "Total BVH construction: " << bvh_time_ms << " ms\n";
    std::clog << "  - OpenCL initialization: " << world_bvh.get_opencl_init_time() << " ms\n";
    std::clog << "  - Morton code computation (GPU): " << world_bvh.get_morton_time() << " ms\n";
    std::clog << "  - Radix sort (CPU): " << world_bvh.get_sort_time() << " ms\n";
    std::clog << "  - Hierarchy construction (GPU): " << world_bvh.get_hierarchy_time() << " ms\n";
    std::clog << "  - Bounding box propagation (CPU): " << world_bvh.get_bbox_time() << " ms\n";

    double algorithmic_time = world_bvh.get_morton_time() + world_bvh.get_sort_time() +
                             world_bvh.get_hierarchy_time() + world_bvh.get_bbox_time();
    std::clog << "\nAlgorithmic work (excluding OpenCL init): " << algorithmic_time << " ms\n";
    std::clog << "Tree depth: " << world_bvh.get_tree_depth() << "\n";
    std::clog << "======================================\n\n";

    // Enable statistics
    world_bvh.enable_statistics();

    cam.render(world_bvh);
}


int main() {
    switch (14) {
        case 1:  bouncing_spheres();          break;
        case 2:  checkered_spheres();         break;
        case 3:  earth();                     break;
        case 4:  perlin_spheres();            break;
        case 5:  quads();                     break;
        case 6:  simple_light();              break;
        case 7:  cornell_box();               break;
        case 8:  cornell_smoke();             break;
        case 9:  final_scene(800, 10000, 40); break;
        case 11: bvh_stress_test();           break;
        case 12: single_triangle_test();      break;
        case 13: mesh_test_scene();           break;
        case 14: dragon_scene();              break;
        default: final_scene(400,   250,  4); break;
    }
}
