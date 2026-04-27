#ifndef LBVH_KARRAS_GPU_H
#define LBVH_KARRAS_GPU_H

#include "morton.h"
#include "bvh.h"
#include "../raytracer/aabb_refactored.h"
#include "../raytracer/hittable.h"
#include "../raytracer/hittable_list.h"
#include "../raytracer/sphere.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

struct int2 {
  int x, y;
  int2() : x(0), y(0) {}
  int2(int x, int y) : x(x), y(y) {}
};

struct gpu_float3 {
    float x, y, z;
    gpu_float3() : x(0), y(0), z(0) {}
    gpu_float3(float x, float y, float z) : x(x), y(y), z(z) {}
    gpu_float3(const point3& p) : x((float)p.x()), y((float)p.y()), z((float)p.z()) {}
};

struct lbvh_node {
    aabb bbox;
    int left;
    int right;
    int parent;
    int primitive_id;
    int atomic_counter;
    int _pad;  // Padding to ensure 48-byte alignment matches OpenCL

    lbvh_node() : left(-1), right(-1), parent(-1), primitive_id(-1), atomic_counter(0), _pad(0) {}
    lbvh_node(const aabb& bbox, int prim_id) : bbox(bbox), left(-1), right(-1), parent(-1), primitive_id(prim_id), atomic_counter(0), _pad(0) {}
    lbvh_node(int l, int r) : left(l), right(r), parent(-1), primitive_id(-1), atomic_counter(0), _pad(0) {}
    bool is_leaf() const { return primitive_id >= 0; }
};

class lbvh_karras_gpu {
  public:
    ~lbvh_karras_gpu() { cleanup_opencl(); }

    lbvh_karras_gpu(std::vector<shared_ptr<hittable>>& objects)
        : objects(objects), N(objects.size()) {
        nodes.resize(2*N - 1);
        auto start_time = std::chrono::high_resolution_clock::now();

        if (N == 0) return;

        auto phase_start = std::chrono::high_resolution_clock::now();
        init_opencl();
        auto phase_end = std::chrono::high_resolution_clock::now();
        opencl_init_time_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();

        phase_start = std::chrono::high_resolution_clock::now();
        morton_list.resize(N);
        cell_count = 1 << k;

        temp_scene_bbox = aabb::empty;
        for (int i = 0; i < N; i++) {
            temp_scene_bbox = aabb(temp_scene_bbox, objects[i]->bounding_box());
            primitive_bboxes.push_back(objects[i]->bounding_box());
            centroids.push_back(objects[i]->get_centroid());
        }

        std::vector<gpu_float3> gpu_centroids(N);
        for (int i = 0; i < N; i++) {
            gpu_centroids[i] = gpu_float3(centroids[i]);
        }

        cl_int err;
        buf_centroids = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 N * sizeof(gpu_float3), gpu_centroids.data(), &err);
        check_cl_error(err, "clCreateBuffer(buf_centroids)");

        buf_bboxes = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        N * sizeof(aabb), primitive_bboxes.data(), &err);
        check_cl_error(err, "clCreateBuffer(buf_bboxes)");

        buf_morton_list = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   N * sizeof(morton_primitive), nullptr, &err);
        check_cl_error(err, "clCreateBuffer(buf_morton_list)");

        buf_nodes = clCreateBuffer(context, CL_MEM_READ_WRITE,
                             (2*N - 1) * sizeof(lbvh_node), nullptr, &err);
        check_cl_error(err, "clCreateBuffer(buf_nodes)");
        phase_end = std::chrono::high_resolution_clock::now();
        data_prep_time_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();

        phase_start = std::chrono::high_resolution_clock::now();
        err = clSetKernelArg(kernel_morton, 0, sizeof(cl_mem), &buf_centroids);
        err |= clSetKernelArg(kernel_morton, 1, sizeof(aabb), &temp_scene_bbox);
        err |= clSetKernelArg(kernel_morton, 2, sizeof(int), &N);
        err |= clSetKernelArg(kernel_morton, 3, sizeof(uint32_t), &cell_count);
        err |= clSetKernelArg(kernel_morton, 4, sizeof(uint32_t), &k);
        err |= clSetKernelArg(kernel_morton, 5, sizeof(cl_mem), &buf_morton_list);
        check_cl_error(err, "clSetKernelArg(kernel_morton)");

        size_t global_work_size = N;
        err = clEnqueueNDRangeKernel(queue, kernel_morton, 1, nullptr,
                               &global_work_size, nullptr, 0, nullptr, nullptr);
        check_cl_error(err, "clEnqueueNDRangeKernel(kernel_morton)");
        clFinish(queue);

        err = clEnqueueReadBuffer(queue, buf_morton_list, CL_TRUE, 0,
                            N * sizeof(morton_primitive), morton_list.data(), 0, nullptr, nullptr);
        check_cl_error(err, "clEnqueueReadBuffer(buf_morton_list)");
        phase_end = std::chrono::high_resolution_clock::now();
        morton_time_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();

        phase_start = std::chrono::high_resolution_clock::now();
        radix_sort();
        err = clEnqueueReadBuffer(queue, buf_morton_list, CL_TRUE, 0,
                            N * sizeof(morton_primitive), morton_list.data(), 0, nullptr, nullptr);
        check_cl_error(err, "clEnqueueReadBuffer(sorted morton_list)");
        phase_end = std::chrono::high_resolution_clock::now();
        sort_time_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();

        phase_start = std::chrono::high_resolution_clock::now();
        err = clSetKernelArg(kernel_init_leaves, 0, sizeof(cl_mem), &buf_morton_list);
        err |= clSetKernelArg(kernel_init_leaves, 1, sizeof(cl_mem), &buf_bboxes);
        err |= clSetKernelArg(kernel_init_leaves, 2, sizeof(int), &N);
        err |= clSetKernelArg(kernel_init_leaves, 3, sizeof(cl_mem), &buf_nodes);
        check_cl_error(err, "clSetKernelArg(kernel_init_leaves)");

        err = clEnqueueNDRangeKernel(queue, kernel_init_leaves, 1, nullptr,
                               &global_work_size, nullptr, 0, nullptr, nullptr);
        check_cl_error(err, "clEnqueueNDRangeKernel(kernel_init_leaves)");
        clFinish(queue);

        err = clSetKernelArg(kernel_hierarchy, 0, sizeof(cl_mem), &buf_morton_list);
        err |= clSetKernelArg(kernel_hierarchy, 1, sizeof(int), &N);
        err |= clSetKernelArg(kernel_hierarchy, 2, sizeof(cl_mem), &buf_nodes);
        check_cl_error(err, "clSetKernelArg(kernel_hierarchy)");

        size_t hierarchy_work_size = N - 1;
        err = clEnqueueNDRangeKernel(queue, kernel_hierarchy, 1, nullptr,
                                     &hierarchy_work_size, nullptr, 0, nullptr, nullptr);
        check_cl_error(err, "clEnqueueNDRangeKernel(kernel_hierarchy)");
        clFinish(queue);

        err = clEnqueueReadBuffer(queue, buf_nodes, CL_TRUE, 0,
                                  (2*N - 1) * sizeof(lbvh_node), nodes.data(), 0, nullptr, nullptr);
        check_cl_error(err, "clEnqueueReadBuffer(buf_nodes)");
        phase_end = std::chrono::high_resolution_clock::now();
        hierarchy_time_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();

        phase_start = std::chrono::high_resolution_clock::now();
        err = clSetKernelArg(kernel_build_bboxes, 0, sizeof(cl_mem), &buf_nodes);
        err |= clSetKernelArg(kernel_build_bboxes, 1, sizeof(int), &N);
        check_cl_error(err, "clSetKernelArg(kernel_build_bboxes)");

        err = clEnqueueNDRangeKernel(queue, kernel_build_bboxes, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueNDRangeKernel(kernel_build_bboxes)");
        clFinish(queue);

        err = clEnqueueReadBuffer(queue, buf_nodes, CL_TRUE, 0,
                            (2*N - 1) * sizeof(lbvh_node), nodes.data(), 0, nullptr, nullptr);
        check_cl_error(err, "clEnqueueReadBuffer(buf_nodes after bbox)");
        phase_end = std::chrono::high_resolution_clock::now();
        bbox_time_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();

        auto end_time = std::chrono::high_resolution_clock::now();
        construction_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        return hit_recursive(r, ray_t, rec, 0);
    }

    void enable_statistics() const { enable_stats = true; stats.reset(); }
    void disable_statistics() const { enable_stats = false; }
    const bvh_stats& get_stats() const { return stats; }
    void increment_ray_count() const { if (enable_stats) stats.rays_traced++; }

    double get_construction_time() const { return construction_time_ms; }
    double get_opencl_init_time() const { return opencl_init_time_ms; }
    double get_morton_time() const { return morton_time_ms; }
    double get_sort_time() const { return sort_time_ms; }
    double get_hierarchy_time() const { return hierarchy_time_ms; }
    double get_bbox_time() const { return bbox_time_ms; }

    void export_statistics(const std::string& filename, long long total_rays,
                           double avg_box_tests, double avg_prim_tests) const {
        std::ofstream out(filename);
        if (!out.is_open()) {
            std::cerr << "Failed to open: " << filename << std::endl;
            return;
        }
        out << std::fixed << std::setprecision(3);
        out << "opencl_init_ms," << opencl_init_time_ms << "\n";
        out << "data_prep_ms," << data_prep_time_ms << "\n";
        out << "morton_code_ms," << morton_time_ms << "\n";
        out << "radix_sort_ms," << sort_time_ms << "\n";
        out << "hierarchy_ms," << hierarchy_time_ms << "\n";
        out << "bbox_propagation_ms," << bbox_time_ms << "\n";
        out << "total_construction_ms," << construction_time_ms << "\n";
        out << "total_rays," << total_rays << "\n";
        out << "avg_box_tests_per_ray," << avg_box_tests << "\n";
        out << "avg_prim_tests_per_ray," << avg_prim_tests << "\n";
        out.close();
    }

    int get_tree_depth() const { return compute_depth(0); }

    cl_context get_opencl_context() const { return context; }
    cl_device_id get_opencl_device() const { return device; }
    cl_command_queue get_opencl_queue() const { return queue; }
    cl_mem get_nodes_buffer() const { return buf_nodes; }
    const std::vector<lbvh_node>& get_nodes() const { return nodes; }
    int get_primitive_count() const { return N; }
    const std::vector<shared_ptr<hittable>>& get_objects() const { return objects; }

  private:
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;

    cl_kernel kernel_morton = nullptr;
    cl_kernel kernel_init_leaves = nullptr;
    cl_kernel kernel_hierarchy = nullptr;
    cl_kernel kernel_build_bboxes = nullptr;
    cl_kernel kernel_create_histogram = nullptr;
    cl_kernel kernel_prefix_sum = nullptr;
    cl_kernel kernel_scatter = nullptr;
    cl_kernel kernel_scan_glob_sum = nullptr;
    cl_kernel kernel_add_glob_sums = nullptr;

    cl_mem buf_centroids = nullptr;
    cl_mem buf_morton_list = nullptr;
    cl_mem buf_bboxes = nullptr;
    cl_mem buf_nodes = nullptr;
    cl_mem buf_histogram = nullptr;
    cl_mem buf_scanned_gram = nullptr;
    cl_mem buf_glob_sum = nullptr;
    cl_mem buf_morton_list_out = nullptr;

    std::vector<shared_ptr<hittable>>& objects;
    int N = 0;
    std::vector<lbvh_node> nodes;
    std::vector<aabb> primitive_bboxes;
    std::vector<point3> centroids;
    aabb temp_scene_bbox;
    std::vector<morton_primitive> morton_list;
    uint32_t k = 10;
    uint32_t cell_count = 0;

    mutable bvh_stats stats;
    mutable bool enable_stats = false;

    double construction_time_ms = 0.0;
    double opencl_init_time_ms = 0.0;
    double data_prep_time_ms = 0.0;
    double morton_time_ms = 0.0;
    double sort_time_ms = 0.0;
    double hierarchy_time_ms = 0.0;
    double bbox_time_ms = 0.0;

    bool hit_recursive(const ray& r, interval ray_t, hit_record& rec, const int node_idx) const {
        if (node_idx >= 2*N - 1 || node_idx < 0) return false;
        if (enable_stats) stats.node_visits++;

        const lbvh_node& node = nodes[node_idx];
        if (!node.bbox.hit(r, ray_t)) return false;

        if (node.is_leaf()) {
            if (enable_stats) stats.primitive_tests++;
            return objects[node.primitive_id]->hit(r, ray_t, rec);
        }

        bool hit_left = hit_recursive(r, ray_t, rec, node.left);
        bool hit_right = hit_recursive(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec, node.right);
        return hit_left || hit_right;
    }

    void radix_sort() {
        cl_int err;
        const int num_buckets = 16;
        const int num_passes = 8;
        const int histo_block_size = 1024;
        const int histo_threads = 256;
        const int num_groups = (N + histo_block_size - 1) / histo_block_size;
        const int histogram_size = num_buckets * num_groups * histo_threads;
        const int prefix_threads = 256;
        const int prefix_elements_per_wg = 512;
        const int num_prefix_wgs = (histogram_size + prefix_elements_per_wg - 1) / prefix_elements_per_wg;

        buf_histogram = clCreateBuffer(context, CL_MEM_READ_WRITE, histogram_size * sizeof(uint32_t), nullptr, &err);
        check_cl_error(err, "clCreateBuffer(buf_histogram)");
        buf_scanned_gram = clCreateBuffer(context, CL_MEM_READ_WRITE, histogram_size * sizeof(uint32_t), nullptr, &err);
        check_cl_error(err, "clCreateBuffer(buf_scanned_gram)");
        buf_glob_sum = clCreateBuffer(context, CL_MEM_READ_WRITE, num_prefix_wgs * sizeof(uint32_t), nullptr, &err);
        check_cl_error(err, "clCreateBuffer(buf_glob_sum)");
        buf_morton_list_out = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(morton_primitive), nullptr, &err);
        check_cl_error(err, "clCreateBuffer(buf_morton_list_out)");

        cl_mem buf_in = buf_morton_list;
        cl_mem buf_out = buf_morton_list_out;

        for (int pass = 0; pass < num_passes; pass++) {
            size_t histo_local = histo_threads;
            size_t histo_global = num_groups * histo_threads;

            err = clSetKernelArg(kernel_create_histogram, 0, sizeof(cl_mem), &buf_in);
            err |= clSetKernelArg(kernel_create_histogram, 1, sizeof(int), &N);
            err |= clSetKernelArg(kernel_create_histogram, 2, sizeof(int), &pass);
            err |= clSetKernelArg(kernel_create_histogram, 3, sizeof(cl_mem), &buf_histogram);
            check_cl_error(err, "clSetKernelArg(create_histogram)");
            err = clEnqueueNDRangeKernel(queue, kernel_create_histogram, 1, nullptr, &histo_global, &histo_local, 0, nullptr, nullptr);
            check_cl_error(err, "clEnqueueNDRangeKernel(create_histogram)");

            size_t prefix_local = prefix_threads;
            size_t prefix_global = num_prefix_wgs * prefix_threads;

            err = clSetKernelArg(kernel_prefix_sum, 0, sizeof(cl_mem), &buf_histogram);
            err |= clSetKernelArg(kernel_prefix_sum, 1, sizeof(cl_mem), &buf_scanned_gram);
            err |= clSetKernelArg(kernel_prefix_sum, 2, sizeof(cl_mem), &buf_glob_sum);
            err |= clSetKernelArg(kernel_prefix_sum, 3, sizeof(int), &num_groups);
            err |= clSetKernelArg(kernel_prefix_sum, 4, sizeof(int), &histogram_size);
            check_cl_error(err, "clSetKernelArg(prefix_sum)");
            err = clEnqueueNDRangeKernel(queue, kernel_prefix_sum, 1, nullptr, &prefix_global, &prefix_local, 0, nullptr, nullptr);
            check_cl_error(err, "clEnqueueNDRangeKernel(prefix_sum)");

            if (num_prefix_wgs > 1) {
                size_t scan_local = 1;
                while (scan_local < (size_t)num_prefix_wgs) scan_local <<= 1;
                size_t scan_global = scan_local;
                const size_t max_local_size = 256;

                if (scan_local <= max_local_size) {
                    err = clSetKernelArg(kernel_scan_glob_sum, 0, sizeof(cl_mem), &buf_glob_sum);
                    err |= clSetKernelArg(kernel_scan_glob_sum, 1, scan_local * sizeof(cl_uint), nullptr);
                    err |= clSetKernelArg(kernel_scan_glob_sum, 2, sizeof(int), &num_prefix_wgs);
                    check_cl_error(err, "clSetKernelArg(scan_glob_sum)");
                    err = clEnqueueNDRangeKernel(queue, kernel_scan_glob_sum, 1, nullptr, &scan_global, &scan_local, 0, nullptr, nullptr);
                    check_cl_error(err, "clEnqueueNDRangeKernel(scan_glob_sum)");
                } else {
                    clFinish(queue);
                    std::vector<uint32_t> glob_sum_host(num_prefix_wgs);
                    err = clEnqueueReadBuffer(queue, buf_glob_sum, CL_TRUE, 0, num_prefix_wgs * sizeof(uint32_t), glob_sum_host.data(), 0, nullptr, nullptr);
                    check_cl_error(err, "clEnqueueReadBuffer(glob_sum fallback)");
                    uint32_t running_sum = 0;
                    for (int i = 0; i < num_prefix_wgs; i++) {
                        uint32_t temp = glob_sum_host[i];
                        glob_sum_host[i] = running_sum;
                        running_sum += temp;
                    }
                    err = clEnqueueWriteBuffer(queue, buf_glob_sum, CL_TRUE, 0, num_prefix_wgs * sizeof(uint32_t), glob_sum_host.data(), 0, nullptr, nullptr);
                    check_cl_error(err, "clEnqueueWriteBuffer(glob_sum fallback)");
                }

                err = clSetKernelArg(kernel_add_glob_sums, 0, sizeof(cl_mem), &buf_glob_sum);
                err |= clSetKernelArg(kernel_add_glob_sums, 1, sizeof(cl_mem), &buf_scanned_gram);
                err |= clSetKernelArg(kernel_add_glob_sums, 2, sizeof(int), &histogram_size);
                check_cl_error(err, "clSetKernelArg(add_glob_sums)");
                err = clEnqueueNDRangeKernel(queue, kernel_add_glob_sums, 1, nullptr, &prefix_global, &prefix_local, 0, nullptr, nullptr);
                check_cl_error(err, "clEnqueueNDRangeKernel(add_glob_sums)");
            }

            err = clSetKernelArg(kernel_scatter, 0, sizeof(cl_mem), &buf_in);
            err |= clSetKernelArg(kernel_scatter, 1, sizeof(cl_mem), &buf_scanned_gram);
            err |= clSetKernelArg(kernel_scatter, 2, sizeof(int), &N);
            err |= clSetKernelArg(kernel_scatter, 3, sizeof(int), &pass);
            err |= clSetKernelArg(kernel_scatter, 4, sizeof(cl_mem), &buf_out);
            check_cl_error(err, "clSetKernelArg(scatter)");
            err = clEnqueueNDRangeKernel(queue, kernel_scatter, 1, nullptr, &histo_global, &histo_local, 0, nullptr, nullptr);
            check_cl_error(err, "clEnqueueNDRangeKernel(scatter)");

            std::swap(buf_in, buf_out);
        }
        clFinish(queue);
    }

    int compute_depth(int node_idx) const {
        if (node_idx < 0) return 0;
        const lbvh_node& node = nodes[node_idx];
        if (node.is_leaf()) return 1;
        return 1 + std::max(compute_depth(node.left), compute_depth(node.right));
    }

    void check_cl_error(cl_int err, const char* operation) const {
        if (err != CL_SUCCESS) {
            std::cerr << "OpenCL error during " << operation << ": " << err << std::endl;
            exit(1);
        }
    }

    void init_opencl() {
        cl_int err;
        err = clGetPlatformIDs(1, &platform, nullptr);
        check_cl_error(err, "clGetPlatformIDs");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        check_cl_error(err, "clGetDeviceIDs");
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        check_cl_error(err, "clCreateContext");
        queue = clCreateCommandQueue(context, device, 0, &err);
        check_cl_error(err, "clCreateCommandQueue");

        std::string source = load_kernel_source_with_includes({
            "src/bvh/opencl_shared_types.cl",
            "src/bvh/lbvh_construction_kernels.cl"
        });
        const char* src_ptr = source.c_str();
        size_t src_len = source.length();

        program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
        check_cl_error(err, "clCreateProgramWithSource");
        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::cerr << "OpenCL kernel compilation error:\n" << log.data() << std::endl;
            exit(1);
        }

        kernel_morton = clCreateKernel(program, "compute_morton", &err);
        check_cl_error(err, "clCreateKernel(compute_morton)");
        kernel_init_leaves = clCreateKernel(program, "init_leaf_nodes", &err);
        check_cl_error(err, "clCreateKernel(init_leaf_nodes)");
        kernel_hierarchy = clCreateKernel(program, "create_hierarchy", &err);
        check_cl_error(err, "clCreateKernel(create_hierarchy)");
        kernel_build_bboxes = clCreateKernel(program, "build_bboxes", &err);
        check_cl_error(err, "clCreateKernel(build_bboxes)");
        kernel_create_histogram = clCreateKernel(program, "create_histogram", &err);
        check_cl_error(err, "clCreateKernel(create_histogram)");
        kernel_prefix_sum = clCreateKernel(program, "prefix_sum", &err);
        check_cl_error(err, "clCreateKernel(prefix_sum)");
        kernel_scatter = clCreateKernel(program, "scatter", &err);
        check_cl_error(err, "clCreateKernel(scatter)");
        kernel_scan_glob_sum = clCreateKernel(program, "scan_glob_sum", &err);
        check_cl_error(err, "clCreateKernel(scan_glob_sum)");
        kernel_add_glob_sums = clCreateKernel(program, "add_glob_sums", &err);
        check_cl_error(err, "clCreateKernel(add_glob_sums)");
    }

    void cleanup_opencl() {
        if (kernel_morton) clReleaseKernel(kernel_morton);
        if (kernel_init_leaves) clReleaseKernel(kernel_init_leaves);
        if (kernel_hierarchy) clReleaseKernel(kernel_hierarchy);
        if (kernel_build_bboxes) clReleaseKernel(kernel_build_bboxes);
        if (kernel_create_histogram) clReleaseKernel(kernel_create_histogram);
        if (kernel_prefix_sum) clReleaseKernel(kernel_prefix_sum);
        if (kernel_scatter) clReleaseKernel(kernel_scatter);
        if (kernel_scan_glob_sum) clReleaseKernel(kernel_scan_glob_sum);
        if (kernel_add_glob_sums) clReleaseKernel(kernel_add_glob_sums);
        if (buf_centroids) clReleaseMemObject(buf_centroids);
        if (buf_morton_list) clReleaseMemObject(buf_morton_list);
        if (buf_bboxes) clReleaseMemObject(buf_bboxes);
        if (buf_nodes) clReleaseMemObject(buf_nodes);
        if (buf_histogram) clReleaseMemObject(buf_histogram);
        if (buf_scanned_gram) clReleaseMemObject(buf_scanned_gram);
        if (buf_glob_sum) clReleaseMemObject(buf_glob_sum);
        if (buf_morton_list_out) clReleaseMemObject(buf_morton_list_out);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    std::string load_kernel_source_with_includes(const std::vector<const char*>& filenames) const {
        std::stringstream combined;
        for (const char* filename : filenames) {
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Failed to open kernel file: " << filename << std::endl;
                exit(1);
            }
            combined << file.rdbuf() << "\n";
        }
        return combined.str();
    }
};

#endif
