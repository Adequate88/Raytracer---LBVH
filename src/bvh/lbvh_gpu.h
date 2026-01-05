#ifndef LBVH_GPU_H
#define LBVH_GPU_H
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
#include <map>

#include <queue>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

struct int2 {
  int x, y;

  int2() : x(0), y(0) {}  // Default constructor
  int2(int x, int y) : x(x), y(y) {}
};

struct lbvh_node {
    aabb bbox;
    int left;
    int right;
    int parent;
    int primitive_id;  // -1 for internal nodes, >=0 for leaf nodes
    int atomic_counter;

    // Default constructor
    lbvh_node()
        : left(-1), right(-1), parent(-1), primitive_id(-1), atomic_counter(0) {}

    // Leaf constructor
    lbvh_node(const aabb& bbox, int prim_id)
        : bbox(bbox), left(-1), right(-1), parent(-1), primitive_id(prim_id), atomic_counter(0) {}

    // Internal constructor
    lbvh_node(int l, int r)
        : left(l), right(r), parent(-1), primitive_id(-1), atomic_counter(0) {}

    bool is_leaf() const { return primitive_id >= 0; }
};

class lbvh_gpu {
  public:
    ~lbvh_gpu() {
      cleanup_opencl();
    }

    lbvh_gpu(std::vector<shared_ptr<hittable>>& objects)
        : objects(objects), N(objects.size()) {

        // Pre-allocate arrays
        nodes.resize(2*N - 1);   

        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();

        // Edge case: empty scene
        if (N == 0) {
            return;
        }

        // Phase 1: OpenCL initialization
        auto phase_start = std::chrono::high_resolution_clock::now();
        init_opencl();
        auto phase_end = std::chrono::high_resolution_clock::now();
        opencl_init_time_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();

        // LBVH setup
        morton_list.resize(N);
        cell_count = 1 << k;  // 2^10 = 1024

        // Compute scene bounding box
        temp_scene_bbox = aabb::empty;
        for (int i = 0; i < N; i++) {
            temp_scene_bbox = aabb(temp_scene_bbox, objects[i]->bounding_box());
            primitive_bboxes.push_back(objects[i]->bounding_box());
            centroids.push_back(objects[i]->get_centroid());
        }

        // Allocating buffers
        cl_int err;

        buf_centroids = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 N * sizeof(point3), centroids.data(), &err);
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

        // Phase 2: Morton code computation
        phase_start = std::chrono::high_resolution_clock::now();

        // Set kernel arguments for compute_morton
        err = clSetKernelArg(kernel_morton, 0, sizeof(cl_mem), &buf_centroids);
        err |= clSetKernelArg(kernel_morton, 1, sizeof(aabb), &temp_scene_bbox);
        err |= clSetKernelArg(kernel_morton, 2, sizeof(int), &N);
        err |= clSetKernelArg(kernel_morton, 3, sizeof(uint32_t), &cell_count);
        err |= clSetKernelArg(kernel_morton, 4, sizeof(uint32_t), &k);
        err |= clSetKernelArg(kernel_morton, 5, sizeof(cl_mem), &buf_morton_list);
        check_cl_error(err, "clSetKernelArg(kernel_morton)");

        // Launch kernel
        size_t global_work_size = N;
        err = clEnqueueNDRangeKernel(queue, kernel_morton, 1, nullptr,
                               &global_work_size, nullptr, 0, nullptr, nullptr);
        check_cl_error(err, "clEnqueueNDRangeKernel(kernel_morton)");

        // Wait for completion
        clFinish(queue);

        // Read morton codes back to CPU
        err = clEnqueueReadBuffer(queue, buf_morton_list, CL_TRUE, 0,
                            N * sizeof(morton_primitive), morton_list.data(),
                            0, nullptr, nullptr);
        check_cl_error(err, "clEnqueueReadBuffer(buf_morton_list)");

        phase_end = std::chrono::high_resolution_clock::now();
        morton_time_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();

        // Phase 3: Sort by Morton code
        phase_start = std::chrono::high_resolution_clock::now();
        radix_sort();
        phase_end = std::chrono::high_resolution_clock::now();
        sort_time_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();

        // Update GPU buffer with sorted morton codes
        err = clEnqueueWriteBuffer(queue, buf_morton_list, CL_TRUE, 0,
                             N * sizeof(morton_primitive), morton_list.data(),
                             0, nullptr, nullptr);
        check_cl_error(err, "clEnqueueWriteBuffer(buf_morton_list after sort)");

        // Phase 4: Hierarchy construction (tree building)
        phase_start = std::chrono::high_resolution_clock::now();

        // Set kernel arguments for init_leaf_nodes
        err = clSetKernelArg(kernel_init_leaves, 0, sizeof(cl_mem), &buf_morton_list);
        err |= clSetKernelArg(kernel_init_leaves, 1, sizeof(cl_mem), &buf_bboxes);
        err |= clSetKernelArg(kernel_init_leaves, 2, sizeof(int), &N);
        err |= clSetKernelArg(kernel_init_leaves, 3, sizeof(cl_mem), &buf_nodes);
        check_cl_error(err, "clSetKernelArg(kernel_init_leaves)");

        err = clEnqueueNDRangeKernel(queue, kernel_init_leaves, 1, nullptr,
                               &global_work_size, nullptr, 0, nullptr, nullptr);
        check_cl_error(err, "clEnqueueNDRangeKernel(kernel_init_leaves)");
        clFinish(queue);

        // Set kernel arguments for create_hierarchy
        err = clSetKernelArg(kernel_hierarchy, 0, sizeof(cl_mem), &buf_morton_list);
        err |= clSetKernelArg(kernel_hierarchy, 1, sizeof(int), &N);
        err |= clSetKernelArg(kernel_hierarchy, 2, sizeof(cl_mem), &buf_nodes);
        check_cl_error(err, "clSetKernelArg(kernel_hierarchy)");
        
        size_t hierarchy_work_size = N - 1;
        err = clEnqueueNDRangeKernel(queue, kernel_hierarchy, 1, nullptr,
                                     &hierarchy_work_size, nullptr, 0, nullptr, nullptr);
        check_cl_error(err, "clEnqueueNDRangeKernel(kernel_hierarchy)");
        clFinish(queue);

        // Read nodes back to CPU
        err = clEnqueueReadBuffer(queue, buf_nodes, CL_TRUE, 0,
                                  (2*N - 1) * sizeof(lbvh_node), nodes.data(),
                                  0, nullptr, nullptr);
        check_cl_error(err, "clEnqueueReadBuffer(buf_nodes)");

        phase_end = std::chrono::high_resolution_clock::now();
        hierarchy_time_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();

        // Phase 5: Bottom-up bounding box propagation
        phase_start = std::chrono::high_resolution_clock::now();

        err = clSetKernelArg(kernel_build_bboxes, 0, sizeof(cl_mem), &buf_nodes);
        err |= clSetKernelArg(kernel_build_bboxes, 1, sizeof(int), &N);
        check_cl_error(err, "clSetKernelArg(kernel_build_bboxes)");
        
        err = clEnqueueNDRangeKernel( queue, kernel_build_bboxes, 1,NULL, &global_work_size, NULL, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueNDRangeKernel(kernel_build_bboxes)");

        clFinish(queue);
        
        err = clEnqueueReadBuffer(queue, buf_nodes, CL_TRUE, 0,
                            (2*N - 1) * sizeof(lbvh_node), nodes.data(),
                            0, nullptr, nullptr);
        check_cl_error(err, "clEnqueueReadBuffer(buf_nodes after bbox)");

        phase_end = std::chrono::high_resolution_clock::now();
        bbox_time_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();

        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();
        construction_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }

    // Entry point for ray intersection
    bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        return hit_recursive(r, ray_t, rec, 0);
    }

    // Statistics methods
    void enable_statistics() const { enable_stats = true; stats.reset(); }
    void disable_statistics() const { enable_stats = false; }
    const bvh_stats& get_stats() const { return stats; }
    void increment_ray_count() const { if (enable_stats) stats.rays_traced++; }

    // Timing methods
    double get_construction_time() const { return construction_time_ms; }
    double get_opencl_init_time() const { return opencl_init_time_ms; }
    double get_morton_time() const { return morton_time_ms; }
    double get_sort_time() const { return sort_time_ms; }
    double get_hierarchy_time() const { return hierarchy_time_ms; }
    double get_bbox_time() const { return bbox_time_ms; }

    // Tree visualization
    void print_tree(int max_depth = 10) const {
        std::clog << "\n=== LBVH Tree Structure ===\n";
        std::clog << "Total primitives: " << N << "\n";
        std::clog << "Tree depth limit for display: " << max_depth << "\n";
        std::clog << "\nTree structure:\n";
        print_node_recursive(0, 0, max_depth);
        std::clog << "===========================\n";
    }

    int get_tree_depth() const {
        return compute_depth(0);
    }

    // GPU resource access (for GPU rendering integration)
    cl_context get_opencl_context() const { return context; }
    cl_device_id get_opencl_device() const { return device; }
    cl_command_queue get_opencl_queue() const { return queue; }
    cl_mem get_nodes_buffer() const { return buf_nodes; }
    const std::vector<lbvh_node>& get_nodes() const { return nodes; }
    int get_primitive_count() const { return N; }
    const std::vector<shared_ptr<hittable>>& get_objects() const { return objects; }

  private:
    // OpenCL member variables
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;

    // Kernels
    cl_kernel kernel_morton = nullptr;
    cl_kernel kernel_init_leaves = nullptr;
    cl_kernel kernel_hierarchy = nullptr;
    cl_kernel kernel_build_bboxes = nullptr;

    // Memory Buffers
    cl_mem buf_centroids = nullptr;
    cl_mem buf_morton_list = nullptr;
    cl_mem buf_bboxes = nullptr;
    cl_mem buf_nodes = nullptr;

    int N = 0;

    std::vector<lbvh_node> nodes;

    std::vector<shared_ptr<hittable>>& objects;
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
    double morton_time_ms = 0.0;
    double sort_time_ms = 0.0;
    double hierarchy_time_ms = 0.0;
    double bbox_time_ms = 0.0;

    bool hit_recursive(const ray& r, interval ray_t, hit_record& rec, const int node_idx) const {
        if (node_idx >= 2*N - 1 || node_idx < 0) return false;

        // Track node visit
        if (enable_stats) stats.node_visits++;

        const lbvh_node& node = nodes[node_idx];
        // Test bounding box
        if (!node.bbox.hit(r, ray_t))
            return false;

        // Is primitive (leaf node)
        if (node.is_leaf()) {
            if (enable_stats) stats.primitive_tests++;
            return objects[node.primitive_id]->hit(r, ray_t, rec);
        }

        // Recurse children (internal node)
        bool hit_left = hit_recursive(r, ray_t, rec, node.left);
        bool hit_right = hit_recursive(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max),
                                       rec, node.right);
        return hit_left || hit_right;
    }

    void bottom_up_bbox_build() {
      std::queue<int> q;
      std::vector<bool> enqueued(N - 1, false);  // Track which internal nodes are in queue

      // Add all leaf parents to queue
      for (int i = 0; i < N; i++) {
        int leaf_idx = N - 1 + i;
        int parent = nodes[leaf_idx].parent;
        if (parent != -1 && !enqueued[parent]) {
          q.push(parent);
          enqueued[parent] = true;
        }
      }

      while (!q.empty()) {
        int current_node_idx = q.front();
        q.pop();
        enqueued[current_node_idx] = false;  // No longer in queue

        lbvh_node& current_node = nodes[current_node_idx];

        // Skip if already processed (bbox has been set to non-empty)
        if (current_node.bbox.min_x <= current_node.bbox.max_x)
          continue;

        lbvh_node& left_child = nodes[current_node.left];
        lbvh_node& right_child = nodes[current_node.right];

        // Check if both children have valid bboxes before computing parent bbox
        bool left_ready = (left_child.bbox.min_x <= left_child.bbox.max_x);
        bool right_ready = (right_child.bbox.min_x <= right_child.bbox.max_x);

        if (!left_ready || !right_ready) {
          // One or both children not ready yet, push back to queue and try later
          if (!enqueued[current_node_idx]) {
            q.push(current_node_idx);
            enqueued[current_node_idx] = true;
          }
          continue;
        }

        current_node.bbox = aabb(left_child.bbox, right_child.bbox);

        // Add parent to queue if not already there
        if (current_node.parent != -1 && !enqueued[current_node.parent]) {
          q.push(current_node.parent);
          enqueued[current_node.parent] = true;
        }
      }
    }

    void radix_sort() {
        // Process each bit from LSB to MSB
        for (int bit = 0; bit < 30; bit++) {
            std::vector<morton_primitive> zero_bucket;
            std::vector<morton_primitive> one_bucket;

            for (const auto& prim : morton_list) {
                if (((prim.morton_code >> bit) & 1) == 0) {
                    zero_bucket.push_back(prim);
                } else {
                    one_bucket.push_back(prim);
                }
            }

            morton_list.clear();
            morton_list.insert(morton_list.end(), zero_bucket.begin(), zero_bucket.end());
            morton_list.insert(morton_list.end(), one_bucket.begin(), one_bucket.end());
        }
    }

    // Helper method to print tree structure recursively
    void print_node_recursive(int node_idx, int depth, int max_depth) const {
        if (node_idx < 0 || depth > max_depth) return;

        const lbvh_node& node = nodes[node_idx];

        // Indentation
        for (int i = 0; i < depth; i++) std::clog << "  ";

        // Node info
        if (node.is_leaf()) {
            std::clog << "LEAF [prim_id=" << node.primitive_id << "]";
        } else {
            std::clog << "INTERNAL";
        }

        // Print bounding box info
        std::clog << " bbox=[(" << node.bbox.min_x << "," << node.bbox.min_y << "," << node.bbox.min_z << ")";
        std::clog << " - (" << node.bbox.max_x << "," << node.bbox.max_y << "," << node.bbox.max_z << ")]";
        std::clog << "\n";

        // Recurse to children if internal node
        if (!node.is_leaf()) {
            print_node_recursive(node.left, depth + 1, max_depth);
            print_node_recursive(node.right, depth + 1, max_depth);
        }
    }

    // Helper method to compute tree depth
    int compute_depth(int node_idx) const {
        if (node_idx < 0) return 0;

        const lbvh_node& node = nodes[node_idx];
        if (node.is_leaf()) return 1;

        int left_depth = compute_depth(node.left);
        int right_depth = compute_depth(node.right);

        return 1 + std::max(left_depth, right_depth);
    }

    // OpenCL host functions
    
    void check_cl_error (cl_int err, const char* operation) const {
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
    }

    void cleanup_opencl() {
      // Release kernels
      if (kernel_morton) clReleaseKernel(kernel_morton);
      if (kernel_init_leaves) clReleaseKernel(kernel_init_leaves);
      if (kernel_hierarchy) clReleaseKernel(kernel_hierarchy);
      if (kernel_build_bboxes) clReleaseKernel(kernel_build_bboxes);

      // Release BVH construction buffers
      if (buf_centroids) clReleaseMemObject(buf_centroids);
      if (buf_morton_list) clReleaseMemObject(buf_morton_list);
      if (buf_bboxes) clReleaseMemObject(buf_bboxes);
      if (buf_nodes) clReleaseMemObject(buf_nodes);

      // Release OpenCL resources
      if (program) clReleaseProgram(program);
      if (queue) clReleaseCommandQueue(queue);
      if (context) clReleaseContext(context);
    }

    std::string load_kernel_source(const char* filename) {
      std::ifstream file(filename);
      if (!file.is_open()) {
          std::cerr << "Failed to open kernel file: " << filename << std::endl;
          exit(1);
      }
      std::stringstream buffer;
      buffer << file.rdbuf();
      return buffer.str();
    }

    std::string load_kernel_source_with_includes(const std::vector<const char*>& filenames) const {
      std::stringstream combined;

      for (const char* filename : filenames) {
        std::ifstream file(filename);
        if (!file.is_open()) {
          std::cerr << "Failed to open kernel file: " << filename << std::endl;
          exit(1);
        }
        combined << "// ============ BEGIN: " << filename << " ============\n";
        combined << file.rdbuf();
        combined << "\n// ============ END: " << filename << " ============\n\n";
      }

      return combined.str();
    }


};

#endif
