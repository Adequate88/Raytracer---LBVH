#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <chrono>

const char* kernel_source = R"(
__kernel void vector_add(__global const float* A,
                          __global const float* B,
                          __global float* C) {
  int i = get_global_id(0);
  C[i] = A[i] + B[i];
})";

int main() {

  const int N = 10000000; // 10 million elements!

  // Get Device and Platform
  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, nullptr);

  cl_device_id device;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

  // Context
  
  cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

  std::vector<float> h_A(N, 1.0f);
  std::vector<float> h_B(N, 2.0f);
  std::vector<float> h_C(N);

  // Create buffers
  cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N*sizeof(float), h_A.data(), nullptr);
  cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N*sizeof(float), h_B.data(), nullptr);
  cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N*sizeof(float), nullptr, nullptr);

  // Compile kernel 
  cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, nullptr);
  cl_int err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

  if (err != CL_SUCCESS) {
    char log[4096];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 4096, log, nullptr);
    std::cerr << "Kernel compile error:\n" << log << "\n";
    return 1;
  }

  cl_kernel kernel = clCreateKernel(program, "vector_add", nullptr);
  
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);

  size_t global_work_size = N;

  // Time GPU execution
  auto gpu_start = std::chrono::high_resolution_clock::now();
  clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
  clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, N*sizeof(float), h_C.data(), 0, nullptr, nullptr);
  auto gpu_end = std::chrono::high_resolution_clock::now();

  double gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

  // 9. Print GPU results
  std::cout << "GPU First 10 results: ";
  for (int i = 0; i < 10; i++) {
    std::cout << h_C[i] << " ";
  }
  std::cout << "\n(Expected: all 3.0)\n";

  // 10. CPU version for comparison
  std::vector<float> h_C_cpu(N);
  auto cpu_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    h_C_cpu[i] = h_A[i] + h_B[i];
  }
  auto cpu_end = std::chrono::high_resolution_clock::now();
  double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

  std::cout << "CPU First 10 results: ";
  for (int i = 0; i < 10; i++) {
    std::cout << h_C_cpu[i] << " ";
  }
  std::cout << "\n";

  // 11. Print timing comparison
  std::cout << "\n=== Performance Comparison ===\n";
  std::cout << "GPU time: " << gpu_time << " ms\n";
  std::cout << "CPU time: " << cpu_time << " ms\n";
  std::cout << "Speedup: " << (cpu_time / gpu_time) << "x\n";
  
  clReleaseMemObject(d_A);
  clReleaseMemObject(d_B);
  clReleaseMemObject(d_C);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return 0;
}
