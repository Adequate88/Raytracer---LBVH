# Report 

### Goals

The aim of this project is to maximize rendering times by optimizing **BVH construction** and **ray tracing**. 

##### BVH

BVH construction is already fully parallelized on the GPU using OpenCL, so optimizations here will be mostly through high-level algorithm changes and low-level optimizations. Our goal is to move the construction onto a Vulkan back-end to have more memory control and to reduce the overhead of OpenCL.

##### Ray tracing

Ray tracing takes place on the CPU. Optimization here will be to move ray tracing onto the GPU, which will incur a significant speed-up. To ensure that rendering is fast, we will also aim to ensure efficient intersection between BVH and rays on the GPU. High-level optimizations will also be significant here since the base code (from Ray Tracing in One Weekend) is not well optimized.

### Metrics

To measure the overall performance of our application, we will use *total run time (ms)*. This metric consists of both BVH construction time and time it takes to render the image. 

##### BVH

To measure BVH performance, we will measure *total BVH construction time (ms)*. This metric gives us an overall measure of both the overhead costs of communicating between host and device, and also a measure how fast each kernel runs. We will also measure the *run time of each individual kernel (ms)* used in BVH construction. This allows us to optimize at each step of the construction.  

##### Ray tracing

For ray tracing performance, we will measure *total rendering time (ms)* - which gives an overall picture of how long it takes to generate the image. We will also measure *rays traced per second*, since total rendering time may vary depending on various initialization parameters (resolution, number of rays sampled, etc.), whilst rays traced per second gives a measure independent of such parameters.

## Baseline Measurements

These are the measurements of the application before any optimizations take place. More precisely, the baseline version of the software consists of

- BVH construction written on OpenCL and fully-parallelized based on the Karras algorithm. 
  - It consists of ...
- Ray tracer is on CPU.

#### Measurements

On:

- CPU: AMD Ryzen 7 7840HS

- *GPU*: AMD Radeon 780M (Integrated) Graphics (RADV PHOENIX) 

##### BVH

- *Total BVH Construction Time (ms)*: **187.15**
- *OpenCL Initialization (ms)*: **178.804**
- *Data Preparation (ms)*: **7.008**
- *Morton Code Kernel (ms)*: **0.155**
- *Radix Sort Kernel (ms)*: **0.511**
- *Hierarchy Creation Kernel (ms)*: **0.394**
- *Bounding Box Kernel (ms)*: **0.235**

##### Ray tracer

- *Total Rendering Time (ms)*: **3418.88**
- *Rays traced per second (Rays/Seconds)*: 1454400 / 3.419 = **4.25387e+05**

## Profiling and Bottlenecks



## Current Todos

- [ ] Implement Vulkan Back-end for ray tracing kernels
  - [ ] Create compute pipeline
  - [ ] Pass BVH structure into local device memory
- [ ] Implement ray tracer kernel



