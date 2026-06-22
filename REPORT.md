# Report 

### Goals

The aim of this project is to reduce the runtime of the application, which includes **BVH construction** and a single **ray traced** frame. To this end, we must optimize both of these components to reduce total application runtime. 

## Baseline

##### BVH

BVH construction on the baseline is already fully parallelized on the GPU using OpenCL. Optimizations here will mostly be through high-level algorithm changes and low-level optimizations. Our goal is to move the construction onto a Vulkan back-end to have more control of memory usage.

##### Ray tracing

Ray tracing on the baseline currently fully runs on CPU. To optimize this component we aim to move it onto the GPU. To ensure that rendering remains fast, we will also aim to ensure efficient traversal of the BVH structure on the GPU. High-level optimizations will also be significant here since the base code (from Ray Tracing in One Weekend) is not well optimized.

### Metrics

To measure the overall performance of our application, we will use *total run time (ms)*. This metric consists of both BVH construction time and time it takes to render the frame. Alongside this, we measure the time for each kernel of the BVH construction and ray tracer (once it is on the GPU). 

##### BVH

To measure BVH performance, we will measure *total BVH construction time (ms)*. This metric gives us an overall measure of both the overhead costs of communicating between host and device, and also a measure how fast each kernel runs. We will also measure the *run time of each individual kernel (ms)* used in BVH construction. This allows us to optimize at each step of the construction.  

##### Ray tracing

For ray tracing performance, we will measure *total rendering time (ms)* - which gives an overall picture of how long it takes to generate the image. Since total rendering time may vary depending on various initialization parameters (resolution, number of rays sampled, etc.), we ensure these parameters are consistent across measuring. 

### Measurements

Measurements are taken on a scene consisting of a single Stanford bunny model of 144,046 triangles. It is comprised of a single Lambertian material. Rendering is done at a 512x512 resolution, with 1 sample per pixel, and a max bounce depth of 20. To ensure accurate measurements, we take multiple measurements of each component: 100 BVH constructions and 15 frames (since frame generation is slower), and then take the mean as our final value. To minimize noise, we let the application warm-up by running - but not measuring - 10 BVH constructions and 3 frames. All measurements take place on an AMD Ryzen 7 7840HS CPU and an AMD Radeon 780M (Integrated) Graphics (RADV PHOENIX) GPU. 

| Measurement | Value (ms) |
| --- | --- |
| Total Application Run Time | 696.4 |
| Total BVH Construction Time | 32.7 |
| Total Rendering Time | 533.3 |

*Note: the gap between total application time and the sum of the two components is caused by initialization of OpenCL.*  

#### Conclusion

Based on the baseline measurements, we clearly see that the main bottleneck of our application is rendering time, which consists of 75% of our application runtime. As a result, our first plan of action is to optimize rendering time. Naturally, since BVH construction is on the GPU, we will move ray tracing onto the GPU as well.

## Phase 1: Vulkan Backend and Raytracing onto GPU

As previously mentioned, Vulkan will be the API of choice for GPU work since it allows us finer grained control over the GPU pipeline. To this end, we wrote the Vulkan context for the project, and wrote a single kernel to perform ray tracing (i.e. the megakernel approach). Before porting the BVH construction onto the Vulkan backend, we measure how performance changes commpared to the baseline when ray tracing on the GPU but with no acceleration structures.

### Initial Measurements

| Measurement | Value (ms) |
| --- | --- |
| Total Rendering Time | 1523.82 |

*Note: we omit BVH construction since it has not changed.*

#### Conclusion

After moving the ray tracer onto GPU (with no BVH), we saw a significant drop in performance - which is not expected. However, we suspect that this is due ray tracing with no acceleration since despite running on significantly more threads, each GPU thread must iterate over each 140k triangles to test for an intersection. On the otherhand, the CPU algorithm (with BVH) only sparsely tests for triangles because of the BVH. Since BVHs tend to improve performance scaling with scene complexity and triangle count, we measure the baseline and phase 1 again but on a scene with lower triangle count. We do this to ensure that our GPU implementation is functioning as expected, and that the decrease in performance is explainable by the lacking acceleration structure.

## Phase 1.5: Testing GPU implementation 

For this test we measure on a smaller object with on 6392 triangles.

### Measurements

| Configuration | Measurement | Value (ms) |
| --- | --- | --- |
| Baseline | Total Rendering Time |  |
| Phase 1 | Total Rendering Time |  |

#### Conclusion

The measurements taken with a smaller scene confirm our suspicion that the decrease in performance is indeed due to ray tracing without an acceleration structure. As a result, porting the BVH onto the Vulkan backend and implementing GPU BVH traversal is our next step.

## Phase 2: BVH to Vulkan Backend

On a shader level, moving the BVH to our Vulkan backened simply required translating the OpenCL kernels to GLSL shaders. On the Vulkan level, we needed to set-up all the context required for the buffers and pipelines. We had the opportunity for an optimization here in the process of doing this porting: since ray tracing now takes place on the GPU, the BVH can solely exist on the GPU. As a result, there is less GPU to CPU transfer overhead, which should improve overall run-time performance. Additionally, this allows us to write the BVH as device local, which ensures that the BVH uses fast GPU memory. 

An additional step taken in this implementation was to move the construction of the world bounding box onto the GPU, since it was done on the CPU in the baseline. Furthermore, we changed how Morton codes and their primitives are stored from an Array of Structures (AoS) to a Structure of Arrays (SoA). 

### BVH Measurements

| Measurement | Baseline (ms) | Phase 2 (ms) |
| --- | --- | --- |
| Kernel Bounding Box | 0.089 | 0.085 |
| Kernel Hierarchy Creation | 0.277 | 0.077 |
| Kernel Morton Code | 0.017 | 0.011 |
| Kernel Primitive+World Bounding Box | 14.129 | 0.535 |
| Kernel(s) Radix Sort | 0.558 | 0.519 |
| Total BVH Kernel Time | 15.071 | 1.229 |
| Total BVH Construction Time | 32.692 | 5.404 |

*Note: the sum of individual parts will not equal the total do to minor costs of timing and other overhead.*

#### Conclusions

Across all measurements, we see a significant improvement over BVH times.  In general, we see that the time spent in kernels dropped by ~15x. This is largely explained by moving the world bounding box construction into the GPU. Another notable improvement comes from the hierarchy construction, which benefits the most from the SoA implemented. **In total, we find a ~6x speed-up in total BVH construction time.**



### Run-time and Rendering Measurements

| Measurement | Baseline (ms) | Phase 2 (ms) |
| --- | --- | --- |
| Total Rendering Time | 533.347 | 3.231 |
| Total Run Time | 696.412 | 28.188 |

#### Conclusions

After adding BVH traversal to the GPU ray tracer, **we see a significant performance improvement of ~165x rendering speed-up, and ~24.7x total run-time speed-up**.  Total run-time is now dominated by API initialization, which is unavoidable and not significant enough to try and optimize. However, if we want to take this application further to try and obtain real-time ray tracing, then we would need to optimize both the BVH construction and rendering kernel further. 
