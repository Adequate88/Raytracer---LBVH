<a id="readme-top"></a>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

A ray tracing implementation featuring GPU-accelerated BVH (Bounding Volume Hierarchy) construction using the LBVH algorithm. Built on top of Peter Shirley's "Ray Tracing: The Next Week" codebase.

This project extends "Ray Tracing: The Next Week" with the following:

### 1. GPU-Accelerated LBVH Construction (`lbvh_karras_gpu.h`)

Implements the Karras (2012) parallel BVH construction algorithm on GPU:
- **Morton code computation** - Maps 3D primitive centroids to 1D Morton codes for spatial locality
- **Parallel radix sort** - GPU-based sorting of Morton codes
- **Hierarchy generation** - Parallel construction of internal nodes using the Karras algorithm
- **Bottom-up bounding box propagation** - Atomic operations to build AABBs from leaves to root

### 2. CPU LBVH Reference (`lbvh_karras_cpu.h`)

CPU implementation of the same algorithm for validation and comparison.

### 3. GPU Path Tracing (`gpu_renderer.h`, `gpu_renderer_kernels.cl`)

DISCLAIMER: This portion of the codebase heavily utilized AI to create the code. The main reason being simply to render higher complexity models after BVH construction. Additionally, a GPU path-tracer is just a supplement and was not the main aim of this project, so AI was used to save time.

Full path tracing on GPU:
- Material system (Lambertian, Metal, Emissive)
- Multi-sample anti-aliasing
- Defocus blur support
- BVH traversal with per-ray statistics

### 4. Triangle Mesh Support (`triangle.h`, `mesh.h`)

- Triangle primitive with Möller-Trumbore intersection
- Smooth shading via vertex normal interpolation
- OBJ file loading with automatic normal computation

### 5. Performance Benchmarking

Automated comparison framework:
- Tests three BVH types on four standard models
- Exports construction time, traversal statistics
- Generates rendered images for visual comparison


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Built With


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Starting

**Requirements:**
- C++17 compiler (GCC, Clang, or MSVC)
- CMake 3.1+
- OpenCL 1.2+ with GPU support
- ImageMagick (optional, for PNG conversion)

**Build:**
```bash
mkdir build
cmake -B build -S .
cmake --build build
```

**For release/debug builds:**
```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
```

**Running**

```bash
./build/theNextWeek
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] Clean-up code base
- [ ] Add Vulkan boilerplate
- [ ] Move ray-tracer onto GPU
- [ ] Port BVH construction onto Vulkan
- [ ] Some low level optimizations

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* Peter Shirley

## References

- Karras, T. (2012). "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"
- Lauterbach et al. (2009). "Fast BVH Construction on GPUs"
- Shirley, P. "Ray Tracing: The Next Week" (base code, CC0 Public Domain)
