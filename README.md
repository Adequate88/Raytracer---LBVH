# GPU-Accelerated BVH Ray Tracer

A ray tracing implementation featuring GPU-accelerated BVH (Bounding Volume Hierarchy) construction using the LBVH algorithm. Built on top of Peter Shirley's "Ray Tracing: The Next Week" codebase.

## Build Instructions

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

For release/debug builds:
```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
```

## Running

```bash
./build/theNextWeek
```

The program runs a BVH performance comparison across four benchmark models (Bunny, Armadillo, Dragon, Lucy), testing three BVH implementations for each. The final submission only includes the Bunny and Armadillo models since the other file size of the other models is too large.

## Output

**Images:** Rendered images are saved to `Data/` as PPM files:
- `Data/{model}_karras_gpu.ppm` - GPU LBVH result
- `Data/{model}_karras_cpu.ppm` - CPU LBVH result
- `Data/{model}_baseline.ppm` - Baseline BVH result

**Statistics:** Performance data saved as CSV files:
- `Data/{model}_{bvh_type}.csv` - Contains construction time, ray counts, and traversal statistics

**Convert to PNG (optional):**
```bash
convert Data/bunny_karras_gpu.ppm Data/bunny_karras_gpu.png
```

## Configuration

Edit `src/raytracer/main.cc` to modify:

**Resolution and samples** (line ~228-229):
```cpp
const int IMAGE_WIDTH = 1280;
const int SAMPLES = 100;
```

**Select specific models** - Comment out unwanted experiments in `main()`:
```cpp
int main() {
    // Run only bunny:
    auto config = load_bunny(IMAGE_WIDTH, SAMPLES);
    run_experiment(config);

    // Comment out others if not needed:
    // auto config = load_armadillo(IMAGE_WIDTH, SAMPLES);
    // auto config = load_dragon(IMAGE_WIDTH, SAMPLES);
    // auto config = load_lucy(IMAGE_WIDTH, SAMPLES);
}
```

**Model files** are loaded from `models/` directory. Supported: OBJ format.

## Project Structure

```
raytracer/
├── src/
│   ├── bvh/                      # BVH implementations
│   │   ├── bvh.h                 # Baseline BVH (spatial median split)
│   │   ├── lbvh_karras_cpu.h     # CPU LBVH (Karras 2012 algorithm)
│   │   ├── lbvh_karras_gpu.h     # GPU LBVH (OpenCL accelerated)
│   │   ├── morton.h              # Morton code primitives
│   │   ├── opencl_shared_types.cl    # Shared OpenCL type definitions
│   │   └── lbvh_construction_kernels.cl  # BVH construction kernels
│   │
│   ├── raytracer/                # Ray tracing core
│   │   ├── main.cc               # Entry point and experiment runner
│   │   ├── camera.h              # Camera and render loop
│   │   ├── gpu_renderer.h        # GPU path tracing
│   │   ├── gpu_renderer_kernels.cl   # Rendering kernels
│   │   ├── triangle.h            # Triangle primitive
│   │   ├── mesh.h                # OBJ mesh loader
│   │   └── [other raytracer files from Shirley]
│   │
│   └── external/                 # Third-party libraries
│       ├── stb_image.h
│       └── tinyobjloader.h
│
├── models/                       # OBJ model files
├── Data/                         # Output images and statistics
└── CMakeLists.txt
```

## Additions to Peter Shirley's Codebase

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

## Algorithm Overview

**LBVH Construction Pipeline:**
1. Compute scene bounding box
2. Calculate Morton codes for each primitive centroid
3. Sort primitives by Morton code (radix sort)
4. Build hierarchy using Karras's parallel algorithm
5. Propagate bounding boxes bottom-up using atomic counters

**GPU Rendering Pipeline:**
1. Convert scene to GPU-compatible format (spheres, triangles, materials)
2. Upload BVH nodes and primitives to GPU
3. Launch render kernel (one thread per pixel)
4. Accumulate samples with jittered rays
5. Read back pixel data and statistics

## References

- Karras, T. (2012). "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"
- Lauterbach et al. (2009). "Fast BVH Construction on GPUs"
- Shirley, P. "Ray Tracing: The Next Week" (base code, CC0 Public Domain)

## DISCLAIMER

This README was primarily generated by AI and then edited for correctness.
