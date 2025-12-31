# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: User's Learning Objectives

**The user wants to LEARN OpenCL - DO NOT edit code unless explicitly requested.**

When the user asks about OpenCL concepts or code:
- Explain the concept thoroughly with "why" it's implemented that way
- Reference specific line numbers from the codebase
- Explain trade-offs and design decisions
- Provide context about GPU programming patterns
- Only suggest changes if explicitly asked

**User is on a learning journey - prioritize education over implementation.**

## Project Overview

**Goal:** Implement GPU-accelerated BVH (Bounding Volume Hierarchy) construction using the LBVH (Linear BVH) algorithm from "Fast BVH Construction on GPUs" (Lauterbach et al., 2009).

**Timeline:** December 11, 2024 → January 21, 2025
**Tech Stack:** C++17, OpenCL 1.2+, AMD Radeon 780M GPU
**Starter Code:** Peter Shirley's "Ray Tracing: The Next Week" (modified)

This is an academic project for Advanced Graphics. The goal is to replace the existing CPU-based BVH builder with a GPU-accelerated LBVH implementation while keeping the ray tracer and rendering pipeline intact.

## Git Branch Structure

**Current branch:** `lbvh_opencl` ✓

**Branch evolution:**
```
master (naive_bvh) → lbvh_cpu → lbvh_gpu → lbvh_opencl
```

**Branch descriptions:**
- `master` / `naive_bvh` - Initial CPU BVH with spatial median split (working baseline)
- `lbvh_cpu` - CPU implementation of LBVH algorithm (for validating algorithm logic)
- `lbvh_gpu` - **LBVH algorithms implemented but NOT parallelized** (CPU-based, sequential)
  - Implements: Morton codes, radix sort, Karras tree construction, bottom-up bbox
  - All algorithms are written in C++ and run sequentially on CPU
  - Commit: "GPU algorithms implemented but not parallized" [978d30b]
- `lbvh_opencl` - **OpenCL kernels that parallelize the algorithms** (current)
  - Adds: `lbvh_kernels.cl` with OpenCL kernels
  - Parallelizes: Morton code computation and tree construction
  - Commit: "added subdivide kernel" [b8ae3c6]

**Key insight:** `lbvh_gpu` has all the algorithms working sequentially. `lbvh_opencl` is where those algorithms get parallelized using OpenCL.

## Build & Run

**Build:**
```bash
cmake -B build -S .
cmake --build build
```

**Run (renders to PPM):**
```bash
./build/theNextWeek > output.ppm
```

**Render time:** ~30-60 seconds depending on scene complexity (cornell_box scene is set by default).

**View output:** Use an image viewer that supports PPM format, or convert with ImageMagick:
```bash
convert output.ppm output.png
```

## Code Architecture

### Current Structure

```
raytracer/
├── CMakeLists.txt              # Build configuration
├── test_opencl/                # OpenCL learning/testing (hello.cpp - simple vector add)
└── src/
    ├── external/               # STB image libraries
    │   ├── stb_image.h
    │   └── stb_image_write.h
    ├── bvh/                    # BVH implementations
    │   ├── bvh.h               # CPU BVH (spatial median split) - BASELINE
    │   ├── lbvh_cpu.h          # CPU LBVH (smart pointers, recursive)
    │   ├── lbvh_gpu.h          # LBVH with algorithms (419 lines) - MAIN IMPLEMENTATION
    │   ├── lbvh_kernels.cl     # OpenCL kernels (179 lines) - GPU ACCELERATION
    │   ├── morton.h            # Morton code primitive struct
    │   └── bkup_bvh.h          # Backup of original BVH
    └── raytracer/              # Ray tracer core
        ├── main.cc             # Entry point with test scenes
        ├── aabb.h              # Axis-aligned bounding box
        ├── camera.h            # Camera & rendering loop
        ├── hittable.h          # Ray-object intersection interface
        ├── sphere.h, quad.h    # Geometric primitives
        ├── material.h          # BSDF materials
        └── [other files]
```

### Key Components

**CPU BVH Baseline (src/bvh/bvh.h):**
- **Status:** ✅ Complete and working reference implementation
- **Node structure:** `bvh_node {aabb bbox, int leftFirst, int primCount}` - flat array layout
- **Construction algorithm:** Spatial median split (NOT SAH)
  - Splits along longest axis at bbox midpoint
  - In-place partitioning using swap
  - O(N log N) build time on CPU
- **Storage:** Flat `pool` vector (size 2N-1), GPU-friendly structure
- **Traversal:** Iterative using node indices (bvh.h:86-113)
- **Features:** Statistics tracking, tree visualization, depth computation

**LBVH GPU Implementation (src/bvh/lbvh_gpu.h - 419 lines):**
- **Status:** ⚠️ Contains LBVH algorithms but currently CPU-only (not using OpenCL yet)
- **Purpose:** This is where the LBVH algorithm lives - main implementation file
- **Node structure:** `lbvh_node` with pointers (left/right/parent) and primitive_id
  - Lines 34-56: Node struct definition
  - Pointer-based tree (different from flat array in bvh.h)
- **Construction phases (all currently CPU):**
  1. **Morton code computation** (341-359): `compute_morton(point3 barycenter)`
     - 30-bit Morton codes (10 bits per dimension)
     - Bit interleaving for spatial locality
  2. **Radix sort** (361-379): `radix_sort()`
     - Sorts morton_list by Morton code
     - LSB to MSB, stable sort
  3. **Tree construction** (189-221): `subdivide()`
     - Implements Karras (2012) parallel algorithm
     - Uses `find_range()` (259-296) and `find_split()` (313-339)
     - Builds binary radix tree from sorted Morton codes
  4. **Bounding box propagation** (223-257): `bottom_up_bbox_build()`
     - Queue-based bottom-up AABB calculation
     - Waits for children before computing parent bbox
- **Key algorithms:**
  - `delta()` (298-305): Count leading zeros - determines longest common prefix
  - `find_range()` (259-296): Karras algorithm to find node range
  - `find_split()` (313-339): Binary search for optimal split point

**OpenCL Kernels (src/bvh/lbvh_kernels.cl - 179 lines):**
- **Status:** ⚠️ Kernels written but NOT integrated with lbvh_gpu.h yet
- **Purpose:** GPU-accelerated versions of algorithms from lbvh_gpu.h
- **Structure:**
  - Lines 1-27: OpenCL struct definitions (point3, aabb, morton_primitive, int2, lbvh_node)
  - Lines 29-103: Helper functions (delta, find_range, find_split) - GPU versions
  - Lines 105-140: `compute_morton` kernel - parallelizes Morton code computation
  - Lines 142-179: `create_hierarchy` kernel - parallelizes tree construction
- **Key differences from C++ version:**
  - Uses `__global` qualifier for GPU memory pointers
  - Uses `get_global_id(0)` to identify work item (thread)
  - Uses OpenCL built-in `clz()` instead of `__builtin_clz()`
  - Structs use plain types (no pointers, no std::)
  - Node structure uses indices instead of pointers (lines 23-27)

**OpenCL Test Program (test_opencl/hello.cpp):**
- **Purpose:** Learning OpenCL basics - simple vector addition
- **What it demonstrates:**
  1. Platform/device selection (lines 18-23)
  2. Context and command queue creation (lines 27-28)
  3. Buffer creation and data transfer (lines 35-37)
  4. Kernel compilation (lines 40-48)
  5. Kernel argument setting (lines 52-54)
  6. Kernel execution (line 60)
  7. Result retrieval (line 61)
  8. Resource cleanup (lines 94-100)
- **Performance comparison:** Compares GPU vs CPU execution time (10M elements)

**Main Rendering Loop (src/raytracer/main.cc):**
- Line 449: `switch(7)` statement selects which scene to render
- Line 456: Default scene is `cornell_box()` - 8 primitives (6 quads + 2 boxes)
- Line 450: `bouncing_spheres()` - ~400-500 primitives for performance testing
- Cornell box scene setup (lines 249-291):
  - Creates hittable_list with primitives
  - Line 289: `lbvh_gpu world_bvh(world.objects)` - builds BVH
  - Line 290: `cam.render(world_bvh)` - renders scene
- Scene prints tree structure and statistics to stderr

**AABB (aabb.h):**
- Stores 3 intervals (x, y, z ranges)
- `hit()` method: Ray-box intersection using slab method
- `longest_axis()`: Returns index (0=x, 1=y, 2=z) of longest dimension

**Morton Code Structure (src/bvh/morton.h):**
- `struct morton_primitive {uint32_t morton_code, primitive_id;}`
- 30-bit codes: 10 bits per dimension (x,y,z) interleaved
- Sorted by Morton code to preserve spatial locality

## LBVH Algorithm Overview

**Key Difference from SAH BVH:**
- **SAH (bvh.h):** High-quality BVH, slow build (recursive sorting), optimal ray tracing performance
- **LBVH (lbvh_gpu.h):** Lower-quality BVH, fast parallel build on GPU, acceptable ray tracing performance

**LBVH Construction Pipeline (4 phases):**

1. **Morton Code Computation:**
   - **Purpose:** Map each primitive's centroid to a 30-bit Morton code
   - **Why:** Space-filling curve preserves spatial locality - nearby objects get similar codes
   - **Implementation:** Bit interleaving of x, y, z coordinates (10 bits each)
   - **Parallelization:** Each primitive is independent → perfect for GPU parallelism

2. **Radix Sort:**
   - **Purpose:** Sort primitives by Morton code
   - **Why:** Sorted Morton codes → tree construction becomes trivial (no recursion needed)
   - **Implementation:** Current: CPU radix sort; Future: GPU radix sort (parallel)
   - **Trade-off:** Sorting time vs construction simplicity

3. **Tree Construction (Karras 2012 algorithm):**
   - **Purpose:** Build binary radix tree from sorted Morton codes
   - **Why:** Fully parallel - each internal node can be constructed independently
   - **Key insight:** Sorted Morton codes define tree structure implicitly
   - **Algorithm:**
     - For each internal node i (parallel):
       - Find range covered by node: `find_range(i)` - binary search
       - Find split point in range: `find_split(start, end)` - binary search
       - Assign left/right children based on split
   - **Parallelization:** All internal nodes processed in parallel (N-1 threads)

4. **Bounding Box Calculation:**
   - **Purpose:** Bottom-up AABB propagation from leaves to root
   - **Why:** Leaves have primitive bboxes, internal nodes need children's bboxes
   - **Implementation:**
     - Current (lbvh_gpu.h): Queue-based CPU traversal
     - Future: Atomic operations on GPU for synchronization
   - **Challenge:** Parent depends on both children → synchronization needed

**Expected Performance:**
- Build time: 20-30× faster than CPU SAH BVH on large scenes (1M+ triangles)
- Traversal: ~20-30% slower than SAH BVH (acceptable trade-off for dynamic scenes)
- See `../Statistics/` for actual benchmark results

## OpenCL Concepts in This Project

### Memory Hierarchy and Qualifiers

**Why OpenCL uses memory qualifiers:**
GPUs have different memory spaces with different speeds:
- Global memory: Large (GB), slow, accessible by all threads
- Local memory: Small (KB), fast, shared within work-group
- Private memory: Tiny (registers), fastest, per-thread

**In lbvh_kernels.cl:**
- `__global const morton_primitive* morton_list` (line 31, 40, 77)
  - **Why `__global`:** Morton list is large (N primitives), must be in global memory
  - **Why `const`:** Read-only access → GPU can optimize caching
  - **Trade-off:** Global memory is slow, but unavoidable for large data

**Future optimization:**
- Use `__local` memory for tiles of Morton codes in radix sort
- Cache frequently accessed data in local memory (work-group shared)

### Parallelization Pattern: Data Parallel

**What it means:**
- Same operation on different data elements
- Each GPU thread (work item) processes one element
- All threads execute same kernel code

**In compute_morton kernel (lines 107-140):**
```c
int i = get_global_id(0);  // Thread ID
if (i >= N) return;        // Bounds check
// Compute Morton code for primitive i
morton_list[i].morton_code = code;
```
- **Why this pattern:** N primitives → N threads, each computes one Morton code
- **Benefit:** Perfect parallelism - no dependencies between threads
- **GPU advantage:** Can run thousands of threads simultaneously

**In create_hierarchy kernel (lines 142-179):**
```c
int i = get_global_id(0);  // Thread ID for internal node i
if (i >= N-1) return;      // N-1 internal nodes
// Build tree structure for node i
```
- **Why this pattern:** N-1 internal nodes → N-1 threads
- **Key insight:** Karras algorithm allows independent node construction
- **Synchronization:** Parent pointers need atomic writes (not implemented yet)

### Work Item Organization

**What is global_work_size:**
- Number of threads to launch
- For compute_morton: N (one thread per primitive)
- For create_hierarchy: N-1 (one thread per internal node)

**Why get_global_id(0):**
- Gets unique thread ID in 1D work space
- Used as array index: `morton_list[i]`, `nodes[i]`
- **Why 1D:** Our data is 1D array of primitives

**Future: Work groups and local size:**
- Threads organized into work-groups for local memory sharing
- Useful for radix sort (share data within group)

### Helper Functions: delta, find_range, find_split

**Why inline:**
- `inline` keyword (lines 31, 40, 77) → GPU inlines function calls
- **Benefit:** Avoids function call overhead on GPU
- **Trade-off:** Code duplication vs performance

**delta function (lines 31-38):**
```c
inline int delta(int i, int j, __global const morton_primitive* morton_list, int N)
```
- **Purpose:** Count leading zeros in XOR of Morton codes
- **Why:** Determines longest common prefix → measures similarity
- **OpenCL built-in:** `clz()` (count leading zeros) - hardware instruction
- **GPU advantage:** Single instruction vs loop on CPU

**find_range function (lines 40-74):**
- **Purpose:** Find range of primitives covered by internal node i (Karras algorithm)
- **Why exponential search then binary search:** O(log n) complexity
- **Parallelization:** Each thread independently finds its range (no communication)

**find_split function (lines 77-103):**
- **Purpose:** Find split point in range [start, end]
- **Why binary search:** O(log n) to find split that maximizes common prefix
- **Key insight:** Sorted Morton codes → split point is deterministic

### Node Indexing Scheme (lines 159-171)

**Why use indices instead of pointers:**
```c
int left_node;
if (split == start) { // Is leaf
  left_node = split + N - 1;  // Leaf index
} else {
  left_node = split;          // Internal node index
}
```

- **CPU version (lbvh_gpu.h):** Uses pointers (`lbvh_node* left`)
- **GPU version (lbvh_kernels.cl):** Uses indices (`int left`)
- **Why:** GPU global memory doesn't support pointers between threads
- **Layout:** Internal nodes [0..N-2], Leaf nodes [N-1..2N-2]
- **Trade-off:** Indirection cost vs memory safety

### Missing Pieces (Not Yet Implemented)

**1. Radix sort kernel:**
- Current: CPU radix sort in lbvh_gpu.h (lines 361-379)
- Needed: GPU parallel radix sort kernel
- **Why parallel radix sort is hard:** Requires synchronization between passes
- **Approach:** Use local memory for buckets, barriers for synchronization

**2. Bounding box kernel:**
- Current: CPU queue-based traversal (lbvh_gpu.h:223-257)
- Needed: GPU bottom-up bbox propagation
- **Challenge:** Parent needs both children → atomic operations for synchronization
- **Approach:** Use atomics to count children, process parent when count=2

**3. Host-device integration:**
- Current: Kernels exist but not called from C++
- Needed: OpenCL setup in lbvh_gpu.h (context, queue, buffers)
- **Reference:** See test_opencl/hello.cpp for OpenCL boilerplate
- **Steps:**
  1. Create context/queue (like hello.cpp:27-28)
  2. Create buffers for morton_list, nodes, centroids
  3. Load and compile lbvh_kernels.cl
  4. Enqueue kernels with proper arguments
  5. Read results back to CPU

## Development Status

### Completed ✅

- ✅ CPU BVH baseline (bvh.h) - working and tested
- ✅ LBVH algorithm implementation (lbvh_gpu.h) - CPU version working
- ✅ Morton code computation (CPU)
- ✅ Radix sort (CPU)
- ✅ Karras tree construction (CPU)
- ✅ Bottom-up bbox propagation (CPU)
- ✅ OpenCL kernels written (lbvh_kernels.cl) - compute_morton and create_hierarchy
- ✅ OpenCL test program (test_opencl/hello.cpp) - basic vector add working
- ✅ Statistics and benchmarking framework
- ✅ Tree visualization and validation

### In Progress / Not Yet Done ⚠️

- ⚠️ **OpenCL integration:** Kernels not called from lbvh_gpu.h yet
- ⚠️ **Radix sort kernel:** Still using CPU version
- ⚠️ **Bounding box kernel:** Still using CPU queue-based version
- ⚠️ **Host-device data transfer:** Buffer management not implemented
- ⚠️ **Kernel compilation in CMake:** lbvh_kernels.cl needs to be loaded at runtime

### Next Steps (In Order)

1. **Learn OpenCL fundamentals:** Study test_opencl/hello.cpp pattern
2. **Integrate compute_morton kernel:** Replace lbvh_gpu.h:341-359 with OpenCL kernel call
3. **Integrate create_hierarchy kernel:** Replace lbvh_gpu.h:189-221 with OpenCL kernel call
4. **Implement radix sort kernel:** Write and integrate parallel radix sort
5. **Implement bbox kernel:** Write and integrate bottom-up bbox propagation
6. **Benchmark:** Compare GPU vs CPU build times
7. **Optimize:** Tune work-group sizes, memory access patterns

## Common Build and Run Instructions

**Scene selection:**
- Edit main.cc line 449: `switch(7)` - change number to select scene
- Case 7: cornell_box (8 primitives) - fast, good for testing
- Case 1: bouncing_spheres (400+ primitives) - medium complexity
- Case 9: final_scene (27,000+ primitives) - stress test

**Performance testing:**
- Reduce samples_per_pixel for faster renders (cornell_box line 263: default 200)
- Enable statistics: BVH constructor automatically tracks stats
- Output goes to stderr (tree structure, timing, node visits)

**Build for OpenCL:**
```bash
# Check if OpenCL is available
ls /usr/lib/libOpenCL.so    # Should exist

# Build with OpenCL
cmake -B build -S .
cmake --build build

# If linking errors, add to CMakeLists.txt:
# target_link_libraries(theNextWeek OpenCL)
```

**Debugging OpenCL kernels:**
- Check kernel compilation errors: clGetProgramBuildInfo (see hello.cpp:43-48)
- Print from kernels: Use printf() in OpenCL C (requires OpenCL 1.2+)
- Validate results: Compare GPU output with CPU version

**Common Issues:**

**"Could not load image file 'earthmap.jpg'"**
- main.cc line 449: Make sure switch is set to 7 (cornell_box)
- Cases 3 and 9 require earthmap.jpg which was removed

**Slow rendering:**
- Default cornell_box uses 200 samples/pixel (high quality)
- For faster testing, edit line 263 to reduce samples_per_pixel (try 10-50)

**Build fails after adding new files:**
- Re-run `cmake -B build -S .` when CMakeLists.txt changes
- Add new source files to CMakeLists.txt variables

**OpenCL errors:**
- CL_DEVICE_NOT_FOUND: Check GPU drivers (ROCm for AMD, CUDA for NVIDIA)
- CL_BUILD_PROGRAM_FAILURE: Check kernel compilation log (see hello.cpp:43-48)
- CL_INVALID_WORK_GROUP_SIZE: Reduce work-group size or use NULL for local_work_size

## References

**Primary Paper:** Lauterbach et al., "Fast BVH Construction on GPUs" (2009) - see ../LBVHPaper.pdf
**Tree Construction:** Karras (2012) "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"
**GPU Sorting:** ../GPUSORT.pdf in project root

**Starter Code Attribution:**
- Base code: Peter Shirley's "Ray Tracing in One Weekend" series (Public Domain, CC0)
- Modified from original: Removed InOneWeekend and TheRestOfYourLife, kept only TheNextWeek for BVH

## Statistics and Benchmarking

Results stored in `../Statistics/` directory:

**GPU Implementation (lbvh_opencl branch):**
- File: `lbvh_gpu_stats.txt`
- Scene: cornell_box (8 primitives)
- BVH construction: 0.052309 ms (extremely fast!)
- Rendering: 16990.1 ms
- Avg node visits/ray: 85.19
- Avg primitive tests/ray: 7.68

**CPU LBVH Baseline (27K primitives):**
- File: `lbvh_cpu_baseline_27k_primitives.txt`
- Scene: Unknown large scene (27,001 primitives)
- BVH construction breakdown:
  1. Morton code computation: 5.12 ms
  2. Radix sort: 19.50 ms
  3. Tree construction: 10.56 ms
  4. Bounding box computation: 11.60 ms
  - **Total: 66.94 ms**
- Rendering: 7.87 ms (50×50, fewer samples)
- Avg node visits/ray: 5.37
- Avg primitive tests/ray: 0.24

**Key insights:**
- Small scenes (8 prims): GPU overhead dominates, CPU wins
- Large scenes (27K prims): GPU parallelism pays off
- Tree quality: CPU LBVH has better traversal stats (fewer visits/tests)
