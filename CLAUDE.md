# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Goal:** Implement GPU-accelerated BVH (Bounding Volume Hierarchy) construction using the LBVH (Linear BVH) algorithm from "Fast BVH Construction on GPUs" (Lauterbach et al., 2009).

**Timeline:** December 11, 2024 → January 21, 2025
**Tech Stack:** C++17, OpenCL 1.2+, AMD Radeon 780M GPU
**Starter Code:** Peter Shirley's "Ray Tracing: The Next Week" (modified)

This is an academic project for Advanced Graphics. The goal is to replace the existing CPU-based BVH builder with a GPU-accelerated LBVH implementation while keeping the ray tracer and rendering pipeline intact.

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
└── src/
    ├── external/               # STB image libraries
    │   ├── stb_image.h
    │   └── stb_image_write.h
    ├── bvh/                    # BVH implementations
    │   ├── bvh.h               # **CPU BVH (spatial median split) - WORKING**
    │   ├── lbvh_cpu.h          # **CPU LBVH implementation - IN PROGRESS**
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

**CPU BVH Implementation (src/bvh/bvh.h):**
- **Status:** ✅ Complete and working
- **Node structure:** `bvh_node {aabb bbox, int leftFirst, int primCount}` - flat array layout
- **Construction algorithm:** Spatial median split (NOT SAH)
  - Splits along longest axis at bbox midpoint
  - In-place partitioning using swap
  - O(N log N) build time on CPU
- **Storage:** Flat `pool` vector (size 2N-1), GPU-friendly structure
- **Traversal:** Iterative using node indices (bvh.h:86-113)
- **Features:** Statistics tracking, tree visualization, depth computation

**AABB (aabb.h):**
- Stores 3 intervals (x, y, z ranges)
- `hit()` method: Ray-box intersection using slab method
- `longest_axis()`: Returns index (0=x, 1=y, 2=z) of longest dimension

**Main Rendering Loop (src/raytracer/main.cc):**
- Line ~404: `switch(7)` statement selects which scene to render
- Default: `cornell_box()` scene (100×100, 200 samples/pixel) - line 275
- Scene functions create `hittable_list`, wrap in `bvh`, then call `cam.render(world_bvh)`
- Cornell box scene prints tree structure and statistics

**Important Note on Image Dependencies:**
- Lines 119 and 365 reference `"earthmap.jpg"` which was removed during cleanup
- These are only used in `earth()` (case 3) and `final_scene()` (case 9/default)
- Current scene (case 7: cornell_box) does NOT require images
- If you get "Could not load image" errors, you're running the wrong scene or need to comment out those lines

**LBVH CPU Implementation (src/bvh/lbvh_cpu.h):**
- **Status:** ⚠️ In progress - has compilation errors
- **Completed components:**
  - ✅ Morton code computation (lbvh_cpu.h:158-177) - 30-bit, 3D interleaved
  - ✅ Radix sort implementation (lbvh_cpu.h:179-197) - LSB to MSB, stable
  - ✅ Split list creation (lbvh_cpu.h:199-207) - finds first differing bit
  - ✅ Bucket sort by level (lbvh_cpu.h:209-225) - O(n) sorting
- **Issues/TODO:**
  - ❌ Line 205: References undefined `level` (should be `first_diff_pos`)
  - ❌ Line 71: `subdivide()` call has wrong parameters (takes 0, calls with 3)
  - ❌ Lines 132-156: `subdivide()` implementation incomplete/broken
  - ❌ Missing statistics tracking (references undefined `stats`)
  - ❌ Line 134-135: `create_bbox()` returns void but assigned to bbox
- **Algorithm:** Follows Lauterbach et al. paper - Morton codes → sort → split list → tree construction

**Morton Code Structure (src/bvh/morton.h):**
- `struct morton_primitive {uint32_t morton_code, primitive_id;}`
- 30-bit codes: 10 bits per dimension (x,y,z) interleaved
- Sorted by Morton code to preserve spatial locality

### Planned Future Components (GPU Implementation)

```
src/
├── opencl/               # OpenCL infrastructure (not yet started)
│   ├── cl_context.h/cpp  # OpenCL setup/management
│   ├── cl_buffer.h/cpp   # GPU memory management
│   └── cl_utils.h/cpp    # Error handling
└── kernels/              # OpenCL kernel files (.cl)
    ├── morton_codes.cl   # Compute Morton codes
    ├── radix_sort.cl     # Parallel sorting (or use library)
    ├── build_tree.cl     # LBVH tree construction (CORE)
    └── compute_bounds.cl # Bottom-up AABB calculation
```

## LBVH Algorithm Overview

**Key Difference from SAH BVH:**
- **SAH (current):** High-quality BVH, slow build (recursive sorting), good ray tracing performance
- **LBVH (goal):** Lower-quality BVH, fast parallel build on GPU, acceptable ray tracing performance

**LBVH Steps:**
1. **Morton Code Computation:** Map each primitive's centroid to a 30-bit Morton code (space-filling curve preserves spatial locality)
2. **Radix Sort:** Sort primitives by Morton code on GPU
3. **Tree Construction:** Build binary radix tree from sorted Morton codes (fully parallel, no recursion)
4. **Bounding Box Calculation:** Bottom-up AABB propagation using atomic operations

**Expected Performance:**
- Build time: 20-30× faster than CPU SAH BVH on large scenes (1M+ triangles)
- Traversal: ~20-30% slower than SAH BVH (acceptable trade-off for dynamic scenes)

## Development Workflow

### Current Phase: Week 2-3 (CPU LBVH Implementation)

**Completed:**
- ✅ Cleaned up starter code (removed unused books, images)
- ✅ Modified CMakeLists.txt to build ray tracer
- ✅ Built and verified ray tracer works
- ✅ Studied BVH implementation (spatial median split, flat array)
- ✅ Implemented Morton code computation (30-bit, 3D interleaved)
- ✅ Implemented radix sort for Morton codes (LSB→MSB)
- ✅ Implemented split list creation (finds first differing bits)
- ✅ Implemented bucket sort by tree level (O(n) sorting)
- ✅ Understood LBVH paper algorithm (split lists, tree construction)

**Current Tasks:**
- [ ] Fix compilation errors in lbvh_cpu.h:
  - [ ] Fix line 205: undefined `level` variable
  - [ ] Fix line 71: `subdivide()` parameter mismatch
  - [ ] Complete `subdivide()` tree construction logic (lines 132-156)
  - [ ] Add statistics tracking (like bvh.h)
  - [ ] Fix bbox creation return type issue
- [ ] Implement tree construction from sorted split list
- [ ] Test LBVH with cornell_box scene
- [ ] Compare build times: CPU BVH vs CPU LBVH
- [ ] Validate tree correctness (rendering should match)

**Future Tasks (Weeks 4-6):**
- [ ] Install ROCm/OpenCL SDK for AMD GPU
- [ ] Implement GPU LBVH in OpenCL
- [ ] Write OpenCL kernels (Morton codes, radix sort, tree build)
- [ ] Benchmark GPU vs CPU build times

### Important Notes for Future Work

**When implementing LBVH:**
1. **Do NOT modify** existing `bvh.h` - keep it as CPU reference
2. **Create parallel implementation** in `src/bvh/lbvh_gpu.h`
3. **BVH node structure compatibility:** LBVH must produce nodes compatible with existing `hit()` traversal code
4. **Test incrementally:** Validate each stage (Morton codes, sort, tree build, bounds) before moving to next
5. **Use simple scenes first:** Test with cornell_box or bouncing_spheres before large scenes

**Morton Code Implementation:**
- **30-bit code:** 10 bits per dimension (x, y, z) - `k=10`, so `3k=30`
- **Bit interleaving:** x|y|z|x|y|z|... pattern (space-filling curve)
- **Normalization:** Primitive centroids normalized to [0, cell_count) where cell_count = 2^k = 1024
- **Formula:** For bit position calculation, use `31 - __builtin_clz(diff)` not `29 - __builtin_clz(diff)`
- **Handles duplicates:** primitive_id used as tie-breaker

**Split List Algorithm (Key Insights):**
- **Purpose:** Determines where to partition primitive sequence at each tree level
- **Creation:** For each adjacent pair (i, i+1) in sorted Morton sequence:
  - XOR their Morton codes to find differing bits
  - Find first (most significant) differing bit: `first_diff = 31 - __builtin_clz(xor_result)`
  - Record split: `(i, first_diff)` - just the FIRST diff, not all levels from h to 3k
- **Optimization:** Paper describes adding splits from h to 3k (redundant), but you only need the first diff
- **Sorting:** Use bucket sort O(n) since levels are in small range [0, 30]
- **Usage:** After sorting by level, splits tell you where to partition at each tree depth

**Memory Management:**
- OpenCL requires explicit CPU↔GPU memory transfers
- Structure of Arrays (SoA) layout for GPU memory coalescing
- Keep BVH data on GPU, only transfer final structure to CPU if needed

**Test Scenes:**
- `cornell_box()`: 32 triangles, sanity check
- `bouncing_spheres()`: ~400 spheres, medium complexity
- For stress testing: Generate scenes with 100K+ primitives

## C++ Implementation Notes

**Using std::pair:**
```cpp
std::pair<int, int> split;  // (index, level)
int index = split.first;    // Access first element
int level = split.second;   // Access second element

// Creation:
split_list.push_back({3, 2});  // Brace initialization

// C++17 structured bindings:
for (const auto& [idx, lvl] : split_list) {
    // Use idx and lvl directly
}
```

**Important:** `.first` and `.second` are member variables, not functions (no parentheses!)

## Common Issues

**"Could not load image file 'earthmap.jpg'"**
- main.cc line 404: Change switch value from 9 or default to 7 (cornell_box)
- Or comment out lines 119 and 365 that reference image textures

**Slow rendering:**
- Default cornell_box uses 200 samples/pixel (high quality)
- For faster testing, edit line 263 in cornell_box() to reduce samples_per_pixel

**Build fails after adding new files:**
- Re-run `cmake -B build -S .` when CMakeLists.txt changes
- Add new source files to CMakeLists.txt variables (e.g., SOURCE_NEXT_WEEK)

## References

**Primary Paper:** Lauterbach et al., "Fast BVH Construction on GPUs" (2009) - see LBVHPaper.pdf in project root
**Detailed Plan:** ~/.claude/plans/nested-mapping-moore.md (6-week implementation timeline)
**Project Options:** Organization/project_options.md (shows this is "1b. GPU BVH Construction (LBVH)", Medium difficulty)

**Starter Code Attribution:**
- Base code: Peter Shirley's "Ray Tracing in One Weekend" series (Public Domain, CC0)
- Modified from original: Removed InOneWeekend and TheRestOfYourLife, kept only TheNextWeek for BVH
