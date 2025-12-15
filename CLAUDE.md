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

### Current Structure (Peter Shirley's Base)

```
raytracer/
├── CMakeLists.txt              # Build configuration (simplified to only build TheNextWeek)
└── src/
    ├── external/               # STB image libraries
    │   ├── stb_image.h
    │   └── stb_image_write.h
    └── TheNextWeek/            # Ray tracer with CPU BVH
        ├── main.cc             # Entry point with test scenes
        ├── bvh.h               # **CPU BVH implementation (SAH-based, recursive)**
        ├── aabb.h              # Axis-aligned bounding box
        ├── camera.h            # Camera & rendering loop
        ├── hittable.h          # Ray-object intersection interface
        ├── sphere.h, quad.h    # Geometric primitives
        ├── material.h          # BSDF materials
        └── [other files]
```

### Key Components

**BVH Implementation (bvh.h):**
- `bvh_node` class: Binary tree node containing AABB and left/right children
- **Construction algorithm:** Recursive SAH-based (Surface Area Heuristic) partitioning
  - Splits objects along longest axis of bounding box
  - Uses `std::sort` to partition primitives
  - O(N log N) build time on CPU
- **Structure:** Stores `shared_ptr<hittable>` for left/right children (can be leaf objects or internal nodes)
- **Traversal:** Recursive ray-AABB intersection test, then recurse into children

**AABB (aabb.h):**
- Stores 3 intervals (x, y, z ranges)
- `hit()` method: Ray-box intersection using slab method
- `longest_axis()`: Returns index (0=x, 1=y, 2=z) of longest dimension

**Main Rendering Loop (main.cc):**
- Line 404: `switch(7)` statement selects which scene to render
- Default: `cornell_box()` scene (600×600, 200 samples/pixel)
- Scene functions create `hittable_list`, wrap in `bvh_node`, then call `cam.render(world)`

**Important Note on Image Dependencies:**
- Lines 119 and 365 reference `"earthmap.jpg"` which was removed during cleanup
- These are only used in `earth()` (case 3) and `final_scene()` (case 9/default)
- Current scene (case 7: cornell_box) does NOT require images
- If you get "Could not load image" errors, you're running the wrong scene or need to comment out those lines

### Planned Structure (LBVH Implementation)

The LBVH implementation will add the following (based on project plan in ~/.claude/plans/nested-mapping-moore.md):

```
src/
├── TheNextWeek/           # Keep existing (CPU reference)
├── bvh/
│   ├── lbvh_cpu.h/cpp    # CPU LBVH reference (Week 3)
│   └── lbvh_gpu.h/cpp    # GPU LBVH implementation (Week 4)
├── opencl/
│   ├── cl_context.h/cpp  # OpenCL setup/management
│   ├── cl_buffer.h/cpp   # GPU memory management
│   └── cl_utils.h/cpp    # Error handling
└── kernels/               # OpenCL kernel files (.cl)
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

### Current Phase: Week 1 (Setup & Learning)

**Completed:**
- ✅ Cleaned up starter code (removed unused books, images)
- ✅ Modified CMakeLists.txt to only build TheNextWeek
- ✅ Built and verified ray tracer works

**Remaining Week 1 Tasks:**
- [ ] Study Peter Shirley's BVH implementation (bvh.h) thoroughly
- [ ] Benchmark CPU BVH build time (baseline for comparison)
- [ ] Install ROCm/OpenCL SDK for AMD GPU
- [ ] Write simple test OpenCL kernel to verify GPU works

### Important Notes for Future Work

**When implementing LBVH:**
1. **Do NOT modify** existing `bvh.h` - keep it as CPU reference
2. **Create parallel implementation** in `src/bvh/lbvh_gpu.h`
3. **BVH node structure compatibility:** LBVH must produce nodes compatible with existing `hit()` traversal code
4. **Test incrementally:** Validate each stage (Morton codes, sort, tree build, bounds) before moving to next
5. **Use simple scenes first:** Test with cornell_box or bouncing_spheres before large scenes

**Morton Code Implementation (from plan):**
- 30-bit code: 10 bits per dimension (x, y, z)
- Bit interleaving: x|y|z|x|y|z|... pattern
- Normalize primitive centroids to [0, 1] before encoding
- Handle duplicate Morton codes with tie-breaking using primitive IDs

**Memory Management:**
- OpenCL requires explicit CPU↔GPU memory transfers
- Structure of Arrays (SoA) layout for GPU memory coalescing
- Keep BVH data on GPU, only transfer final structure to CPU if needed

**Test Scenes:**
- `cornell_box()`: 32 triangles, sanity check
- `bouncing_spheres()`: ~400 spheres, medium complexity
- For stress testing: Generate scenes with 100K+ primitives

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
