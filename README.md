# AdvGraphics-P2: Fully Parallel LBVH Implementation

Linear Bounding Volume Hierarchies (LBVHs) use Morton codes to transform hierarchy construction into a parallelizable sorting problem, enabling much faster construction at the expense of slightly reduced tree quality. This project implements a fully GPU-parallel LBVH construction algorithm based on the work of Lauterbach and Karras, using OpenCL.

## Implementation

The implementation is broken down into four GPU kernel stages: Morton code computation, radix sort, hierarchy construction, and bounding box propagation. It is evaluated against CPU-based LBVH construction and a naive spatial median BVH, assessing performance based on construction time, bounding-box intersection tests, and ray-primitive intersection tests.

## Results

All tests were performed on an AMD Radeon RX 9060 XT GPU using OpenCL 1.2, rendering at 1280×720 with 100 samples per pixel (~92 million rays per scene), across four benchmark models ranging from 5K to 28M triangles.

For small scenes, kernel launch costs and memory transfers dominate and cause the GPU algorithm to be slower than CPU implementations. However, as scene complexity increases, the strengths of GPU parallelism become apparent — for the Dragon and Lucy models, GPU LBVH construction significantly outperforms the CPU implementations with up to a **4.65× speedup**.

The traversal statistics show the expected trade-off: LBVH trees require slightly more bounding box tests per ray compared to the naive BVH, but the difference is minimal and LBVH still produces trees of comparable quality.

## References
- Lauterbach et al., *Fast BVH Construction on GPUs*, Eurographics 2009
- Karras, *Maximizing Parallelism in the Construction of BVHs*, HPG 2012
