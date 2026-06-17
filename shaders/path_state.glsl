struct PathState {
  vec3 throughput;
  vec3 radiance;
  uint og_path_idx;
  int bounce;
};