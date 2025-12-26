#ifndef MORTON_H
#define MORTON_H

#include <cstdint>

struct morton_primitive {
  uint32_t morton_code;
  uint32_t primitive_id;
};


#endif
