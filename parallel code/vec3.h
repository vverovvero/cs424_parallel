#ifndef VEC3_H
#define VEC3_H

#include <math.h>

struct vec3{
  float x;
  float y;
  float z;
};

__device__ vec3 UP     = { .x=0.0f, .y=1.0f, .z=0.0f };
__device__ vec3 ZERO   = { .x=0.0f, .y=0.0f, .z=0.0f };
__device__ vec3 WHITE  = { .x=255.0f, .y=255.0f, .z=255.0f };

__device__ float dotProduct(vec3 a, vec3 b) {
  return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

__device__ float length(vec3 a) {
  return sqrtf(dotProduct(a, a));
}

__device__ vec3 crossProduct(vec3 a, vec3 b) {
  vec3 ret_vec = {
    .x = (a.y * b.z) - (a.z * b.y),
    .y = (a.z * b.x) - (a.x * b.z),
    .z = (a.x * b.y) - (a.y * b.x)
  };
  return ret_vec;
}

__device__ vec3 scale(vec3 a, float t) {
  vec3 ret_vec =  {
    .x = a.x * t,
    .y = a.y * t,
    .z = a.z * t
  };
  return ret_vec;
}

__device__ vec3 compScale(vec3 a, vec3 b) {
  vec3 ret_vec =  {
    .x = a.x * b.x,
    .y = a.y * b.y,
    .z = a.z * b.z
   };
   return ret_vec;
}

__device__ vec3 unitVector(vec3 a) {
  return scale(a, 1.0f / length(a));
}

__device__ vec3 add(vec3 a, vec3 b) {
  vec3 ret_vec =  {
      .x = a.x + b.x,
      .y = a.y + b.y,
      .z = a.z + b.z
  };
  return ret_vec;
}

__device__ vec3 add3(vec3 a, vec3 b, vec3 c) {
  vec3 ret_vec =  {
      .x = a.x + b.x + c.x,
      .y = a.y + b.y + c.y,
      .z = a.z + b.z + c.z
  };
  return ret_vec;
}

__device__ vec3 subtract(vec3 a, vec3 b) {
  vec3 ret_vec =  {
      .x = a.x - b.x,
      .y = a.y - b.y,
      .z = a.z - b.z
  };
  return ret_vec;
}

__device__ vec3 reflectThrough(vec3 a, vec3 normal) {
    vec3 d = scale(normal, dotProduct(a, normal));
    return subtract(scale(d, 2.0f), a);
}
#endif
