#ifndef RTWEEKEND_CUH
#define RTWEEKEND_CUH

#include <cmath>
#include <limits>
#include <curand_kernel.h>

#include "cuda_helper.cuh"

constexpr double infinity = std::numeric_limits<double>::infinity();

constexpr double pi = 3.1415926535897932385;

__host__ __device__ inline double degrees_to_radians(double degrees) { return degrees * pi / 180; }

__host__ __device__ inline double ffmin(double a, double b) { return a <= b ? a : b; }
__host__ __device__ inline double ffmax(double a, double b) { return a >= b ? a : b; }

__host__ __device__ inline double clamp(double x, double min, double max)
{
  if (x < min) return min;
  if (x > max) return max;
  return x;
}

__device__ inline float curandom(curandState* rand_state)
{
  // Returns a random real in [0,1).
  return curand_uniform(rand_state);
}

__device__ inline float curandom(double min, double max, curandState* rand_state)
{
  // Returns a random real in [min,max).
  return min + (max - min) * curandom(rand_state);
}

__host__ __device__ inline double schlick(double cosine, double ref_idx)
{
  auto r0 = (1 - ref_idx) / (1 + ref_idx);
  r0      = r0 * r0;
  return r0 + (1 - r0) * pow((1 - cosine), 5);
}

#endif  // RTWEEKEND_H