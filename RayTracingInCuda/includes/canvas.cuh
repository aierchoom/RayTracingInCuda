#ifndef CANVAS_CUH
#define CANVAS_CUH

#include "cuda_helper.cuh"
#include "vec3.cuh"

class Canvas
{
 public:
  __host__ Canvas(int width, int height) : d_data_(nullptr), width_(width), height_(height) { CheckCudaErrors(cudaMallocManaged(&d_data_, width_ * height_ * sizeof(Vec3))); }

  __host__ ~Canvas() { cudaFree(d_data_); }

 public:
  Vec3* d_data_;

  // saved by column pixel.
  int width_;
  int height_;
};

#endif  // CANVAS_CUH