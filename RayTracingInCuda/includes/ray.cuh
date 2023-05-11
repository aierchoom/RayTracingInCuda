#ifndef RAY_CUH
#define RAY_CUH

#include "vec3.cuh"

class Ray
{
 public:
  __device__ Ray() {}
  __device__ Ray(const Vec3& a, const Vec3& b) : a_(a), b_(b) {}
  __device__ Vec3 Origin() const { return a_; }
  __device__ Vec3 Direction() const { return b_; }
  __device__ Vec3 At(float t) const { return a_ + t * b_; }

 private:
  Vec3 a_;
  Vec3 b_;
};
#endif