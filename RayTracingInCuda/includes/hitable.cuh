#ifndef HITABLE_CUH
#define HITABLE_CUH

#include "cuda_helper.cuh"
#include "vec3.cuh"
#include "ray.cuh"

class Material;

struct HitRecord {
  double t_;
  Vec3 p_;
  Vec3 normal_;
  Material* mat_ptr_;
  bool front_face_;

  __device__ inline void SetFaceNormal(const Ray& r, const Vec3& outward_normal)
  {
    front_face_ = dot(r.Direction(), outward_normal) < 0;
    normal_     = front_face_ ? outward_normal : -outward_normal;
  }
};

class Hitable
{
 public:
  __device__ virtual bool Hit(const Ray& ray, double tmin, double tmax, HitRecord& record) const = 0;
};
#endif