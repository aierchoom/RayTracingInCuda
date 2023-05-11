#ifndef SPHERE_CUH
#define SPHERE_CUH
#include "hitable.cuh"
#include "vec3.cuh"

class Sphere : public Hitable
{
 public:
  __device__ Sphere() : center_(), radius_(), mat_ptr_(nullptr) {}
  __device__ Sphere(Vec3 center, double radius, Material* mat_ptr) : center_(center), radius_(radius), mat_ptr_(mat_ptr){};

  __device__ virtual bool Hit(const Ray& ray, double tmin, double tmax, HitRecord& record) const
  {
    Vec3 origin_to_center = ray.Origin() - center_;
    // dot(B,B)
    float a = dot(ray.Direction(), ray.Direction());

    // b=2h, h*h-ac
    float half_b = dot(ray.Direction(), origin_to_center);

    // dot(A-C,A-C)-R*R
    float c = dot(origin_to_center, origin_to_center) - radius_ * radius_;
    // t*t*dot(B,B)+2*t*dot(B,A-C)+dot(A-C,A-C)-R*R=0
    // ���������巽�̿�������ʽ��t�Ľ�
    float discriminant = half_b * half_b - a * c;
    // �������޽�(������δ��������ʱ������δ����)
    if (discriminant > 0) {
      float delta = sqrt(discriminant);
      // +root
      float root = (-half_b - delta) / a;
      if (root < tmax && root > tmin) {
        record.t_           = root;
        record.p_           = ray.At(record.t_);
        Vec3 outward_normal = (record.p_ - center_) / radius_;
        record.SetFaceNormal(ray, outward_normal);
        record.mat_ptr_ = mat_ptr_;
        return true;
      }
      // -root
      root = (-half_b - delta) / a;
      if (root < tmax && root > tmin) {
        record.t_           = root;
        record.p_           = ray.At(record.t_);
        Vec3 outward_normal = (record.p_ - center_) / radius_;
        record.SetFaceNormal(ray, outward_normal);
        record.mat_ptr_ = mat_ptr_;
        return true;
      }
    }
    return false;
  }

 public:
  Vec3 center_;
  double radius_;
  Material* mat_ptr_;
};
#endif