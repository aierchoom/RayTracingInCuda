#ifndef HITABLE_LIST_CUH
#define HITABLE_LIST_CUH

#include "hitable.cuh"

class HitableList : public Hitable
{
 public:
  __device__ HitableList() {}
  __device__ HitableList(Hitable** list, int num)
  {
    list_      = list;
    list_size_ = num;
  }

  __device__ virtual bool Hit(const Ray& ray, double tmin, double tmax, HitRecord& record) const
  {
    HitRecord temp_record;
    bool hit_anything     = false;
    double closest_so_far = tmax;

    for (int i = 0; i < list_size_; i++) {
      if (list_[i]->Hit(ray, tmin, closest_so_far, temp_record)) {
        hit_anything   = true;
        closest_so_far = temp_record.t_;
        record         = temp_record;
      }
    }
    return hit_anything;
  }

 public:
  Hitable** list_;
  int list_size_;
};

#endif