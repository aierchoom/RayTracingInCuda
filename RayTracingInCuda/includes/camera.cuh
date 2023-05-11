#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "ray.cuh"

class Camera
{
 public:
  __device__ Camera(Vec3 look_from, Vec3 look_at, Vec3 vup, double vfov, double aspect, double aperture, double focus_dist)
  {
    origin_      = look_from;
    lens_radius_ = aperture / 2;

    float theta       = degrees_to_radians(vfov);
    float half_height = tan(theta / 2);
    float half_width  = aspect * half_height;

    w_ = UnitVector(look_from - look_at);
    u_ = UnitVector(cross(vup, w_));
    v_ = cross(w_, u_);

    lower_left_corner_ = origin_ - half_width * focus_dist * u_ - half_height * focus_dist * v_ - focus_dist * w_;
    horizontal_        = 2 * half_width * focus_dist * u_;
    vertical_          = 2 * half_height * focus_dist * v_;
  }

  __device__ Ray GetRay(float s, float t, curandState *rand_state)
  {
    Vec3 radius = lens_radius_ * random_in_unit_disk(rand_state);
    Vec3 offset = u_ * radius.x + v_ * radius.y;

    return Ray(origin_ + offset, lower_left_corner_ + s * horizontal_ + t * vertical_ - origin_ - offset);
  }

  Vec3 origin_;
  Vec3 lower_left_corner_;
  Vec3 horizontal_;
  Vec3 vertical_;

  Vec3 u_;
  Vec3 v_;
  Vec3 w_;

  double lens_radius_;
};

#endif