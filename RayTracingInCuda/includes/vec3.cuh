#ifndef VEC3_CUH
#define VEC3_CUH

#include "cuda_helper.cuh"
#include "rtweekend.cuh"
#include <stdio.h>

class Vec3
{
 public:
  __host__ __device__ Vec3() : x(0), y(0), z(0) {}
  __host__ __device__ Vec3(float xcomp, float ycomp, float zcomp)
  {
    this->x = xcomp;
    this->y = ycomp;
    this->z = zcomp;
  }

  __host__ __device__ inline const Vec3 operator+() const { return *this; }
  __host__ __device__ inline Vec3 operator-() const
  {
    Vec3 tmp = {-x, -y, -z};
    return tmp;
  }

  __host__ __device__ inline Vec3& operator=(const Vec3& vec);

  __host__ __device__ inline Vec3& operator+=(const Vec3& vec);
  __host__ __device__ inline Vec3& operator-=(const Vec3& vec);
  __host__ __device__ inline Vec3& operator*=(const Vec3& vec);
  __host__ __device__ inline Vec3& operator/=(const Vec3& vec);
  __host__ __device__ inline Vec3& operator*=(const float scale);
  __host__ __device__ inline Vec3& operator/=(const float scale);

  __host__ __device__ inline float Length() const { return sqrt(x * x + y * y + z * z); }
  __host__ __device__ inline float SquaredLength() const { return x * x + y * y + z * z; }
  __host__ __device__ inline void MakeUnitVector();

 public:
  __device__ inline static Vec3 Random(curandState* rand_state) { return Vec3(curandom(rand_state), curandom(rand_state), curandom(rand_state)); }

  __device__ inline static Vec3 Random(double min, double max, curandState* rand_state)
  {
    return Vec3(curandom(min, max, rand_state), curandom(min, max, rand_state), curandom(min, max, rand_state));
  }

 public:
  union {
    struct {
      float x, y, z;
    };
    struct {
      float r, g, b;
    };
  };
};

__host__ __device__ inline Vec3& Vec3::operator=(const Vec3& vec)
{
  this->x = vec.x;
  this->y = vec.y;
  this->z = vec.z;
  return *this;
}

__host__ __device__ inline void Vec3::MakeUnitVector()
{
  float k = 1.0f / Length();
  x *= k;
  y *= k;
  z *= k;
}

__host__ __device__ inline Vec3& Vec3::operator+=(const Vec3& vec)
{
  this->x += vec.x;
  this->y += vec.y;
  this->z += vec.z;
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator-=(const Vec3& vec)
{
  this->x -= vec.x;
  this->y -= vec.y;
  this->z -= vec.z;
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const Vec3& vec)
{
  this->x *= vec.x;
  this->y *= vec.y;
  this->z *= vec.z;
  return *this;
}

inline Vec3& Vec3::operator/=(const Vec3& vec)
{
  this->x /= vec.x;
  this->y /= vec.y;
  this->z /= vec.z;
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const float scale)
{
  this->x *= scale;
  this->y *= scale;
  this->z *= scale;
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const float scale)
{
  this->x /= scale;
  this->y /= scale;
  this->z /= scale;
  return *this;
}

__host__ __device__ inline std::istream& operator>>(std::istream& is, Vec3& vec)
{
  is >> vec.x >> vec.y >> vec.z;
  return is;
}

__host__ __device__ inline std::ostream& operator<<(std::ostream& os, const Vec3& vec)
{
  os << vec.x << " "
     << " " << vec.y << " " << vec.z;
  return os;
}

__host__ __device__ inline Vec3 operator+(const Vec3& lhs, const Vec3& rhs) { return Vec3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z); }

__host__ __device__ inline Vec3 operator-(const Vec3& lhs, const Vec3& rhs) { return Vec3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z); }

__host__ __device__ inline Vec3 operator*(const Vec3& lhs, const Vec3& rhs) { return Vec3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z); }

__host__ __device__ inline Vec3 operator/(const Vec3& lhs, const Vec3& rhs) { return Vec3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z); }

__host__ __device__ inline Vec3 operator*(float scale, const Vec3& vec) { return Vec3(scale * vec.x, scale * vec.y, scale * vec.z); }

__host__ __device__ inline Vec3 operator*(const Vec3& vec, float scale) { return Vec3(scale * vec.x, scale * vec.y, scale * vec.z); }

__host__ __device__ inline Vec3 operator/(const Vec3& vec, float scale) { return Vec3(vec.x / scale, vec.y / scale, vec.z / scale); }

__host__ __device__ inline float dot(const Vec3& lhs, const Vec3& rhs)
{
  float result = lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
  return result;
}

__host__ __device__ inline Vec3 cross(const Vec3& lhs, const Vec3& rhs)
{
  return Vec3((lhs.y * rhs.z - lhs.z * rhs.y), (-(lhs.x * rhs.z - lhs.z * rhs.x)), (lhs.x * rhs.y - lhs.y * rhs.x));
}

__host__ __device__ inline Vec3 UnitVector(Vec3 vec) { return vec / vec.Length(); }

__device__ inline Vec3 random_in_unit_sphere(curandState* rand_state)
{
  while (true) {
    Vec3 p = Vec3::Random(-1, 1, rand_state);
    if (p.SquaredLength() >= 1) continue;
    return p;
  }
}

__device__ inline Vec3 random_unit_vector(curandState* rand_state)
{
  auto a = curandom(0, 2 * pi, rand_state);
  auto z = curandom(-1, 1, rand_state);
  auto r = sqrt(1 - z * z);
  return Vec3(r * cos(a), r * sin(a), z);
}

__device__ inline Vec3 random_in_unit_disk(curandState* rand_state)
{
  while (true) {
    Vec3 p = Vec3(curandom(-1, 1, rand_state), curandom(-1, 1, rand_state), 0);
    if (p.SquaredLength() >= 1) continue;
    return p;
  }
}

__host__ __device__ inline Vec3 gamma_correct(const Vec3& color)
{
  auto r = sqrt(color.x);
  auto g = sqrt(color.y);
  auto b = sqrt(color.z);

  return Vec3(r, g, b);
}

__host__ __device__ inline Vec3 anti_aliasing(Vec3 color, int samples_count)
{
  auto scale = 1.0f / samples_count;

  color = color * scale;

  auto r = clamp(color.x, 0.0f, 0.999f);
  auto g = clamp(color.y, 0.0f, 0.999f);
  auto b = clamp(color.z, 0.0f, 0.999f);

  return Vec3(r, g, b);
}

__host__ __device__ inline Vec3 reflect(const Vec3& v, const Vec3& n) { return v - 2 * dot(v, n) * n; }

__host__ __device__ inline Vec3 refract(const Vec3& uv, const Vec3& n, double refraction)
{
  float tmp_cos_theta = dot(-uv, n);
  float cos_theta     = tmp_cos_theta < 1.0f ? tmp_cos_theta : 1.0f;
  Vec3 r_out_perp     = refraction * (uv + cos_theta * n);
  Vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.SquaredLength())) * n;
  return r_out_perp + r_out_parallel;
}

#endif  // VEC3_CUH