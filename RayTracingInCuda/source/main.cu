#include <stdio.h>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <curand_kernel.h>

#include "hitable.cuh"
#include "hitable_list.cuh"
#include "sphere.cuh"
#include "cuda_helper.cuh"
#include "canvas.cuh"
#include "ray.cuh"
#include "camera.cuh"
#include "material.cuh"

using byte = char;

using Color = Vec3;

__device__ Color RayColor(const Ray& ray, const Hitable** world, curandState* rand_state)
{
  Ray cur_ray          = ray;
  Vec3 cur_attenuation = Vec3(1.0, 1.0, 1.0);
  for (int i = 0; i < 100; i++) {
    HitRecord rec;
    if ((*world)->Hit(cur_ray, 0.001f, FLT_MAX, rec)) {
      Ray scattered;
      Vec3 attenuation;
      if (rec.mat_ptr_->Scatter(cur_ray, rec, attenuation, scattered, rand_state)) {
        cur_attenuation *= attenuation;
        cur_ray = scattered;
      } else {
        return Vec3(0.0, 0.0, 0.0);
      }
    } else {
      Vec3 unit_direction = UnitVector(cur_ray.Direction());
      float t             = 0.5f * (unit_direction.y + 1.0f);
      Vec3 c              = (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
      return cur_attenuation * c;
    }
  }
  return Vec3(0.0, 0.0, 0.0);  // exceeded recursion
}

__global__ void CreateWorld(Hitable** d_list, Hitable** d_world, Camera** d_camera, int width, int height, curandState* rand_state)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    d_list[0] = new Sphere(Vec3(0, -1000.0, -1), 1000, new Lambertian(Vec3(0.5, 0.5, 0.5)));

    int i = 1;
    for (int a = -11; a < 11; a++) {
      for (int b = -11; b < 11; b++) {
        float choose_mat = curandom(rand_state);
        Vec3 center(a + curandom(rand_state), 0.2, b + curandom(rand_state));
        if (choose_mat < 0.8f) {
          d_list[i++] = new Sphere(
              center, 0.2,
              new Lambertian(Vec3(curandom(rand_state) * curandom(rand_state), curandom(rand_state) * curandom(rand_state), curandom(rand_state) * curandom(rand_state))));
        } else if (choose_mat < 0.95f) {
          d_list[i++] = new Sphere(
              center, 0.2,
              new Metal(Vec3(0.5f * (1.0f + curandom(rand_state)), 0.5f * (1.0f + curandom(rand_state)), 0.5f * (1.0f + curandom(rand_state))), 0.5f * curandom(rand_state)));
        } else {
          d_list[i++] = new Sphere(center, 0.2, new Dielectric(1.5));
        }
      }
    }
    d_list[i++] = new Sphere(Vec3(0, 1, 0), 1.0, new Dielectric(1.5));
    d_list[i++] = new Sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(Vec3(0.4, 0.2, 0.1)));
    d_list[i++] = new Sphere(Vec3(4, 1, 0), 1.0, new Metal(Vec3(0.7, 0.6, 0.5), 0.0));
    *d_world    = new HitableList(d_list, 22 * 22 + 1 + 3);

    Vec3 look_from(13, 2, 3);
    Vec3 look_at(0, 0, 0);
    Vec3 vup(0.0, 1.0, 0.0);

    float aspect_ratio  = width / height;
    float vfov          = 20;
    float dist_to_focus = 10;
    float aperture      = 0.1;
    *d_camera           = new Camera(look_from, look_at, vup, vfov, aspect_ratio, aperture, dist_to_focus);
  }
}

__global__ void FreeWorld(Hitable** d_list, Hitable** d_world, Camera** d_camera)
{
  for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
    delete ((Sphere*)d_list[i])->mat_ptr_;
    delete d_list[i];
  }
  delete *d_world;
  delete *d_camera;
}

__global__ void RandInit(curandState* rand_state)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init(1984, 0, 0, rand_state);
  }
}

__global__ void RenderInit(int width, int height, curandState* rand_state)
{
  const int x = blockIdx.x;
  const int y = threadIdx.x;
  if ((x >= width) || (y >= height)) return;
  int pixel_index = y * width + x;
  // Each thread gets same seed, a different sequence number, no offset
  curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void Render(Vec3* canvas_data, const int width, const int height, Hitable** d_world, Camera** d_camera, curandState* rand_state)
{
  const int x           = blockIdx.x;
  const int y           = threadIdx.x;
  const int pixel_index = y * width + x;

  curandState local_rand_state = rand_state[pixel_index];
  constexpr int samples_count  = 100;
  Color color                  = {0, 0, 0};
  for (int s = 0; s < samples_count; s++) {
    float u = float(x + curand_uniform(&local_rand_state)) / float(width);
    float v = float(height - y + curand_uniform(&local_rand_state)) / float(height);
    Ray ray = (*d_camera)->GetRay(u, v, rand_state);
    color += RayColor(ray, d_world, &local_rand_state);
  }

  canvas_data[y * width + x] = gamma_correct(anti_aliasing(color, samples_count));
}

int main()
{
  auto cuda_infos = SelectCudaInfos();

  if (cuda_infos.size() > 0) {
    PrintCudaDeviceInfo(cuda_infos[0]);
    cudaSetDevice(cuda_infos[0].mDeviceIndex);
  } else {
    return -1;
  }

  Canvas canvas(1600, 800);

  std::cerr << "\nRendering a " << canvas.width_ << "x" << canvas.height_ << " image.\n";

  const int blocks     = canvas.width_;
  const int threads    = canvas.height_;
  const int num_pixels = canvas.width_ * canvas.height_;

  Hitable** d_list;
  int num_hitables = 22 * 22 + 1 + 3;

  CheckCudaErrors(cudaMalloc((void**)&d_list, num_hitables * sizeof(Hitable*)));
  Hitable** d_world;
  CheckCudaErrors(cudaMalloc((void**)&d_world, sizeof(Hitable*)));
  Camera** d_camera;
  CheckCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));

  curandState* d_tmp_rand;
  CheckCudaErrors(cudaMalloc((void**)&d_tmp_rand, 1 * sizeof(curandState)));
  RandInit<<<1, 1>>>(d_tmp_rand);
  CheckCudaErrors(cudaGetLastError());
  CheckCudaErrors(cudaDeviceSynchronize());

  CreateWorld<<<1, 1>>>(d_list, d_world, d_camera, canvas.width_, canvas.height_, d_tmp_rand);
  CheckCudaErrors(cudaGetLastError());
  CheckCudaErrors(cudaDeviceSynchronize());

  clock_t start, stop;
  start = clock();

  curandState* d_rand_state;
  CheckCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
  RenderInit<<<blocks, threads>>>(canvas.width_, canvas.height_, d_rand_state);
  CheckCudaErrors(cudaGetLastError());
  CheckCudaErrors(cudaDeviceSynchronize());

  Render<<<blocks, threads>>>(canvas.d_data_, canvas.width_, canvas.height_, d_world, d_camera, d_rand_state);
  CheckCudaErrors(cudaGetLastError());
  CheckCudaErrors(cudaDeviceSynchronize());

  stop = clock();

  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  std::ofstream ppm_file("image.ppm", std::ios::binary);
  ppm_file << "P6\n" << canvas.width_ << " " << canvas.height_ << "\n255\n";

  std::vector<Vec3> data_host(canvas.width_ * canvas.height_);
  cudaMemcpy(data_host.data(), canvas.d_data_, canvas.width_ * canvas.height_ * sizeof(Vec3), cudaMemcpyDeviceToHost);

  const int COLOR_TO_BYTE_SIZE = 3;

  byte* data = new byte[canvas.width_ * canvas.height_ * COLOR_TO_BYTE_SIZE];

  // row scanning.
  for (int y = 0; y < canvas.height_; y++) {
    for (int x = 0; x < canvas.width_; x++) {
      data[(y * canvas.width_ + x) * COLOR_TO_BYTE_SIZE]     = data_host[y * canvas.width_ + x].x * 255.99f;
      data[(y * canvas.width_ + x) * COLOR_TO_BYTE_SIZE + 1] = data_host[y * canvas.width_ + x].y * 255.99f;
      data[(y * canvas.width_ + x) * COLOR_TO_BYTE_SIZE + 2] = data_host[y * canvas.width_ + x].z * 255.99f;
    }
  }
  ppm_file.write(data, canvas.width_ * canvas.height_ * COLOR_TO_BYTE_SIZE);
  ppm_file.flush();
  ppm_file.close();

  int w, h, channel;
  unsigned char* datas = stbi_load("image.ppm", &w, &h, &channel, 0);

  std::string output_path = "image.png";
  stbi_write_png(output_path.c_str(), w, h, channel, datas, 0);

  std::string show_output = "mspaint " + output_path;

  system(show_output.c_str());

  // clean up
  CheckCudaErrors(cudaDeviceSynchronize());
  FreeWorld<<<1, 1>>>(d_list, d_world, d_camera);
  CheckCudaErrors(cudaGetLastError());
  CheckCudaErrors(cudaFree(d_camera));
  CheckCudaErrors(cudaFree(d_world));
  CheckCudaErrors(cudaFree(d_list));
  CheckCudaErrors(cudaFree(d_rand_state));
  CheckCudaErrors(cudaFree(d_tmp_rand));

  cudaDeviceReset();
}
