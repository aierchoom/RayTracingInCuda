#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH
#include <iostream>
#include <vector>

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define CheckCudaErrors(val) CheckCuda((val), #val, __FILE__, __LINE__)

inline void CheckCuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
    cudaDeviceReset();
    exit(99);
  }
}

struct CudaDeviceInfo {
  struct Version {
    int major;
    int minor;
  };

  std::string mDeviceName;
  int mDeviceIndex;
  Version mDirverVersion;
  Version mRuntimeVersion;
  Version mCapabilityVersion;

  float mGlobalMemoryMB;
  float mGPUClockRateMB;
  int mMemoryBusWidthBits;
  int mMemoryL2CacheSizeBytes;

  int mConstantMemoryBytes;
  int mSharedMemoryPerBlockBytes;
  int mRegistersAvailablePerBlockCount;

  int mWarpSize;
  int mMaxThreadCountPerMultiProcesser;
  int mMaxTheadCountPerBlock;
  size_t mMaxMemoryPitchBytes;

  struct {
    int x;
  } mMaxTextureSize1D;

  struct {
    int x, y;
  } mMaxTextureSize2D;

  struct {
    int x, y, z;
  } mMaxTextureSize3D;

  struct {
    int dimx;
    int layers;
  } mMaxLayerTexSize1D;

  struct {
    int dimx, dimy;
    int layers;
  } mMaxLayerTexSize2D;

  struct {
    int x, y, z;
  } mMaxThreadDim;

  struct {
    int x, y, z;
  } mMaxGridSize;
};

inline std::vector<CudaDeviceInfo> SelectCudaInfos()
{
  int device_count = 0;
  CheckCudaErrors(cudaGetDeviceCount(&device_count));

  std::vector<CudaDeviceInfo> cuda_device_infos;

  if (device_count == 0) {
    //  There are no available device(s) that support CUDA
    return cuda_device_infos;
  }
  cuda_device_infos.resize(device_count);

  for (int i = 0; i < cuda_device_infos.size(); i++) {
    cudaDeviceProp device_prop;
    cuda_device_infos[i].mDeviceIndex = i;
    cudaGetDeviceProperties(&device_prop, cuda_device_infos[i].mDeviceIndex);
    cuda_device_infos[i].mDeviceName              = device_prop.name;
    cuda_device_infos[i].mCapabilityVersion.major = device_prop.major;
    cuda_device_infos[i].mCapabilityVersion.minor = device_prop.minor;

    int driver_version, runtime_version;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);
    cuda_device_infos[i].mDirverVersion  = {driver_version / 1000, (driver_version % 100) / 10};
    cuda_device_infos[i].mRuntimeVersion = {(runtime_version / 1000), ((runtime_version % 100)) / 10};
    cuda_device_infos[i].mGPUClockRateMB = static_cast<float>(device_prop.clockRate * 1e-3f);

    cuda_device_infos[i].mMaxMemoryPitchBytes             = device_prop.memPitch;
    cuda_device_infos[i].mGlobalMemoryMB                  = static_cast<float>(device_prop.totalGlobalMem) / static_cast<float>(pow(1024.0, 3));
    cuda_device_infos[i].mConstantMemoryBytes             = static_cast<int>(device_prop.totalConstMem);
    cuda_device_infos[i].mSharedMemoryPerBlockBytes       = static_cast<int>(device_prop.sharedMemPerBlock);
    cuda_device_infos[i].mRegistersAvailablePerBlockCount = device_prop.regsPerBlock;
    cuda_device_infos[i].mWarpSize                        = device_prop.warpSize;
    cuda_device_infos[i].mMaxThreadCountPerMultiProcesser = device_prop.maxThreadsPerMultiProcessor;
    cuda_device_infos[i].mMaxTheadCountPerBlock           = device_prop.maxThreadsPerBlock;

    cuda_device_infos[i].mMaxTextureSize1D.x = device_prop.maxTexture1D;

    cuda_device_infos[i].mMaxTextureSize2D.x = device_prop.maxTexture2D[0];
    cuda_device_infos[i].mMaxTextureSize2D.y = device_prop.maxTexture2D[1];

    cuda_device_infos[i].mMaxTextureSize3D.x = device_prop.maxTexture3D[0];
    cuda_device_infos[i].mMaxTextureSize3D.y = device_prop.maxTexture3D[1];
    cuda_device_infos[i].mMaxTextureSize3D.z = device_prop.maxTexture3D[2];

    cuda_device_infos[i].mMemoryBusWidthBits     = device_prop.memoryBusWidth;
    cuda_device_infos[i].mMemoryL2CacheSizeBytes = device_prop.l2CacheSize;

    cuda_device_infos[i].mMaxLayerTexSize1D.dimx = device_prop.maxTexture1DLayered[0];

    cuda_device_infos[i].mMaxLayerTexSize2D.dimx   = device_prop.maxTexture2DLayered[0];
    cuda_device_infos[i].mMaxLayerTexSize2D.dimy   = device_prop.maxTexture2DLayered[1];
    cuda_device_infos[i].mMaxLayerTexSize2D.layers = device_prop.maxTexture2DLayered[2];

    cuda_device_infos[i].mMaxThreadDim.x = device_prop.maxThreadsDim[0];
    cuda_device_infos[i].mMaxThreadDim.y = device_prop.maxThreadsDim[1];
    cuda_device_infos[i].mMaxThreadDim.z = device_prop.maxThreadsDim[2];

    cuda_device_infos[i].mMaxGridSize.x = device_prop.maxGridSize[0];
    cuda_device_infos[i].mMaxGridSize.y = device_prop.maxGridSize[1];
    cuda_device_infos[i].mMaxGridSize.z = device_prop.maxGridSize[2];
  }
  return cuda_device_infos;
}

inline void PrintCudaDeviceInfo(const CudaDeviceInfo &device_info)
{
  printf("Device %d:\"%s\"\n", device_info.mDeviceIndex, device_info.mDeviceName.c_str());

  printf("  CUDA Driver Version / Runtime Version         %d.%d  /  %d.%d\n", device_info.mDirverVersion.major, device_info.mDirverVersion.major, device_info.mRuntimeVersion.major,
         device_info.mRuntimeVersion.minor);
  printf("  CUDA Capability Major/Minor version number:   %d.%d\n", device_info.mCapabilityVersion.major, device_info.mCapabilityVersion.minor);
  int global_memory_to_bytes = device_info.mGlobalMemoryMB * 1024 * 1024;
  printf("  Total amount of global memory:                %.2f MBytes (%llu bytes)\n", device_info.mGlobalMemoryMB, global_memory_to_bytes);
  printf("  GPU Clock rate:                               %.0f MHz (%0.2f GHz)\n", device_info.mGPUClockRateMB, device_info.mGPUClockRateMB * 1e-3f);
  printf("  Memory Bus width:                             %d-bits\n", device_info.mMemoryBusWidthBits);
  printf("  L2 Cache Size:                            	%d bytes\n", device_info.mMemoryL2CacheSizeBytes);
  printf("  Max Texture Dimension Size (x,y,z)            1D=(%d),2D=(%d,%d),3D=(%d,%d,%d)\n", device_info.mMaxTextureSize1D.x, device_info.mMaxTextureSize2D.x,
         device_info.mMaxTextureSize2D.y, device_info.mMaxTextureSize3D.x, device_info.mMaxTextureSize3D.y, device_info.mMaxTextureSize3D.z);
  printf("  Max Layered Texture Size (dim) x layers       1D=(%d) x %d,2D=(%d,%d) x %d\n", device_info.mMaxLayerTexSize1D.dimx, device_info.mMaxLayerTexSize1D.layers,
         device_info.mMaxLayerTexSize2D.dimx, device_info.mMaxLayerTexSize2D.dimy, device_info.mMaxLayerTexSize1D.layers);
  printf("  Total amount of constant memory               %lu bytes\n", device_info.mConstantMemoryBytes);
  printf("  Total amount of shared memory per block:      %lu bytes\n", device_info.mSharedMemoryPerBlockBytes);
  printf("  Total number of registers available per block:%d\n", device_info.mRegistersAvailablePerBlockCount);
  printf("  Wrap size:                                    %d\n", device_info.mWarpSize);
  printf("  Maximun number of thread per multiprocesser:  %d\n", device_info.mMaxThreadCountPerMultiProcesser);
  printf("  Maximun number of thread per block:           %d\n", device_info.mMaxTheadCountPerBlock);
  printf("  Maximun size of each dimension of a block:    %d x %d x %d\n", device_info.mMaxThreadDim.x, device_info.mMaxThreadDim.y, device_info.mMaxThreadDim.z);
  printf("  Maximun size of each dimension of a grid:     %d x %d x %d\n", device_info.mMaxGridSize.x, device_info.mMaxGridSize.y, device_info.mMaxGridSize.z);
  printf("  Maximu memory pitch                           %lu bytes\n", device_info.mMaxMemoryPitchBytes);
}
#endif  // CUDA_HELPER_CUH