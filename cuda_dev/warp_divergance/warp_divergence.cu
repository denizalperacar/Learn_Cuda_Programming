
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void no_warp_divergance() {

  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  int warpid = gid / 32;

  int a, b;

  if (warpid % 2 == 0) {
    a = 1;
    b = 2;
  }
  else {
    a = 2;
    b = 1;
  }

}

__global__ void with_warp_divergance() {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;

  int a, b;

  if (gid % 2 == 0) {
    a = 1;
    b = 2;
  }
  else {
    a = 2;
    b = 1;
  }
}


int main() {

  dim3 block(110);
  dim3 grid(8);

  no_warp_divergance<<<grid, block>>>();
  cudaDeviceSynchronize();
  with_warp_divergance<<<grid, block>>>();
  cudaDeviceSynchronize();
  
  return 0;
}
