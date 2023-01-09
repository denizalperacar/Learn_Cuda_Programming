

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdio.h"

__global__ void print_array(int * arr, int N) {

  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < N) {
    printf("thread %d, block %d, index %d, value %d\n", 
    threadIdx.x, blockIdx.x, gid, arr[gid]
  );
  }
}

int main() {

  int num_ele = 130;
  unsigned arr_size = sizeof(int) * num_ele;
  int * h_data = new int[num_ele];
  
  for (int i = 0; i < num_ele; i++) {
    h_data[i] = i * 2;
  }

  // send the data to device
  int * d_data;
  cudaMalloc((void**) &d_data, arr_size);
  cudaMemcpy(d_data, h_data, arr_size, cudaMemcpyHostToDevice);

  // run the kernel
  dim3 grid(32);
  dim3 block(5);

  // num_ele is not explicitly sent to the device
  print_array<<<grid, block>>>(d_data, num_ele); 

  // wait for the device execution
  cudaDeviceSynchronize();
  
  // free the memory
  cudaFree(d_data);
  delete [] h_data;
  
  return 0;
}


