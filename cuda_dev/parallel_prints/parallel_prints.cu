#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


#include "settings.h"

__global__ void print_from_cuda() {
  printf("This is the first line \n");
}


__global__ void print_thread_ids() {
  printf("thread x:%d y:%d z:%d\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

__global__ void print_thread_info() {
  printf("block x:%d/%d y:%d/%d z:%d/%d thread x:%d/%d y:%d/%d z:%d/%d\n", 
  blockIdx.x,  gridDim.x, 
  blockIdx.y,  gridDim.y, 
  blockIdx.z,  gridDim.z,
  threadIdx.x, blockDim.x, 
  threadIdx.y, blockDim.y, 
  threadIdx.z, blockDim.z
  );
}

__global__ void print_array_elements_1D(int *data) {
  // assume 1 block with N threads
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < blockDim.x * gridDim.x) {
    printf("array has value %d at index %d\n", data[index], index);
  }
}

__global__ void print_array_elements_2D(int *data) {
  // assume 1 block with N threads
  int index = (
    threadIdx.x + blockIdx.x * blockDim.x +
    threadIdx.y * blockDim.x * gridDim.x 
  );
  if (index < blockDim.x * blockDim.y * gridDim.x) {
    printf(
      "array has value %d at index %d for thread x %d, y %d, block %d\n", 
      data[index], index, threadIdx.x, threadIdx.y, blockIdx.x
    );
  }
}

__global__ void print_array_elements(int *data) {
  // assumes a 2D grid
  int index = (
    threadIdx.x + threadIdx.y * blockDim.x + // index of an element inside a block
    blockIdx.x * blockDim.x * blockDim.y +   // column offset of the index of a block
    blockIdx.y * gridDim.x * blockDim.x * blockDim.y // row offset of the index
  );

  printf(
    "array has value %d at index %d for thread x %d, y %d, block x %d y %d\n", 
    data[index], index, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y 
  );  
}

int main() {

  #if PRINTFROMCUDA

  int n_block_x{NBX}, n_block_y{NBY}, n_block_z{1};
  int n_tpb_x{NX/NBX}, n_tpb_y{NUMTHREADS/NX/NBY}, n_tpb_z{1}; // tpb threads per block 

  dim3 grid(n_block_x, n_block_y, n_block_z);
  dim3 block(n_tpb_x, n_tpb_y, n_tpb_z);
  // print_from_cuda <<<grid, block>>>();
  print_thread_info <<<grid, block>>>();
  #endif

  #if PRINT1BLOCKARRAY

  int num_elements = NUMTHREADS;
  // create the array and initialize it.
  int host_data[num_elements];
  for (int i = 0; i < num_elements; i++) {
    host_data[i] = i;
  }

  // create the array in cuda
  int * device_data;
  int host_array_size = sizeof(int) * num_elements;

  // allocate memory in cuda
  cudaMalloc((void**) &device_data, host_array_size);

  // copy the array to the allocated memory in the cuda
  cudaMemcpy(device_data, host_data, host_array_size, cudaMemcpyHostToDevice);

  #if PRINT1DARRAYIN1D
  // define the grid size
  dim3 grid(NBX);
  dim3 block(NUMTHREADS/NBX);

  print_array_elements_1D<<<grid, block>>>(device_data);
  #endif

  #if PRINT1DARRAYIN2D
  // define the grid size
  dim3 grid(NBX);
  dim3 block(NX/NBX, NUMTHREADS/NX);

  print_array_elements_2D<<<grid, block>>>(device_data);
  #endif

  #if PRINT2DARRAY2D
  dim3 grid(NBX, NBY);
  dim3 block(NX/NBX, NUMTHREADS/NX/NBY);

  print_array_elements<<<grid, block>>>(device_data);
  #endif

  #endif

  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;

}