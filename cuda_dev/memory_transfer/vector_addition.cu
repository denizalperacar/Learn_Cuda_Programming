#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include "chrono"

#include "../utils/error_handling.cu"

__global__ void vector_sum(int * a, int * b, int * c, int size) {

  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < size) {
    c[gid] = a[gid] + b[gid];
  }
}


int main() {

  int num_ele = 2000;
  std::vector<int> h_a(num_ele);
  std::vector<int> h_b(num_ele);
  int * h_c = new int[num_ele];

  for (int i=0; i < static_cast<int>(h_a.size()); i++) {
    h_a[i] = i * 2;
    h_b[i] = i * 3;
  }

  // allocate the cude arrays
  int array_byte_size = sizeof(int) * num_ele;
  int *d_a, *d_b, *d_c;
  gpuErrChk(cudaMalloc((void**) &d_a, array_byte_size));
  gpuErrChk(cudaMalloc((void**) &d_b, array_byte_size));
  gpuErrChk(cudaMalloc((void**) &d_c, array_byte_size));
  
  // transfer data from host to device
  gpuErrChk(cudaMemcpy(d_a, h_a.data(), array_byte_size, cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(d_b, h_b.data(), array_byte_size, cudaMemcpyHostToDevice));

  // execute the kernel
  dim3 block(128);
  dim3 grid((num_ele / block.x) + 1);
  
  auto start = std::chrono::high_resolution_clock::now();
  vector_sum <<<grid, block>>>(d_a, d_b, d_c, h_a.size());
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "execution time : " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " \xC2\xB5s"<< std::endl;

  // synchronize (might not be needed? I havnt added it in the initial implementation)
  gpuErrChk(cudaDeviceSynchronize());

  // transfer data to h_c 
  gpuErrChk(cudaMemcpy(h_c, d_c, array_byte_size, cudaMemcpyDeviceToHost));
  
  //for (int i = 0; i < h_a.size(); i++) {
  //  std::cout << h_c[i] << " ";
  //}

  std::cout << std::endl;

  // free the allocated arrays
  gpuErrChk(cudaFree(d_a)); gpuErrChk(cudaFree(d_b)); gpuErrChk(cudaFree(d_c));
  delete [] h_c; h_c = nullptr;  

  return 0;
}
