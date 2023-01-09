

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdio.h"
#include <iostream>

#define gpuErrChk(res) {gpuAssert((res), __FILE__, __LINE__);}

inline void gpuAssert(cudaError_t ret, const char* file, int line, bool abort=true) {
  // check if ret was a success
  if (ret != cudaSuccess) {
    std::cout << "gpu assert " << cudaGetErrorString(ret) << file << " " << line << " \n";
    if (abort) exit(ret);
  }
}
