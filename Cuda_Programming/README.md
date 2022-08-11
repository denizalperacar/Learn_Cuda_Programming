This repository contains my notes from the book learn cuda programming

# This book is abandaned due to many errors present in it.

# Syntax Cheat
**Kernel call**: `MyKernel<<< blocks, threads >>> (...);`

# Chapter 1 Notes

In CUDA, there are two processors that work with each other. The `host` is usually referred to as the CPU, while the `device` is usually referred to as the GPU.

* __global__: This keyword, when added before the function, tells the compiler that this is a function that will run on the device and not on the host. However, note that it is called by the host. Another important thing to note here is that the return type of the device function is always "void". Data-parallel portions of an algorithm are executed on the device as kernels.
* <<<,>>>: This keyword tells the compiler that this is a call to the device function and not the host function. Additionally, the 1,1 parameter basically dictates the number of threads to launch in the kernel.
* threadIdx.x, blockIdx.x: This is a unique ID that's given to all threads.
* cudaDeviceSynchronize(): All of the kernel calls in CUDA are asynchronous in nature. The host becomes free after calling the kernel and starts executing the next instruction afterward. This should come as no big surprise since this is a heterogeneous environment and hence both the host and device can run in parallel to make use of the types of processors that are available. In case the host needs to wait for the device to finish, APIs have been provided as part of CUDA programming that make the host code wait for the device function to finish. One such API is cudaDeviceSynchronize, which waits until all of the previous calls to the device have finished.

## GPU Architecture


* CUDA thread [Executes as] CUDA Core/SIMD code 
* CUDA block [Executes on] Streaming multiprocessor 
* GRID/kernel [Executes on] GPU device

---

* CUDA Threads: CUDA threads execute on a CUDA core. CUDA threads are different from CPU threads. CUDA threads are extremely lightweight and provide fast context switching. The reason for fast context switching is due to the availability of a large register size in a GPU and hardware-based scheduler. The thread context is present in registers compared to CPU, where the thread handle resides in a lower memory hierarchy such as a cache. Hence, when one thread is idle/waiting, another thread that is ready can start executing with almost no delay. _Each CUDA thread must execute the same kernel and work independently on different data (SIMT)_.

* CUDA blocks: CUDA threads are grouped together into a logical entity called a CUDA block. CUDA blocks execute on a single Streaming Multiprocessor (SM). _One block runs on a single SM, that is, all of the threads within one block can only execute on cores in one SM and do not execute on the cores of other SMs_. Each GPU may have one or more SM and hence to effectively make use of the whole GPU; the user needs to divide the parallel computation into blocks and threads. 

* GRID/kernel: CUDA blocks are grouped together into a logical entity called a CUDA GRID. A CUDA GRID is then executed on the device.

## Vector Addition in Cuda

* **Memory allocation on GPU**: CPU memory and GPU memory are physically1. separate memory. malloc allocates memory on the CPU's RAM. The GPU kernel/device function can only access memory that's allocated/pointing to the device memory. To allocate memory on the GPU, _we need to use the cudaMalloc API_. _Unlike the malloc command, cudaMalloc does not return a pointer to allocated memory; instead, it takes a pointer reference as a parameter and updates the same with the allocated memory_. 
This syntax shown in the book doesn not work: `cudaMalloc((void *)&d_a, N * sizeof(int));` remove the temporary void * cast then it works. `cudaMalloc(&d_a, N * sizeof(int));`

* **Transfer data from host memory to device memory**: The host data is then copied to the device's memory, which was allocated using the cudaMalloc command used in the previous step. The API that's used to copy the data between the host and device and vice versa is cudaMemcpy. Like other memcopy commands, this API requires the destination pointer, source pointer, and size. One additional parameter it takes is the direction of copy, that is, whether we are copying from the host to the device or from the device to the host. In the latest version of CUDA, this is optional since the driver is capable of understanding whether the pointer points to the host memory or device memory. Note that there is an asynchronous alternative to cudaMemcpy. This will be covered in more detail in other chapters.
`cudaMemcpy(cpyto, cpyfrom, sizeofarray, cudaMemcpyHostToDevice);`

* **<<<block size, thread size>>>**

* **cudaDeviceSynchronize**: wait for the kernel execution to finish.

* **transfer data from device to host**: Use the same `cudaMemcpy` API to copy the data back from the device to the host for post-processing or validation duties such as printing. The only change here, compared to the first step, is that _we reverse the direction of the copy, that is, the destination pointer points to the host while the source pointer points to the device allocated in memory_.
`cudaMemcpy(cpyto, cpyfrom, sizeofarray, cudaMemcpyDeviceToHost);`

* **Free the allocated GPU memory**: use cudaFree

### multiple blocks

device_add<<<N, 1>>> : this will execute the device_add function N times in parallel instead of once. *Each parallel invocation of the device_add function is referred to as a block.* By using `blockIdx.x` to index the array, each block handles a different element of the array.

### multiple threads 

device_add<<<1,N>>> : This will execute the device_add function N times in parallel instead of once. *Each parallel invocation of the device_add function is referred to as a thread.* By using `threadIdx.x` to index the array, each block handles a different element of the array.

### Combination of block and thread

`blockDim.x` : The number of threads in each block

```C
__global__ void gpu_add(int *a, int* b, int* c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
}
```

## Notes on threads and blocks

* CUDA programming model allows communication for threads within the same block.
* Threads belonging to different blocks cannot communicate/synchronize with each other during the execution of the kernel.

* **Blocks are scheduled independently -> use for embarassingly parallel tasks.** also more hardware blocks more computations at the same time.
_For instance for matrix vector multiplications one can send sets of rows to different blocks to be multiplied in parallel._

* threads communicate via `shared memory`.

## Error Reporting in Cuda
Most CUDA functions call `cudaError_t`, which is basically an enumeration type. cudaSuccess (value 0) indicates a 0 error.

```C
cudaError_t = error;
error = cudaMemcpy(...)
if (error) printf("%s", cudaGetErrorString(err));
```

No return value for kernel launches, so for such cases use cudaGetLastError();
cudaGetLastError() : cuda error for the last launched function including the kernel.

## Data types in cuda

* supports all the standard datatypes
* make sure to align the data call of the same type then cuda accesses the same memory.
* CUDA programming supports complex data structures such as structures and classes.
* one can enforce alignment for class types via `__align__`

```C
struct __align__(16) {
    float r, g, b;
}
```

# Chapter 2

## Memory Profiler

call 'nvvp' 

nvcc -o vector_addition vector_addition.cu
nvprof -o vector_addition.nvvp ./vector_addition

## Global / device memory

1. what is global memory

* This mem is visible to all threads in the kernel
* This mem is visible to CPU
* use `cudaMalloc` and `cudaFree` explicitly to manage the memory
* data is declared as `__device__`
* data can be transfered by `cudaMemcpy()`
* After memory allocation in device *pointer of those vaibles will point to the gloabal memory*

2. How to load aor store data from global mem to cache

3. How to access mem optimally

* **Concept of warp** : The warp is a unit of thread scheduling/execution in SMs. Once a block has been assigned to an SM, it is divided into a
32 -thread unit known as a warp 
* All of the threads in a warp fetch and execute the same instruction when selected
* To optimally utilize access from global memory, the access should coalesce 
> * Coalesced global memory access: sequential memory access is adjacent
> * Uncoalesced global memory access: sequential memory access is not adjacent

4. How data reaches to warf form global memory via cache lines:

Coalesced global memory access: sequential memory access is adjacent

## Shared Memory

* This provides a mechanism for users so that they can read/write data in a coalesced fashion from global memory and store it in memory, which acts like a cache but can be controlled by the user.

* It is only visible to the threads in the same block. 
* All of the threads in a block see the same version of a shared variable.
* CUDA programmers can use shared variables to hold the data that was reused many times during the execution phase of the kernel.

### The example code for the matrix transpose using the shared memory is wrong 
It allocates a shared memory of size [BlockSize] [BlockSize] and then requires the block to __syncthreads() but the number of threads is higher 

