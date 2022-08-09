This repository contains my notes from the book learn cuda programming

# Chapter 1 Notes

In CUDA, there are two processors that work with each other. The `host` is usually referred to as the CPU, while the `device` is usually referred to as the GPU.

* __global__: This keyword, when added before the function, tells the compiler that this is a function that will run on the device and not on the host. However, note that it is called by the host. Another important thing to note here is that the return type of the device function is always "void". Data-parallel portions of an algorithm are executed on the device as kernels.
* <<<,>>>: This keyword tells the compiler that this is a call to the device function and not the host function. Additionally, the 1,1 parameter basically dictates the number of threads to launch in the kernel.
* threadIdx.x, blockIdx.x: This is a unique ID that's given to all threads.
* cudaDeviceSynchronize(): All of the kernel calls in CUDA are asynchronous in nature. The host becomes free after calling the kernel and starts executing the next instruction afterward. This should come as no big surprise since this is a heterogeneous environment and hence both the host and device can run in parallel to make use of the types of processors that are available. In case the host needs to wait for the device to finish, APIs have been provided as part of CUDA programming that make the host code wait for the device function to finish. One such API is cudaDeviceSynchronize, which waits until all of the previous calls to the device have finished.

## GPU Architecture


CUDA thread [Executes as] CUDA Core/SIMD code 
CUDA block [Executes on] Streaming multiprocessor 
GRID/kernel [Executes on] GPU device

* CUDA Threads: CUDA threads execute on a CUDA core. CUDA threads are different from CPU threads. CUDA threads are extremely lightweight and provide fast context switching. The reason for fast context switching is due to the availability of a large register size in a GPU and hardware-based scheduler. The thread context is present in registers compared to CPU, where the thread handle resides in a lower memory hierarchy such as a cache. Hence, when one thread is idle/waiting, another thread that is ready can start executing with almost no delay. _Each CUDA thread must execute the same kernel and work independently on different data (SIMT)_.

* CUDA blocks: CUDA threads are grouped together into a logical entity called a CUDA block. CUDA blocks execute on a single Streaming Multiprocessor (SM). _One block runs on a single SM, that is, all of the threads within one block can only execute on cores in one SM and do not execute on the cores of other SMs_. Each GPU may have one or more SM and hence to effectively make use of the whole GPU; the user needs to divide the parallel computation into blocks and threads. 

* GRID/kernel: CUDA blocks are grouped together into a logical entity called a CUDA GRID. A CUDA GRID is then executed on the device.

## Vector Addition in Cuda

* **Memory allocation on GPU**: CPU memory and GPU memory are physically1. separate memory. malloc allocates memory on the CPU's RAM. The GPU kernel/device function can only access memory that's allocated/pointing to the device memory. To allocate memory on the GPU, _we need to use the cudaMalloc API_. _Unlike the malloc command, cudaMalloc does not return a pointer to allocated memory; instead, it takes a pointer reference as a parameter and updates the same with the allocated memory_. `cudaMalloc((void *)&d_a, N * sizeof(int));`

* **Transfer data from host memory to device memory**: The host data is then copied to the device's memory, which was allocated using the cudaMalloc command used in the previous step. The API that's used to copy the data between the host and device and vice versa is cudaMemcpy. Like other memcopy commands, this API requires the destination pointer, source pointer, and size. One additional parameter it takes is the direction of copy, that is, whether we are copying from the host to the device or from the device to the host. In the latest version of CUDA, this is optional since the driver is capable of understanding whether the pointer points to the host memory or device memory. Note that there is an asynchronous alternative to cudaMemcpy. This will be covered in more detail in other chapters.

* **<<<block size, thread size>>>**

* **cudaDeviceSynchronize**: wait for the kernel execution to finish.

* **transfer data from device to host**: Use the same `cudaMemcpy` API to copy the data back from the device to the host for post-processing or validation duties such as printing. The only change here, compared to the first step, is that _we reverse the direction of the copy, that is, the destination pointer points to the host while the source pointer points to the device allocated in memory_.

* **Free the allocated GPU memory**: use cudaFree

