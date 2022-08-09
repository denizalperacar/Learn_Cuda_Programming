#include <stdio.h>
#include <stdlib.h>

__global__ void print_from_gpu(void) {
    printf("Printed From thread [%d, %d] from device\n", threadIdx.x, blockIdx.x);
}


int main() {
    print_from_gpu<<<2,1>>>();
    print_from_gpu<<<1,2>>>();
    printf ("print from host\n");
    cudaDeviceSynchronize();
    return 0;
}
