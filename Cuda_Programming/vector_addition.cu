#include "stdio.h"
#include "stdlib.h"

#define N 512

void add(int *a, int *b, int *c) {
    for (int idx=0; idx <N; idx++) {
        c[idx] = a[idx] + b[idx];
    }
}

void arangeN(int *data) {
    for (int idx=0; idx<N; idx++) {
        data[idx] = idx;
    }
}

void print(int *a, int *b, int *c) {
    for (int idx=N-10; idx<N; idx++) {
        printf("\n %d + %d = %d", a[idx] , b[idx], c[idx]);
    }    
}

__global__ void gpu_add(int *a, int* b, int* c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
}

int main() {
    int *a, *b, *c;
    int *a_, *b_, *c_;
    int size = N * sizeof(int);

    // allocate in Host
    a = (int *) malloc(size); 
    b = (int *) malloc(size); 
    c = (int *) malloc(size);

    // initialize in Host
    arangeN(a); 
    arangeN(b);
    
    // allocate device pointers
    cudaMalloc(&a_, N * sizeof(int));
    cudaMalloc(&b_, N * sizeof(int));
    cudaMalloc(&c_, N * sizeof(int));

    // copy the data to the device
    cudaMemcpy(a_, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_, c, size, cudaMemcpyHostToDevice);

    gpu_add<<<4,128>>>(a_, b_,c_);
    cudaMemcpy(c, c_, size, cudaMemcpyDeviceToHost);


    // perform operations
    print(a,b,c);
    free(a); free(b); free(c);
    cudaFree(a_); cudaFree(b_); cudaFree(c_);
    return 0;
}