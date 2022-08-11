#include <stdio.h>
#include <stdlib.h>

#define SM 2

__global__ void naiveMatrixTranspose(int *input, int *output, int N, int M) {

    /*
    Assumes that a NxM matrix is flattened and provided to the Kernel
    */

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    int index_in = idx * M + idy;
    int index_out = idy * N + idx;
    output[index_out] = input[index_in];
}

__global__ void sharedMatrixTranspose(int *input, int *output, int N, int M) {

	__shared__ int sharedMemory [SM] [SM];

	// global index	
	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;

	// transposed global memory index
	int tindexX = threadIdx.x + blockIdx.y * blockDim.x;
	int tindexY = threadIdx.y + blockIdx.x * blockDim.y;

	// local index
	int localIndexX = threadIdx.x;
	int localIndexY = threadIdx.y;

	int index = indexY * N + indexX;
	int transposedIndex = tindexY * N + tindexX;

	// reading from global memory in coalesed manner and performing tanspose in shared memory
	sharedMemory[localIndexX][localIndexY] = input[index];

	__syncthreads();

	// writing into global memory in coalesed fashion via transposed data in shared memory
	output[transposedIndex] = sharedMemory[localIndexY][localIndexX];
}

void fillMatrix(int *Mat, int N, int M){
    int idx;
    int count = 0;
    for (int i=0; i<N; i++) {
        for (int j=0; j<M; j++) {
            idx = i * M + j;
            Mat[idx] = count;
            count++;
        }
    }
}

void printMatrix(int *Mat, int N, int M) {
    int idx;
    for (int i=0; i<N; i++) {
        for (int j=0; j<M; j++) {
            idx = i * M + j;
            printf("%d ", Mat[idx]);
        }
        printf("\n");
    }
}

void matTranspose(int *inp, int *out, int N, int M, int bs) {
    int *cudVar, *cudOut;
    int size = N * M * sizeof(int);
    cudaMalloc(&cudVar, size);
    cudaMemcpy(cudVar, inp, size, cudaMemcpyHostToDevice);
    cudaMalloc(&cudOut, size);

    dim3 blockSize(bs, bs, 1);
    dim3 gridSize(N/bs, M/bs, 1);

    // naiveMatrixTranspose<<<blockSize, gridSize>>>(cudVar, cudOut, N, M);
    sharedMatrixTranspose<<<blockSize, gridSize>>>(cudVar, cudOut, N, M);
    cudaMemcpy(out, cudOut, size, cudaMemcpyDeviceToHost);
    // Free Device Memory
    cudaFree(cudVar); cudaFree(cudOut);
}


int main() {


    int N = 6;
    int M = 6;
    int bs = 2;
    int size = N * M * sizeof(int);
    int *Mat, *TransposedMat;

    Mat = (int *) malloc(size);
    TransposedMat = (int *) malloc(size);

    fillMatrix(Mat, N, M);
    printMatrix(Mat, N, M);

    
    matTranspose(Mat, TransposedMat, N, M, bs);
    printMatrix(TransposedMat, M, N);

    free(Mat); free(TransposedMat);

    return 0;
}