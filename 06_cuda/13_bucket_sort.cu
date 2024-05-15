#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void count_keys(int *keys, int *buckets, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        atomicAdd(&buckets[keys[index]], 1);
    }
}

__global__ void fill_keys(int *keys, int *buckets, int *sum, int range) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < range && buckets[index] > 0) {
        int start = sum[index];
        int key = index;
        for (int i = 0; i < buckets[index]; i++) {
            keys[start + i] = key;
        }
    }
}


__global__ void exclusive_scan(int *input, int *output, int n) {
    extern __shared__ int temp[];
    int thid = threadIdx.x;
    int offset = 1;

    temp[thid] = (thid > 0) ? input[thid - 1] : 0;
    __syncthreads();

    for (; offset < n; offset *= 2) {
        if (thid >= offset)
            temp[thid] += temp[thid - offset];
        __syncthreads();
    }

    output[thid] = temp[thid];
}


int main() {
    int n = 50;
    int range = 5;
    int *keys, *buckets, *sum;
    int *d_keys, *d_buckets, *d_sum;

    keys = (int *)malloc(n * sizeof(int));
    buckets = (int *)calloc(range, sizeof(int));
    sum = (int *)malloc(range * sizeof(int));

    cudaMalloc(&d_keys, n * sizeof(int));
    cudaMalloc(&d_buckets, range * sizeof(int));
    cudaMalloc(&d_sum, range * sizeof(int));

    for (int i = 0; i < n; i++) {
        keys[i] = rand() % range;
        printf("%d ", keys[i]);
    }
    printf("\n");

    cudaMemcpy(d_keys, keys, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_buckets, 0, range * sizeof(int));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    count_keys<<<numBlocks, blockSize>>>(d_keys, d_buckets, n);
    cudaDeviceSynchronize();

    cudaMemcpy(buckets, d_buckets, range * sizeof(int), cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    exclusive_scan<<<1, range, range * sizeof(int)>>>(d_buckets, d_sum, range);
    cudaDeviceSynchronize();

    fill_keys<<<numBlocks, blockSize>>>(d_keys, d_buckets, d_sum, range);
    cudaDeviceSynchronize();

    cudaMemcpy(keys, d_keys, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("%d ", keys[i]);
    }
    printf("\n");

    free(keys);
    free(buckets);
    free(sum);
    cudaFree(d_keys);
    cudaFree(d_buckets);
    cudaFree(d_sum);

    return 0;
}

