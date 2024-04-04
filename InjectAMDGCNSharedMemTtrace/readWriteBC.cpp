#include "hip/hip_runtime.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

#define SHARED_SIZE 32

__global__ void kernel(int offset)
{
 int out;
    __shared__ uint32_t sharedMem[SHARED_SIZE];

    if (threadIdx.x == 0){
        for (int i = 0; i < SHARED_SIZE; i++) sharedMem[i] = 0;
    }
    __syncthreads();

    // repeatedly read and write to shared memory
    uint32_t index = threadIdx.x * offset;
    for (int i = 0; i < 10000; i++)
    {
        sharedMem[index] += index * i;
        index += 32;
        index %= SHARED_SIZE;
    }
}


// __global__ void vecAdd(int n, double *a, double *b, double *c) {
// int out;
// int tid = threadIdx.x;

// __asm__ __volatile__("s_mov_b32 m0 %1\n"\
//                      "s_mov_b32 %0 m0\n"\
//                       : "=s"(out) : "I" (5));
// printf("Thread %d m0 value: %d\n", tid, out);
//   int id = blockIdx.x * blockDim.x + threadIdx.x; 
//   if (id < n)
//     c[id] = a[id] + b[id];
// }

int main(int argc, char *argv[]) {

  // Size of vectors
  int n = 32;

  // Host input vectors
  // double *h_a;
  // double *h_b;
  // // Host output vector
  // double *h_c;

  // // Device input vectors
  // double *d_a;
  // double *d_b;
  // // Device output vector
  // double *d_c;

  // // Size, in bytes, of each vector
  // size_t bytes = n * sizeof(double);

  // // Allocate memory for each vector on host
  // h_a = (double *)malloc(bytes);
  // h_b = (double *)malloc(bytes);
  // h_c = (double *)malloc(bytes);

  // // Allocate memory for each vector on GPU
  // (void)hipMalloc(&d_a, bytes);
  // (void)hipMalloc(&d_b, bytes);
  // (void)hipMalloc(&d_c, bytes);

  // int i;
  // // Initialize vectors on host
  // for (i = 0; i < n; i++) {
  //   h_a[i] = sin(i) * sin(i);
  //   h_b[i] = cos(i) * cos(i);
  // }

  // // Copy host vectors to device
  // (void)hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
  // (void)hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);

  // int blockSize, gridSize;

  // // Number of threads in each thread block
  // blockSize = 32;

  // // Number of thread blocks in grid
  // gridSize = (int)ceil((float)n / blockSize);

  // Execute the kernel
  kernel<<<1, 5>>>(n);

  // Copy array back to host
  // (void)hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost);

  // // Sum up vector c and print result divided by n, this should equal 1 within
  // // error
  // double sum = 0;
  // for (i = 0; i < n; i++)
  //   sum += h_c[i];
  // printf("Result (should be 1.0): %f\n", sum / n);

  // // Release device memory
  // (void)hipFree(d_a);
  // (void)hipFree(d_b);
  // (void)hipFree(d_c);

  // // Release host memory
  // free(h_a);
  // free(h_b);
  // free(h_c);

  return 0;
}
