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


int main(int argc, char *argv[]) {

  int offset = 32;

  int blockSize, gridSize;

  // Number of threads in each thread block
  blockSize = 32;

  // Number of thread blocks in grid
  gridSize = (int)ceil((float)offset / blockSize);

  // Execute the kernel
  kernel<<<gridSize, blockSize>>>(offset);

  return 0;
}
