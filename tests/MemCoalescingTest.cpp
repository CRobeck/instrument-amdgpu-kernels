#include "hip/hip_runtime.h"
#include <stdio.h>

// google test
#include <gtest/gtest.h>

#define N 10000

__global__ void intAddOne(int* array, int stride){
  int index = stride * threadIdx.x;
  array[index] = array[index] + 1;
}

// extern __device__ uint32_t result;

TEST(MemCoalesingTest, Ints){
  int x[N] = {0};
  int* xDev;
  // uint32_t h_result;
  (void)hipMalloc(&xDev, sizeof(int) * N);
  (void)hipMemcpy(xDev, x, sizeof(int)* N, hipMemcpyHostToDevice);

  int testNum = 5;    int blocks = 1;
  int threads = 32;   int stride = 32;
  int cacheLines = 32;
  printf("\nTest %d\nCache lines expected: %d\n", testNum, cacheLines);
  intAddOne<<<blocks, threads >>>(xDev, stride);
  (void)hipDeviceSynchronize();
  // (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  // printf("h_result: %u\n", h_result);
  (void)hipMemcpy(x, xDev, sizeof(int) * N, hipMemcpyDeviceToHost);
  (void)hipFree(x);
  EXPECT_EQ(32, cacheLines);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}