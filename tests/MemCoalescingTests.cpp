#include "hip/hip_runtime.h"
#include <stdio.h>

// google test
#include <gtest/gtest.h>

#define N 10000

__global__ void IntPlusOne(int* array, int stride){
  int index = stride * threadIdx.x;
  array[index] = array[index] + 1;
}
extern __device__ uint32_t result;
TEST(MemCoalesingTest, Ints){
  int x[N] = {0};
  int* d_x;
  uint32_t h_result;
  (void)hipMalloc(&d_x, sizeof(int) * N);
  (void)hipMemcpy(d_x, x, sizeof(int)* N, hipMemcpyHostToDevice);

  int blocks = 1; int stride = 1;
  int threads = 32;   
  int expectedCacheLines = 1;
  IntPlusOne<<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines);


  blocks = 1; threads = 32; stride = 2;
  expectedCacheLines = 2;
  IntPlusOne<<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines);

  blocks = 1; threads = 32; stride = 32;
  expectedCacheLines = 32;
  IntPlusOne<<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines);

  (void)hipMemcpy(x, d_x, sizeof(int) * N, hipMemcpyDeviceToHost);
  (void)hipFree(x);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
