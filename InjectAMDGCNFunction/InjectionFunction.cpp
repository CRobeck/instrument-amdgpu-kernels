#include "hip/hip_runtime.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__attribute__((used)) __device__ void PrintKernel(int idx) {
  if (idx < 10)
    printf("Injected Function: %s, Idx: %d\n", __func__, idx);
}
