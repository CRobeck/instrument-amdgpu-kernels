#include "hip/hip_runtime.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__attribute__((used)) __device__ void PrintKernel(int idx) {
    printf("m0 value: %d\n", idx);
}

// void AsmFunction(void)
// {
// int out;
// __asm__ __volatile__("s_mov_b32 m0 %1\n"\
//                      "s_mov_b32 %0 m0\n"\
//                       : "=s"(out) : "I" (5));
// }

// int out;
// int tid = threadIdx.x;

// __asm__ __volatile__("s_mov_b32 m0 %1\n"\
//                      "s_mov_b32 %0 m0\n"\
//                       : "=s"(out) : "I" (5));
// printf("Thread %d m0 value: %d\n", tid, out);  