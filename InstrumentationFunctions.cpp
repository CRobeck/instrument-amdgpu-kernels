#include "hip/hip_runtime.h"
#include <stdint.h>
#include <stdio.h>

#define WarpSize 32

__attribute__((used)) 
__attribute__((always_inline))
__device__ void PrintCacheLines(uint32_t idx) {
    if(threadIdx.x == 0)
      printf("Value to send to trace stream:    %u    NumCacheLines: %u    LocationIdx: %u\n", idx,
            static_cast<uint32_t>(idx >> 26), static_cast<uint32_t>(idx & (67108864 - 1)));
}

__attribute__((always_inline))
__device__ int getNthBit(uint32_t bitArray, int nth){
  return 1 & (bitArray >> nth);
}

__attribute__((always_inline))
 __device__ bool isSharedMemPtr(const void *Ptr) {
  return __builtin_amdgcn_is_shared(
      (const __attribute__((address_space(0))) void *)Ptr);
}

__attribute__((used)) __device__ void memTrace(void* addressPtr)
{
	uint64_t address = reinterpret_cast<uint64_t>(addressPtr);	
    uint32_t lower = address & 0xffffffff, 
    uint32_t upper = address >> 32;
	int m0_save;
    __asm__ __volatile__("s_mov_b32 %0 m0\n"    //save the existing value in M0 to the m0_save variable
                         "s_mov_b32 m0 %1\n"    //set the value of M0 to value of lower 32 bits
                         "s_nop 0\n"            //Required before a s_ttracedata instruction
                         "s_ttracedata\n"       //Send data from M0 into thread trace stream
    	                 "s_mov_b32 m0 %2\n"    //set the value of M0 to value of upper 32 bits
                         "s_nop 0\n"            //Required before a s_ttracedata instruction
                         "s_ttracedata\n"       //Send data from M0 into thread trace stream			      
                         "s_mov_b32 m0 %0\n"    //Restore the value of M0 from m0_save
                          : "=s"(m0_save) : "s" (lower) : "s" (upper));
}

//Needed for gtest
#ifdef BUILD_TESTING
	__attribute__((used)) __device__ uint32_t result;
#endif
__attribute__((used)) __device__ uint32_t numCacheLines(void* addressPtr, uint32_t LoadOrStore, uint32_t LocationIdx, uint32_t typeSize){
  uint32_t NumCacheLines = 1;
 //TODO: See if this is check is actually needed since we're already checking for addresspace 3 or 4
 //in the compiler pass before injecting this function
 if(isSharedMemPtr(addressPtr))
   return NumCacheLines;

  int activeThreadMask =__ballot(1);

  uint64_t address = reinterpret_cast<uint64_t>(addressPtr);

  uint64_t addrArray[2 * WarpSize];

  int baseThread = -1;
  for(int i = 0; i < WarpSize; i++)
    if(getNthBit(activeThreadMask, i) == 1){
      baseThread = i;
      break;
    }

  // Shuffle values from all threads into addrArray using active threads
  for(int i = 0; i < WarpSize; i++){
    if(getNthBit(activeThreadMask, i) == 0)
      addrArray[2 * i] = address;
    else{
      addrArray[2 * i] = __shfl(address, i, WarpSize);
    }
  }
  
  uint32_t LaneId = (WarpSize - 1) & threadIdx.x;
  if(baseThread == LaneId){
    NumCacheLines = 1;
    // Divide all threads by cacheLineSize (128 bytes). Every other thread represents the max address that
    // is accessed then compute (address + typeSize - 1) / cacheLineSize (128 bytes).
    for(int i = 0; i < WarpSize; i++){
      addrArray[2 * i + 1] = (addrArray[2 * i] + typeSize - 1) >> 7;
      addrArray[2 * i] >>= 7;
    }

    uint64_t baseAddr = addrArray[0];
    for(int i = 0; i < 2 * WarpSize; i++)
      if(addrArray[i] != baseAddr){
        uint64_t current = addrArray[i];
        NumCacheLines++;
        for(int j = i + 1; j < 2 * WarpSize; j++)
          if(addrArray[j] == current)
            addrArray[j] = baseAddr;
      }
#ifdef BUILD_TESTING
  result = NumCacheLines;
#endif
  }
  return ((NumCacheLines << 26) | LocationIdx);
}
