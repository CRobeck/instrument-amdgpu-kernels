#include "hip/hip_runtime.h"
#include <stdint.h>
#include <stdio.h>

#define WarpSize 32
#define WaveFrontSize 64
#define cacheLineSize 128

__attribute__((always_inline)) __device__ int
isThreadActive(uint32_t activeMaskArray, int i) {
  return 1 & (activeMaskArray >> i);
}

__attribute__((always_inline)) __device__ bool isSharedMemPtr(const void *Ptr) {
  return __builtin_amdgcn_is_shared(
      (const __attribute__((address_space(0))) void *)Ptr);
}

// Needed for gtest
#ifdef BUILD_TESTING
extern __device__ uint32_t result;
#endif
__attribute__((used)) __device__ uint32_t numCacheLines(void *addressPtr,
                                                        uint32_t LocationIdx,
                                                        uint32_t typeSize) {
  uint32_t NumCacheLines = 1;
  // TODO: See if this is check is actually needed since we're already checking
  // for addresspace 3 or 4 in the compiler pass before injecting this function
  if (isSharedMemPtr(addressPtr))
    return NumCacheLines;

  int activeThreadMask = __builtin_amdgcn_read_exec();

  uint64_t address = reinterpret_cast<uint64_t>(addressPtr);

  uint64_t addrArray[2 * WarpSize];

  // Shuffle values from all threads into addrArray using only the active
  // threads
  for (int i = 0; i < WarpSize; i++) {
    if (!isThreadActive(activeThreadMask, i))
      addrArray[2 * i] = address;
    else {
      // Broadcast lane index i's address value to all other threads addrArray
      // 2 * i represents the starting address of lane i's read/write access
      // we will fill in end address later
      addrArray[2 * i] = __shfl(address, i, WarpSize);
    }
  }

  uint32_t LaneId =
      __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
  const int firstActiveLane = __ffs(activeThreadMask) - 1;
  if (LaneId == firstActiveLane) {
    NumCacheLines = 1;
    // Divide all threads address values by cacheLineSize (128 bytes) to
    // determine the required number of memory transactions. Odd index values
    // (i.e. 2 * i + 1) represent the end address of the access (start address +
    // data type size) Even indexes represent the starting address of the access
    for (int i = 0; i < WarpSize; i++) {
      addrArray[2 * i + 1] =
          (addrArray[2 * i] + typeSize - 1) / cacheLineSize; // ending address
      addrArray[2 * i] /= cacheLineSize;                     // starting address
    }
    // After we've divided the address by the cache line size it is assumed that
    // if the value in the addrArray is not the same as the base address it will
    // require a seperate memory transaction to fetch.
    for (int i = 0; i < 2 * WarpSize; i++) {
      if (addrArray[i] != addrArray[0]) {
        NumCacheLines++;
        // If the address is one we have already seen (i.e. not unique) then it
        // won't require an additional transaction therefore just set any
        // duplicate values to the base address to keep us from counting it
        // again
        for (int j = i + 1; j < 2 * WarpSize; j++) {
          if (addrArray[j] == addrArray[i])
            addrArray[j] = addrArray[0];
        }
      }
    }
#ifdef BUILD_TESTING
    result = NumCacheLines;
#endif
  }
  return ((NumCacheLines << 26) | LocationIdx);
}
