#include "hip/hip_runtime.h"
#include <stdint.h>
#include <stdio.h>

#define WaveFrontSize 64
#define HexLen 15

__attribute__((always_inline))
__device__ uint32_t getThreadIdInBlock() { return __builtin_amdgcn_workitem_id_x(); }

__attribute__((always_inline))
__device__ uint32_t getWaveId() {
  return getThreadIdInBlock() / WaveFrontSize;
}
__attribute__((always_inline))
 __device__ bool isSharedMemPtr(const void *Ptr) {
  return __builtin_amdgcn_is_shared(
      (const __attribute__((address_space(0))) void *)Ptr);
}

__attribute__((used))
__device__ void memTrace(void* addressPtr, uint32_t LocationId){
 if(isSharedMemPtr(addressPtr))
   return;
  uint64_t address = reinterpret_cast<uint64_t>(addressPtr);
  //Mask of the active threads in the wave
  int activeMask = __builtin_amdgcn_read_exec();
//  //Find first active thread in the wave by finding the position of the least significant bit set to 1 in the activeMask
  const int firstActiveLane = __ffs(activeMask) - 1;
  uint64_t addrArray[WaveFrontSize];
  for(int i = 0; i < WaveFrontSize; i++){
	  addrArray[i] = __shfl(address, i, WaveFrontSize);
  }
   uint32_t Lane = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));

  if(Lane == firstActiveLane){
	unsigned int hw_id = 0;
	uint64_t Time = 0;
#if !defined(__gfx1100__) && !defined(__gfx1101__)	
	Time = __builtin_amdgcn_s_memrealtime();	
	asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(hw_id));
#endif	
	char hex_str[]= "0123456789abcdef";
	char out[WaveFrontSize*HexLen + 1];
	(out)[WaveFrontSize*HexLen] = '\0';
	for (size_t i = 0; i < WaveFrontSize; i++) {
	        (out)[i * HexLen + 0] = '0';
                (out)[i * HexLen + 1] = 'x';
		(out)[i * HexLen + 2] = hex_str[(addrArray[i] >> 44) & 0x0F];
		(out)[i * HexLen + 3] = hex_str[(addrArray[i] >> 40) & 0x0F];
		(out)[i * HexLen + 4] = hex_str[(addrArray[i] >> 36) & 0x0F];
		(out)[i * HexLen + 5] = hex_str[(addrArray[i] >> 32) & 0x0F];
		(out)[i * HexLen + 6] = hex_str[(addrArray[i] >> 28) & 0x0F];
		(out)[i * HexLen + 7] = hex_str[(addrArray[i] >> 24) & 0x0F];
		(out)[i * HexLen + 8] = hex_str[(addrArray[i] >> 20) & 0x0F];
		(out)[i * HexLen + 9] = hex_str[(addrArray[i] >> 16) & 0x0F];
		(out)[i * HexLen + 10] = hex_str[(addrArray[i] >> 12) & 0x0F];
		(out)[i * HexLen + 11] = hex_str[(addrArray[i] >> 8) & 0x0F];
		(out)[i * HexLen + 12] = hex_str[(addrArray[i] >> 4) & 0x0F];
      		(out)[i * HexLen + 13] = hex_str[(addrArray[i]     ) & 0x0F];
		(out)[i * HexLen + 14] = ',';
	}
	(out)[WaveFrontSize * HexLen - 1] = '\n';
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)	
	unsigned int xcc_id;
	asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xcc_id));
	printf("%ld,%d,%d,%d,%d,%d,%d, %s", Time, LocationId, (hw_id & 0xf), ((hw_id & 0x30) >> 4), ((hw_id & 0xf00) >> 8), ((hw_id & 0xe000) >> 13), xcc_id, out);
#else
	printf("%ld,%d,%d,%d,%d,%d,%s", Time, LocationId, (hw_id & 0xf), ((hw_id & 0x30) >> 4), ((hw_id & 0xf00) >> 8), ((hw_id & 0xe000) >> 13),out);
#endif	

  }
}

