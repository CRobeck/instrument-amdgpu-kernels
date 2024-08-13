#include "hip/hip_runtime.h"
#include <stdint.h>
#include <stdio.h>

#define WaveFrontSize 64

__attribute__((always_inline))
__device__ uint32_t getThreadIdInBlock() { return __builtin_amdgcn_workitem_id_x(); }

__attribute__((always_inline))
__device__ uint32_t getWaveId() {
  return getThreadIdInBlock() / WaveFrontSize;
}

__attribute__((used))
__device__ void memTrace(void* addressPtr, uint32_t LocationId, void* bufferPtr)
{

  uint64_t address = reinterpret_cast<uint64_t>(addressPtr);
  //Mask of the active threads in the wave
  int activeMask = __builtin_amdgcn_read_exec();
  //Find first active thread in the wave by finding the position of the least significant bit set to 1 in the activeMask
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
if (!bufferPtr){
//TODO: make this cleaner
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)	
	unsigned int xcc_id;
	asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xcc_id));
	printf("CYCLE: %ld, LocationId %d, Wave %d, SIMD %d, CU %d, SE %d, XCD %d, MEMTRACE: "
	       "0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,"
	       "0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,"
	       "0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,"
	       "0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx\n", 
	       Time, LocationId, (hw_id & 0xf), ((hw_id & 0x30) >> 4), ((hw_id & 0xf00) >> 8), ((hw_id & 0xe000) >> 13), xcc_id, 
	       addrArray[0],   addrArray[1], addrArray[2],  addrArray[3],  addrArray[4],  addrArray[5],  addrArray[6],  addrArray[7],  addrArray[8],  addrArray[9],  addrArray[10], addrArray[11], addrArray[12], addrArray[13], addrArray[14], addrArray[15],
	       addrArray[16], addrArray[17], addrArray[18], addrArray[19], addrArray[20], addrArray[21], addrArray[22], addrArray[23], addrArray[24], addrArray[25], addrArray[26], addrArray[27], addrArray[28], addrArray[29], addrArray[30], addrArray[31],
	       addrArray[32], addrArray[33], addrArray[34], addrArray[35], addrArray[36], addrArray[37], addrArray[38], addrArray[39], addrArray[40], addrArray[41], addrArray[42], addrArray[43], addrArray[44], addrArray[45], addrArray[46], addrArray[47],
	       addrArray[48], addrArray[49], addrArray[50], addrArray[51], addrArray[52], addrArray[53], addrArray[54], addrArray[55], addrArray[56], addrArray[57], addrArray[58], addrArray[59], addrArray[60], addrArray[61], addrArray[62], addrArray[63]);	  
#else
	printf("CYCLE: %ld, LocationId %d, Wave %d, SIMD %d, CU %d, SE %d, MEMTRACE: "
	       "0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,"
	       "0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,"
	       "0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,"
	       "0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx,  0x%lx\n", 
	       Time, LocationId, (hw_id & 0xf), ((hw_id & 0x30) >> 4), ((hw_id & 0xf00) >> 8), ((hw_id & 0xe000) >> 13),
	       addrArray[0],   addrArray[1], addrArray[2],  addrArray[3],  addrArray[4],  addrArray[5],  addrArray[6],  addrArray[7],  addrArray[8],  addrArray[9],  addrArray[10], addrArray[11], addrArray[12], addrArray[13], addrArray[14], addrArray[15],
	       addrArray[16], addrArray[17], addrArray[18], addrArray[19], addrArray[20], addrArray[21], addrArray[22], addrArray[23], addrArray[24], addrArray[25], addrArray[26], addrArray[27], addrArray[28], addrArray[29], addrArray[30], addrArray[31],
	       addrArray[32], addrArray[33], addrArray[34], addrArray[35], addrArray[36], addrArray[37], addrArray[38], addrArray[39], addrArray[40], addrArray[41], addrArray[42], addrArray[43], addrArray[44], addrArray[45], addrArray[46], addrArray[47],
	       addrArray[48], addrArray[49], addrArray[50], addrArray[51], addrArray[52], addrArray[53], addrArray[54], addrArray[55], addrArray[56], addrArray[57], addrArray[58], addrArray[59], addrArray[60], addrArray[61], addrArray[62], addrArray[63]);	  
#endif
  }
}
}
