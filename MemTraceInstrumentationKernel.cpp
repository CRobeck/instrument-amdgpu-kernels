#include "hip/hip_runtime.h"
#include <stdint.h>
#include <stdio.h>

__attribute__((used)) 
__attribute__((always_inline))
__device__ void memTrace(void* addressPtr)
{
    uint64_t address = reinterpret_cast<uint64_t>(addressPtr);
    uint32_t lower = address & 0xffffffff;
    uint32_t upper = address >> 32;
    uint32_t m0_save, new_m0;
	__asm__ __volatile__(
		"s_mov_b32 %0 m0\n"           // Save the existing value in M0 to the m0_save variable
		"v_readfirstlane_b32 %1 %2\n" // Read the value of the lower 32 bits from the first thread
		"s_mov_b32 m0 %1\n"           // Set the value of M0 to value of lower 32 bits
		"s_nop 0\n"                   // Required before a s_ttracedata instruction
		"s_ttracedata\n"              // Send data from M0 into thread trace stream
		"v_readfirstlane_b32 %1 %3\n" // Read the value of the upper 32 bits from the first thread
		"s_mov_b32 m0 %1\n"           // Set the value of M0 to value of upper 32 bits
		"s_nop 0\n"                   // Required before a s_ttracedata instruction
		"s_ttracedata\n"              // Send data from M0 into thread trace stream
		"s_mov_b32 m0 %0\n"           // Restore the value of M0 from m0_save
	         : "=s"(m0_save), "=s"(new_m0) : "v"(lower), "v"(upper) : "memory");
}

