# LLVM Based Instrumention of AMDGPU Kernels

LLVM provides a variety of pass APIs to interact with, and modify, the compilation pipeline. The goal of this project is to develop a set of transformation passes to instrument AMDGPU kernels to get a variety of performance related information. The passes and examples are developed to be used with the AMDGPU software stack, Rocm. Although HIP kernels can be compiled directly with clang/clang++ (i.e. clang++ -x hip) the vast majority of Rocm developers use the HIP compiler driver [hipcc](https://github.com/ROCm/llvm-project/tree/amd-staging/amd/hipcc#hipcc), therefore the instrumentation passes and examples presented focus on getting the LLVM pass manager and compiler tool chain to interact with both Rocm and hipcc.

A list of the currently implemented instrumentation passes is below. The list is under development and being actively added to.

### Implemented Instrumentation Passes
[Device Function Kernel Injection](#example-1-transformation-pass-to-inject-a-device-function-into-an-amdgpu-kernel) - Transformation pass that inserts (injects) a device function into an existing HIP GPU kernel.

[Read Register Contents With Inline ASM Injection](#example-2-transformation-pass-to-inject-reading-register-contents-into-an-amdgpu-kernel) - Transformation pass that inserts (injects) an Inline ASM function that reads the value in the vector register VGPR V0, makes a new integer variable, places the register contents in to the new variable, and injects it into an existing HIP GPU kernel.

# Getting Started

## Build LLVM
```bash
cd $HOME
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build
cd build
cmake -GNinja -DLLVM_ENABLE_PROJECTS='llvm;clang;lld' \
-DLLVM_TARGETS_TO_BUILD='X86;AMDGPU' \
-DCMAKE_C_COMPILER=clang \
-DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DCMAKE_INSTALL_PREFIX=$HOME/llvm-project/build \
../llvm
cmake --build .
```
# Example 1: Transformation Pass To Inject a Device Function Into An AMDGPU Kernel

## Overview
Take the following AMDGPU kernel which adds two vectors
```C++
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < n)
        c[id] = a[id] + b[id];
}
```
and the following instrumentation device function which prints an integer, if less than some threshold, that are passed to it.

```C++
__device__ void PrintKernel(int idx){
    if(idx < 10)
      printf("Injected %s: %d\n", __func__, idx);
}
```
We would like to develop a LLVM Transformation Pass which inserts this device function at the beginning of the GPU kernel such that the equivalent C++ code, which will never actually exist since the call is added internally in the compiler, would be
```C++
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    PrintKernel(threadIdx.x);
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < n)
        c[id] = a[id] + b[id];
}
```
The steps to do this, with the Rocm/HIP toolchain, are outlined below.
### Build the InjectAMDGPUFuncCall LLVM Pass
```bash
git clone https://github.com/CRobeck/InstrumentAMDGPUKernels.git
cd InstrumentAMDGPUKernels
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
-DLLVM_INSTALL_DIR=$HOME/llvm-project/build ..
cmake --build .
```
### Build the baseline, uninstrumented, version
```bash
hipcc $PWD/InjectAMDGCNFunction/vectorAdd.cpp -o vectorAdd
```

### Build the instrumented version using hipcc and rdc
```bash
hipcc -c -fgpu-rdc -fpass-plugin=$PWD/build/lib/libInjectAMDGCNFunction.so \
$PWD/InjectAMDGCNFunction/vectorAdd.cpp -o vectorAdd.o
hipcc -c -fgpu-rdc $PWD/InjectAMDGCNFunction/InjectionFunction.cpp -o InjectionFunction.o
hipcc -fgpu-rdc InjectionFunction.o vectorAdd.o -o instrumented
```

Some magic behind the scenes: One might notice that the instrumentation function PrintKernel is defined in InjectionFunction.cpp but used in vectorAdd.cpp. This would usually be dealt with using a forward declaration for PrintKernel in vectorAdd.cpp and the rdc flag in hipcc, the function resolution then handled by the linker. However, this is unattractive for our use case for a variety of reasons:

1. It requires modification of the uninstrumented file(s) with, a potentially large number of, instrumentation functions.
2. The actually instrumentation function(s) that would need to be forward declared is unknown until the actual instrumentation pass is called. Therefore this would essentially require forward declaring every instrumentation function in every file containing a possible target kernel function.

Therefore the instrumentation function pass adds an external function call declaration for each instrumentation function that the pass adds such that the equivalent C++ code for the vectorAdd kernel, after the device function injection pass, would look like

```C++
__device__ void PrintKernel(int idx);
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    PrintKernel(threadIdx.x);
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < n)
        c[id] = a[id] + b[id];
}
```

### Run it
```bash
./vectorAdd
./instrumented
```
### Output

<table>
<tr>
<th>Baseline</th>
<th>Instrumented</th>
</tr>
<tr>
<td>

```bash
Result (should be 1.0): 1.000000
```

</td>
<td>

```bash
Function To Be Injected: _Z11PrintKerneli
Injecting Device Function Into AMDGPU Kernel: _Z6vecAddPdS_S_i
Injected Function: PrintKernel, Idx: 0
Injected Function: PrintKernel, Idx: 1
Injected Function: PrintKernel, Idx: 2
Injected Function: PrintKernel, Idx: 3
Injected Function: PrintKernel, Idx: 4
Injected Function: PrintKernel, Idx: 5
Injected Function: PrintKernel, Idx: 6
Injected Function: PrintKernel, Idx: 7
Injected Function: PrintKernel, Idx: 8
Injected Function: PrintKernel, Idx: 9
Result (should be 1.0): 1.000000
```

</td>
</tr>
</table>

# Example 2: Transformation Pass To Inject Reading Register Contents Into An AMDGPU Kernel
We again take the following AMDGPU kernel which adds two vectors
```C++
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < n)
        c[id] = a[id] + b[id];
}
```
We again would like to insert a device function that prints the kernel name and threadIdx.x. However, in this case we are going to instead read the thread index directly from the vector register (VGPR0) that holds it. The equivalent C++ would be
```C++
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    int threadIdx_x;
    __asm__ __volatile__("v_mov_b32 %0 v0\n" : "=v"(threadIdx_x)); //v0 holds threadIdx.x
    PrintKernel(threadIdx_x);
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < n)
        c[id] = a[id] + b[id];
}
```
The steps to do this, with the Rocm/HIP toolchain, are the same as the previous example just using the InjectAMDGCNInlineASM pass instead.

### Build the instrumented version using hipcc and rdc
```bash
hipcc -c -fgpu-rdc -fpass-plugin=$PWD/build/lib/libInjectAMDGCNInlineASM.so \
$PWD/InjectAMDGCNInlineASM/vectorAdd.cpp -o vectorAdd.o
hipcc -c -fgpu-rdc $PWD/InjectAMDGCNInlineASM/InjectionFunction.cpp -o InjectionFunction.o
hipcc -fgpu-rdc InjectionFunction.o vectorAdd.o -o instrumented
```

We notice identical output from the previous example however in this case a call to the injected Inline ASM would show up in the dissassembled ISA.
