# LLVM/MLIR Based Instrumentation of AMDGPU Kernels

LLVM/MLIR provide a variety of pass APIs to interact with, and modify, the compilation pipeline. The goal of this project is to develop a set of transformation passes to instrument AMDGPU kernels to get a variety of performance analysis and optimization related information. The passes and examples are developed to be used with the AMDGPU software stack HIP/Rocm, the AMDGPU LLVM backend.

Although HIP kernels can be compiled directly with clang/clang++ (i.e., clang++ -x hip) the vast majority of Rocm developers use the HIP compiler driver [hipcc](https://github.com/ROCm/llvm-project/tree/amd-staging/amd/hipcc#hipcc) or a MLIR ML Compiler pipeline (e.g. Triton, PyTorch, IREE). Therefore, the instrumentation passes and examples presented focus on getting the LLVM 17+ tool chain (LLVM/MLIR) and new pass manager integrated with Rocm, [6.0.2](https://github.com/ROCm/llvm-project/tree/rocm-6.0.2) at the time of writing, and hipcc. 

A list of the currently implemented instrumentation passes is below. The list is under development and being actively added to.

### Implemented Instrumentation Passes
[Device Function Kernel Injection](#example-1-transformation-pass-to-inject-a-device-function-into-an-amdgpu-kernel) - Transformation pass that inserts (injects) a device function into an existing HIP GPU kernel. This pass is run right after passes that do basic simplification of the input IR.

[Read Register Contents With Inline ASM Injection](#example-2-transformation-pass-to-inject-reading-register-contents-into-an-amdgpu-kernel) - Transformation pass that inserts (injects) an Inline ASM function that reads the value in the vector register VGPR V0, makes a new integer variable, places the register contents in to the new variable, and injects it into an existing HIP GPU kernel. This pass is run right after passes that do basic simplification of the input IR.

[Instrument LDS Reads and Writes With Thread Trace Instructions to Detect Bank Conflicts](#example-3-instrument-lds-reads-and-writes-with-thread-trace-instructions-to-detect-bank-conflicts) - Transformation pass that inserts (injects) an Inline ASM function to emit s_ttracedata instruction prior to each LDS load or store instruction, sets M0 to a unique integer for each of the s_ttracedata instructions, and resets M0 to its default value after the s_ttracedata instruction it into an existing HIP GPU kernel. Nops are inserted as needed. The injected s_ttracedata instructions can then be used in down stream profiling tools for detecting bank conflicts. This pass is run at the very end of the function optimization pipeline.

[Instrument Global Reads and Writes to Detect Uncoalesced Memory Accesses](#example-4-instrument-global-reads-and-writes-to-detect-uncoalesced-memory-accesses) - Transformation pass that inserts a function to count the number of cache lines a global load or store uses to determine uncoalesed accesses. Any number of memory transations (cache lines) needed for a particular load or store great than one indicates an uncoalesed accesses. This pass is run at the very end of the function optimization pipeline.

[Instrument Global Reads and Writes to Generate Memory Traces in Triton MLIR Based ML Compiler](#example-5-instrument-global-reads-and-writes-to-generate-memory-traces-in-triton-mlir-based-ml-compiler) - Transformation pass that inserts a function to output the per wave addresses of each global load and store. This pass is run at the very end of the function optimization pipeline.


# Getting Started
Assuming you have a system with Rocm installed  set the correct paths and environment variables. An example module file would be:

```
module-whatis   rocm
prepend-path    PATH /opt/rocm/bin:/opt/rocm/llvm/bin
prepend-path    CMAKE_PREFIX_PATH /opt/rocm
prepend-path    LIBRARY_PATH /opt/rocm/lib:/opt/rocm/hip/lib:/opt/rocm/hsa/lib:/opt/rocm/lib64:/opt/rocm/opencl/lib:/opt/rocm/opencl/lib/x86_64:/opt/rocm/llvm/lib
prepend-path    LD_LIBRARY_PATH /opt/rocm/lib:/opt/rocm/hip/lib:/opt/rocm/hsa/lib:/opt/rocm/lib64:/opt/rocm/opencl/lib:/opt/rocm/opencl/lib/x86_64:/opt/rocm/llvm/lib
prepend-path    LD_RUN_PATH /opt/rocm/lib:/opt/rocm/hip/lib:/opt/rocm/hsa/lib:/opt/rocm/lib64:/opt/rocm/opencl/lib:/opt/rocm/opencl/lib/x86_64:/opt/rocm/llvm/lib
prepend-path    PKG_CONFIG_PATH /opt/rocm/lib/pkgconfig
prepend-path    MANPATH /opt/rocm/share/man
prepend-path    CPATH /opt/rocm/include:/opt/rocm/include/hip:/opt/rocm/hsa/include:/opt/rocm/llvm/include
setenv          ROCM_PATH /opt/rocm
setenv          HIP_PATH /opt/rocm
setenv          ROCM_LLVM /opt/rocm/llvm
setenv          DEVICE_LIB_PATH /opt/rocm/amdgcn/bitcode
```

### Build the AMDGPU Instrumentation LLVM Passes
```bash
git clone https://github.com/CRobeck/instrument-amdgpu-kernels.git
cd instrument-amdgpu-kernels
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
-DLLVM_INSTALL_DIR=$ROCM_LLVM ..
cmake --build .
```

# Example 1: Transformation Pass To Inject a Device Function Into An AMDGPU Kernel

Take the following AMDGPU kernel which adds two vectors
```C++
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < n)
        c[id] = a[id] + b[id];
}
```
and the following instrumentation device function which prints an integer, if less than some threshold, that are passed to it. The threshold is hardcoded to 10 in this case but could easily be added as another kernel argument which the pass adds as well.

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
The steps to do this, with the Rocm/HIP toolchain, using the [InjectAMDGCNFunc](lib/InjectAMDGCNFunction.cpp) pass, are outlined below.

### Build the baseline, uninstrumented, version
```bash
hipcc $PWD/examples/vectorAdd.cpp -o vectorAdd
```

### Build the instrumented version using hipcc and rdc
```bash
hipcc -c -fgpu-rdc -fpass-plugin=$PWD/build/lib/libInjectAMDGCNFunction.so \
$PWD/examples/vectorAdd.cpp -o vectorAdd.o
hipcc -c -fgpu-rdc $PWD/examples/ExampleInjectionFunction.cpp -o InjectionFunction.o
hipcc -fgpu-rdc InjectionFunction.o vectorAdd.o -o instrumented
```

Some magic behind the scenes: One might notice that the instrumentation function PrintKernel is defined in ExampleInjectionFunction.cpp but used in vectorAdd.cpp. This would usually be dealt with using a forward declaration for PrintKernel in vectorAdd.cpp and the rdc flag in hipcc, the function resolution then handled by the linker. However, this is unattractive for our use case for a variety of reasons:

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
The steps to do this, with the Rocm/HIP toolchain, are the same as the previous example just using the [InjectAMDGCNInlineASM](lib/InjectAMDGCNInlineASM.cpp) pass instead.

### Build the instrumented version using hipcc and rdc
```bash
hipcc -c -fgpu-rdc -fpass-plugin=$PWD/build/lib/libInjectAMDGCNInlineASM.so \
$PWD/examples/vectorAdd.cpp -o vectorAdd.o
hipcc -c -fgpu-rdc $PWD/examples/ExampleInjectionFunction.cpp -o InjectionFunction.o
hipcc -fgpu-rdc InjectionFunction.o vectorAdd.o -o instrumented
```

We notice identical output from the previous example however in this case a call to the injected Inline ASM would show up in the dissassembled ISA.

# Example 3: Instrument LDS Reads and Writes With Thread Trace Instructions to Detect Bank Conflicts
The s_ttracedata instruction takes whatever data is in the M0 register at the time the instruction is called and sends it to thread trace stream to be viewed during profiling.

In this example we take a HIP kernel with known bank conflicts and instrument the shared memory ds_reads and ds_writes and inject the following instructions:

```bash
__asm__ __volatile__("s_mov_b32 $0 m0\n"    //save the existing value in M0 to the m0_save variable
                     "s_mov_b32 m0 $1\n"    //set the value of M0 to value of ttrace_counter variable, the value we want to send to thread trace stream
                     "s_nop 0\n"            //Required before a s_ttracedata instruction
                     "s_ttracedata\n"       //Send data from M0 into thread trace stream
                     "s_mov_b32 m0 $0\n"    //Restore the value of M0 from m0_save
                     "s_add_i32 $1 $1 1\n"  //Increment the s_ttracedata counter variable, ttrace_counter
                      : "=s"(m0_save) : "s" (ttrace_counter));
```
ttrace_counter is an global integer value used to identify each s_ttracedata. The ttrace_counter integer variable is injected and handled entirely by the InjectAMDGCNSharedMemTtrace pass. 

An additional thing that is slightly different in this example, compared to the previously presented ones, is where the pass is run in the compiler pass pipeline. In the pass initalization we replace registerPipelineEarlySimplificationEPCallback with registerOptimizerLastEPCallback. This moves the pass from right after passes that do basic simplification of the input IR to the very end of the function optimization pipeline. The reason for this is the ds_reads and ds_writes are often, if not always, found inside loops. The compiler may, or may not, unroll the loops. Therefore we need to make sure when we inject the s_ttracedata instructions it is done after the loop unrolling is done to get both the correct number and placement of the injected s_ttracedata instruction in each loop iteration. If we kept the pass in the original spot using EarlySimplificationEPCallback it would be impossible to know, at the the time the pass is run, how many s_ttracedata will get actually get injected into the ISA.

The steps to do this, with the Rocm/HIP toolchain, are the same as before just swapping out in the [InjectAMDGCNSharedMemTtrace](lib/InjectAMDGCNSharedMemTtrace.cpp) pass.

### Build the instrumented version using hipcc and rdc
```bash
hipcc -ggdb --save-temps -c -fgpu-rdc -ggdb -fpass-plugin=$PWD/build/lib/libInjectAMDGCNSharedMemTtrace.so \
$PWD/InjectAMDGCNSharedMemTtrace/readWriteBC.cpp -o readWriteBC.o
hipcc --save-temps -fgpu-rdc readWriteBC.o -o instrumented
llvm-objdump -d a.out-hip-amdgcn-amd-amdhsa-gfx90a > instrumented-amdgcn-isa.log
```
### Inspecting The Instrumented ISA

Looking at the instrumented-amdgcn-isa.log file we can see the desired ASM instructions inserted correctly, before each ds_read and ds_write instruction, in the ISA.

A unique identifying index of each s_ttracedata instruction will be printed to the terminal along with its corresponding source file, line, and column number.

```bash
0 _Z6kerneli InjectAMDGCNSharedMemTtrace/readWriteBC.cpp:16:51
1 _Z6kerneli InjectAMDGCNSharedMemTtrace/readWriteBC.cpp:16:51
2 _Z6kerneli InjectAMDGCNSharedMemTtrace/readWriteBC.cpp:16:51
3 _Z6kerneli InjectAMDGCNSharedMemTtrace/readWriteBC.cpp:16:51
4 _Z6kerneli InjectAMDGCNSharedMemTtrace/readWriteBC.cpp:16:51
5 _Z6kerneli InjectAMDGCNSharedMemTtrace/readWriteBC.cpp:16:51
6 _Z6kerneli InjectAMDGCNSharedMemTtrace/readWriteBC.cpp:16:51
7 _Z6kerneli InjectAMDGCNSharedMemTtrace/readWriteBC.cpp:16:51
8 _Z6kerneli InjectAMDGCNSharedMemTtrace/readWriteBC.cpp:16:51
9 _Z6kerneli InjectAMDGCNSharedMemTtrace/readWriteBC.cpp:16:51
10 _Z6kerneli InjectAMDGCNSharedMemTtrace/readWriteBC.cpp:16:51
11 _Z6kerneli InjectAMDGCNSharedMemTtrace/readWriteBC.cpp:16:51
12 _Z6kerneli InjectAMDGCNSharedMemTtrace/readWriteBC.cpp:16:51
13 _Z6kerneli InjectAMDGCNSharedMemTtrace/readWriteBC.cpp:16:51
14 _Z6kerneli InjectAMDGCNSharedMemTtrace/readWriteBC.cpp:16:51
15 _Z6kerneli InjectAMDGCNSharedMemTtrace/readWriteBC.cpp:16:51
16 _Z6kerneli InjectAMDGCNSharedMemTtrace/readWriteBC.cpp:24:26
17 _Z6kerneli InjectAMDGCNSharedMemTtrace/readWriteBC.cpp:24:26
Injected LDS Load/Store s_ttrace instructions at 18 source locations
```
We can see compiler has chosen to unroll, at least some part, of the loops in the kernel. Therefore, in this case, multiple s_ttracedata will be associated with the same source code line, but a different loop index. 

Looking at the instrumented-amdgcn-isa.log file we can see the desired inline ASM instructions inserted correctly, before each ds_read and ds_write instruction. As well as the nops needed for memory operations to complete. Additionally, counting the total number of s_ttracedata in the instrumented-amdgcn-isa.log will yield the same number, 18, as the number of indexes output from the pass.

### Instrumenting Only a Single Kernel
By default all AMDGPU kernels are instrumented during the LDS reads and writes Thread Trace pass. However if only a single kernel is of interest it can be selected through the instrument-amdgpu-function command line argument as follows.

```bash
hipcc --save-temps -c -fgpu-rdc -ggdb \
-fplugin=$PWD/build/lib/libInjectAMDGCNSharedMemTtrace.so \
-fpass-plugin=$PWD/build/lib/libInjectAMDGCNSharedMemTtrace.so \
-Xarch_device -mllvm=-instrument-amdgpu-function="_Z6kerneli" \
$PWD/InjectAMDGCNSharedMemTtrace/readWriteBC.cpp -o readWriteBC.o
hipcc --save-temps -fgpu-rdc readWriteBC.o -o instrumented
llvm-objdump -d a.out-hip-amdgcn-amd-amdhsa-gfx90a > instrumented-amdgcn-isa.log
```
if the instrument-amdgpu-function command line argument is left off or is an empty string the default, of all kernels being instrumented, is used.

# Example 4: Instrument Global Reads and Writes To Detect Uncoalesced Memory Accesses
In this case the example [MemCoalescingTests](tests/MemCoalescingTests.cpp) is built directly into the CMake file and includes some Googletest infrastructure. To exercise this example build with the tests on:

```bash
cmake -DCMAKE_C_COMPILER=hipcc -DCMAKE_CXX_COMPILER=hipcc \
-DBUILD_TESTING=ON -DLLVM_INSTALL_DIR=$ROCM_LLVM ..
```
the test executable will be located in ```build/bin``` and can be executed directly or through running ```make test``` in the build directory


# Example 5: Instrument Global Reads and Writes to Generate Memory Traces in Triton MLIR Based ML Compiler

## Install and build Triton
```bash
git clone https://github.com/triton-lang/triton.git
cd triton/python
python -m pip install -e .
```
## Install Pip Packages
```bash
pip install numpy==1.20.3
pip install matplotlib==3.4.3
pip install numba==0.54.1
pip install scipy==1.6.3
pip install pandas==1.2.4
```
## Install and Build instrument-amdgpu-kernels:
```bash
cd /var/lib/jenkins/
git clone https://github.com/CRobeck/instrument-amdgpu-kernels.git
cd instrument-amdgpu-kernels/instrumentation
```

```bash
# Triton PR #4638 introduced masked load/store IR operations. We don't support that yet.
git checkout 368c864e9a084296d887851fdd0974d3a17b78c4
# --offload-arch=gfx90a assumes MI250 or MI210
hipcc -mcode-object-version=4 -c --save-temps --offload-arch=gfx90a MemTraceInstrumentationKernel.cpp
cd ..
mkdir build && cd build
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_INSTALL_DIR=~/.triton/llvm/llvm-4713bd4c-ubuntu-x64/ .. <-this exact path hash will be different
cmake --build .
```
## Run Memory Trace Instrumentation With Triton vector-add tutorial
```bash
TRITON_ALWAYS_COMPILE=1 TRITON_DISABLE_LINE_INFO=0 AMDCGN_INSTRUMENTATION_FUNCTIONS_FILE=./instrument-amdgpu-kernels/instrumentation/MemTraceInstrumentationKernel-hip-amdgcn-amd-amdhsa-gfx90a.bc LLVM_PASS_PLUGIN_PATH=./instrument-amdgpu-kernels/build/lib/libAMDGCNMemTrace.so python ~/triton/python/tutorials/01-vector-add.py
```
