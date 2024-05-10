add_executable(MemCoalescing MemCoalescingTests.cpp)

set(InstrumentationFunction "_Z13numCacheLinesPvjjj")
set(InstrumentationFunctionFile ${PROJECT_SOURCE_DIR}/tests/InstrumentationFunctions-hip-amdgcn-amd-amdhsa-gfx90a.bc)
target_compile_options(MemCoalescing PRIVATE
-fgpu-rdc
-fplugin=${PROJECT_BINARY_DIR}/lib/libAMDGCNMemCoalescing.so
-fpass-plugin=${PROJECT_BINARY_DIR}/lib/libAMDGCNMemCoalescing.so
-Xarch_device -mllvm=-amdgcn-instrumentation-function=${InstrumentationFunction}
-Xarch_device -mllvm=-amdgcn-instrumentation-function-file=${InstrumentationFunctionFile}
)

target_include_directories(MemCoalescing PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(MemCoalescing gtest_main hip::device)
target_link_options(MemCoalescing PRIVATE -fgpu-rdc --hip-link)
target_compile_definitions(MemCoalescing PRIVATE ${LLVM_DEFINITIONS} ENABLE_TESTING)

add_test(NAME MemCoalescingTest COMMAND MemCoalescing)
