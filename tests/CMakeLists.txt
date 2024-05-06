add_library(InstrumentationFunctions STATIC ../InstrumentationFunctions.cpp)
target_link_libraries(InstrumentationFunctions PRIVATE hip::device)
target_compile_options(InstrumentationFunctions PRIVATE
-fgpu-rdc
)
target_compile_definitions(InstrumentationFunctions PUBLIC $<$<BOOL:${BUILD_TESTING}>:BUILD_TESTING>)

add_executable(MemCoalescing MemCoalescingTest.cpp)

set(InstrumentationFunction "_Z13numCacheLinesPvjjj")

target_compile_options(MemCoalescing PRIVATE
-fgpu-rdc
-fplugin=${PROJECT_BINARY_DIR}/lib/libAMDGCNMemCoalescing.so
-fpass-plugin=${PROJECT_BINARY_DIR}/lib/libAMDGCNMemCoalescing.so
-Xarch_device -mllvm=-amdgcn-instrumentation-function=${InstrumentationFunction}
-Xarch_device -mllvm=-amdgcn-instrumentation-point="global loads, global stores"
)

target_include_directories(MemCoalescing PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(MemCoalescing InstrumentationFunctions gtest_main hip::device)
target_link_options(MemCoalescing PRIVATE -fgpu-rdc --hip-link)
target_compile_definitions(MemCoalescing PRIVATE ${LLVM_DEFINITIONS} ENABLE_TESTING)

add_test(NAME MemCoalescingTest COMMAND MemCoalescing)
