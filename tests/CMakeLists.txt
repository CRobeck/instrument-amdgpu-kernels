add_library(InstrumentationFunctions STATIC ../instrumentation/InstrumentationFunctions.cpp)
target_link_libraries(InstrumentationFunctions PRIVATE hip::device)
target_compile_options(InstrumentationFunctions PRIVATE
-fgpu-rdc
)
target_compile_definitions(InstrumentationFunctions PUBLIC $<$<BOOL:${BUILD_TESTING}>:BUILD_TESTING>)

add_executable(NumCacheLines NumCacheLinesTests.cpp)

target_compile_options(NumCacheLines PRIVATE
-fgpu-rdc
-fplugin=${PROJECT_BINARY_DIR}/lib/libAMDGCNNumCacheLines.so
-fpass-plugin=${PROJECT_BINARY_DIR}/lib/libAMDGCNNumCacheLines.so
)

target_include_directories(NumCacheLines PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(NumCacheLines InstrumentationFunctions gtest_main hip::device)
target_link_options(NumCacheLines PRIVATE -fgpu-rdc --hip-link)
target_compile_definitions(NumCacheLines PRIVATE ${LLVM_DEFINITIONS} ENABLE_TESTING)

add_test(NAME NumCacheLinesTest COMMAND NumCacheLines)
