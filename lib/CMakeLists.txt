set(AMDGCN_INSTRUMENTATION_PASSES
	InjectAMDGCNFunction
    InjectAMDGCNInlineASM
    InjectAMDGCNSharedMemTtrace
    AMDGCNNumCacheLines
    AMDGCNMemTrace
    AMDGCNMemTraceHip
    )

set(InjectAMDGCNFunction_SOURCES
  InjectAMDGCNFunction.cpp)

set(InjectAMDGCNInlineASM_SOURCES
  InjectAMDGCNInlineASM.cpp)

set(InjectAMDGCNSharedMemTtrace_SOURCES
    InjectAMDGCNSharedMemTtrace.cpp)

set(AMDGCNNumCacheLines_SOURCES
    AMDGCNNumCacheLines.cpp)

set(AMDGCNMemTrace_SOURCES
	AMDGCNMemTrace.cpp)

set(AMDGCNMemTraceHip_SOURCES
	AMDGCNMemTraceHip.cpp)

set(AMDGCN_INSTRUMENTATION_PLUGINS "")

function(link_against_llvm llvm_version)
  string(TOLOWER ${llvm_version} install_suffix)
  foreach( plugin ${AMDGCN_INSTRUMENTATION_PASSES} )
    add_library(
      ${plugin}-${install_suffix}
      SHARED
      ${${plugin}_SOURCES}
    )

    target_include_directories(
      ${plugin}-${install_suffix}
      PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/../include"
      ${LLVM_INCLUDE_DIRS_${llvm_version}}
    )

    if(${llvm_version} STREQUAL "TRITON")
      target_link_directories(
        ${plugin}-${install_suffix}
        PRIVATE
        ${LLVM_LIBRARY_DIRS_${llvm_version}}
      )  

      target_link_libraries(
        ${plugin}-${install_suffix}
        PRIVATE
        LLVMCore
        LLVMIRReader
        LLVMLinker
        "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
      )
    endif()


    list(APPEND AMDGCN_INSTRUMENTATION_PLUGINS ${plugin}-${install_suffix})
  endforeach()
endfunction()

link_against_llvm("ROCM")
link_against_llvm("TRITON")

set(AMDGCN_INSTRUMENTATION_PLUGINS "${AMDGCN_INSTRUMENTATION_PLUGINS}" PARENT_SCOPE)
