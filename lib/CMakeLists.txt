set(AMDGCN_INSTRUMENTATION_PASSES
	InjectAMDGCNFunction
    InjectAMDGCNInlineASM
    InjectAMDGCNSharedMemTtrace
    AMDGCNNumCacheLines
    AMDGCNMemTrace
    AMDGCNSubmitAddressMessages
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

set(AMDGCNSubmitAddressMessages_SOURCES
	AMDGCNSubmitAddressMessages.cpp)

set(AMDGCN_INSTRUMENTATION_PLUGINS "")

set(install_suffix "rocm")
if ("${LLVM_INSTALL_DIR}" MATCHES ".*triton.*")
  set(install_suffix "triton")
endif()


foreach( plugin ${AMDGCN_INSTRUMENTATION_PASSES} )
  add_library(
    ${plugin}-${install_suffix}
    SHARED
    ${${plugin}_SOURCES}
  )

  target_include_directories(
    ${plugin}-${install_suffix}
    BEFORE
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
  )

 
#   if (${install_suffix} STREQUAL "triton")
#     target_link_libraries(
#       ${plugin}-${install_suffix}
#       PRIVATE
#       LLVMCore
#       LLVMIRReader
#       LLVMLinker
#       "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
#     )
#   endif()
# endforeach()


set(AMDGCN_INSTRUMENTATION_PLUGINS "${AMDGCN_INSTRUMENTATION_PLUGINS}" PARENT_SCOPE)
