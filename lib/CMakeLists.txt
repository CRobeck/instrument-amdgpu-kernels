set(AMDGCN_INSTRUMENTATION_PASSES
	InjectAMDGCNFunction
    InjectAMDGCNInlineASM
    InjectAMDGCNSharedMemTtrace
    AMDGCNNumCacheLines
    AMDGCNMemTrace
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


set(AMDGCN_INSTRUMENTATION_PLUGINS "")

foreach( plugin ${AMDGCN_INSTRUMENTATION_PASSES} )
  add_library(
    ${plugin}
    SHARED
    ${${plugin}_SOURCES}
  )

  target_include_directories(
    ${plugin}
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
  )

  target_link_libraries(
    ${plugin}
    PRIVATE
    LLVMCore
    LLVMIRReader
    LLVMLinker
    "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
    )

  list(APPEND AMDGCN_INSTRUMENTATION_PLUGINS ${plugin})
endforeach()

set(AMDGCN_INSTRUMENTATION_PLUGINS "${AMDGCN_INSTRUMENTATION_PLUGINS}" PARENT_SCOPE)
