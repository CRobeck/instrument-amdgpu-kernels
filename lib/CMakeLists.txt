set(AMDGCN_INSTRUMENTATION_PASSES
	  InjectAMDGCNFunction
    InjectAMDGCNInlineASM
    InjectAMDGCNSharedMemTtrace
    AMDGCNMemCoalescing
    AMDGCNMemTrace
    AMDGCNMemTraceHIP
    )


set(InjectAMDGCNFunction_SOURCES
  InjectAMDGCNFunction.cpp)

set(InjectAMDGCNInlineASM_SOURCES
  InjectAMDGCNInlineASM.cpp)

set(InjectAMDGCNSharedMemTtrace_SOURCES
    InjectAMDGCNSharedMemTtrace.cpp)

set(AMDGCNMemCoalescing_SOURCES
  AMDGCNMemCoalescing.cpp)

set(AMDGCNMemTrace_SOURCES
	AMDGCNMemTrace.cpp)

  set(AMDGCNMemTraceHIP_SOURCES
	AMDGCNMemTraceHip.cpp)


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

#    target_link_libraries(
#      ${plugin}
#      LLVMCore
#      LLVMIRReader
#      LLVMLinker
#      LLVMSupport
#      LLVMBitWriter
#      "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
#      )
endforeach()

