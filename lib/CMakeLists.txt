set(AMDGCN_INSTRUMENTATION_PASSES
	  InjectAMDGCNFunction
    InjectAMDGCNInlineASM
    DumpAMDGCNRegisters
    )

set(InjectAMDGCNFunction_SOURCES
  InjectAMDGCNFunction.cpp)

set(InjectAMDGCNInlineASM_SOURCES
  InjectAMDGCNInlineASM.cpp)  

set(DumpAMDGCNRegisters_SOURCES
    DumpAMDGCNRegisters.cpp)


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
      "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
      )
endforeach()
