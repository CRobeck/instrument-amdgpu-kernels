cmake_minimum_required(VERSION 3.21)
project(InstrumentAMDGPUKernel LANGUAGES C CXX)

# Default ROCm path can be infered
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH})
else()
  set(ROCM_PATH "/opt/rocm")
endif()
message("ROCM_PATH: ${ROCM_PATH}")

# Define supported LLVM versions
set(TRITON_LLVM "" CACHE PATH "Path to the Triton LLVM directory")
set(ROCM_LLVM "${ROCM_PATH}/llvm" CACHE PATH "Path to the ROCm LLVM directory")

function(find_and_configure_llvm llvm_install_dir llvm_version)
  # Confirm that the LLVM installation directory exists
  set(LLVM_INCLUDE_DIR "${llvm_install_dir}/include/llvm")
  if(NOT EXISTS "${LLVM_INCLUDE_DIR}")
  message(FATAL_ERROR
    " llvm_install_dir (${LLVM_INCLUDE_DIR}) is invalid.")
  endif()

  # Check that the LLVMConfig.cmake file exists 
  set(- FALSE)

  # Ubuntu + Darwin
  if(EXISTS "${llvm_install_dir}/lib/cmake/llvm/LLVMConfig.cmake")
    set(VALID_INSTALLATION TRUE)
  endif()
  # Fedora
  if(EXISTS "${llvm_install_dir}/lib64/cmake/llvm/LLVMConfig.cmake")
    set(VALID_INSTALLATION TRUE)
  endif()
  if(NOT ${VALID_INSTALLATION})
    message(FATAL_ERROR
      "LLVM installation directory, (${llvm_install_dir}), is invalid. Couldn't
      find LLVMConfig.cmake.")
  endif()

  # find_package(LLVM CONFIG)
  include("${llvm_install_dir}/lib/cmake/llvm/LLVMConfig.cmake")

  if("${LLVM_VERSION_MAJOR}" VERSION_LESS 17)
    message(FATAL_ERROR "Found LLVM ${LLVM_VERSION_MAJOR}, but need LLVM 17 or above")
  endif()

  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${llvm_install_dir}")
  message(STATUS "Using LLVMConfig.cmake in: ${llvm_install_dir}")
  message("LLVM STATUS:
    Definitions ${LLVM_DEFINITIONS}
    Includes    ${LLVM_INCLUDE_DIRS}
    Libraries   ${LLVM_LIBRARY_DIRS}
    Targets     ${LLVM_TARGETS_TO_BUILD}"
  )

  # include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
  # link_directories(${LLVM_LIBRARY_DIRS})
  # add_definitions(${LLVM_DEFINITIONS})
  set(LLVM_INCLUDE_DIRS_${llvm_version} ${LLVM_INCLUDE_DIRS} PARENT_SCOPE)
  set(LLVM_LIBRARY_DIRS_${llvm_version} ${LLVM_LIBRARY_DIRS} PARENT_SCOPE)
  set(LLVM_DEFINITIONS_${llvm_version} ${LLVM_DEFINITIONS} PARENT_SCOPE)


endfunction()

# TODO: CONFIRM THIS LINE
list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/llvm/")


find_and_configure_llvm(${ROCM_LLVM} "ROCM")
if(TRITON_LLVM)
  find_and_configure_llvm(${TRITON_LLVM} "TRITON")
endif()

set(CMAKE_CXX_STANDARD 17 CACHE STRING "")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE
      STRING "Build type (default Debug):" FORCE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall \
  -DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING \
  -fdiagnostics-color=always")

if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fvisibility-inlines-hidden" SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG)
if (${SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG} EQUAL "1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
endif()

# Set the build directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
	cmake_policy(SET CMP0135 NEW)
endif()


# enable_testing()

add_subdirectory(lib)

option(BUILD_TESTING "Enable tests" OFF)
if (${BUILD_TESTING})
	include(FetchContent)
	FetchContent_Declare(
	  googletest
	  # Specify the commit you depend on and update it regularly.
	  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
	)
	FetchContent_MakeAvailable(googletest)

	include(CTest)
	enable_testing()

	find_package(hip REQUIRED)
	add_subdirectory(tests)
endif()

# Installation
install(TARGETS ${AMDGCN_INSTRUMENTATION_PLUGINS} 
        LIBRARY DESTINATION lib/${CMAKE_PROJECT_NAME}
        PUBLIC_HEADER DESTINATION include/${CMAKE_PROJECT_NAME})

