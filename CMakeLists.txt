cmake_minimum_required(VERSION 3.21)
project(InstrumentAMDGPUKernel LANGUAGES C CXX)


set(LLVM_INSTALL_DIR "" CACHE PATH "LLVM installation directory")

set(LLVM_INCLUDE_DIR "${LLVM_INSTALL_DIR}/include/llvm")
if(NOT EXISTS "${LLVM_INCLUDE_DIR}")
message(FATAL_ERROR
  " LLVM_INSTALL_DIR (${LLVM_INCLUDE_DIR}) is invalid.")
endif()

# Check that the LLVMConfig.cmake file exists 
set(- FALSE)

# Ubuntu + Darwin
if(EXISTS "${LLVM_INSTALL_DIR}/lib/cmake/llvm/LLVMConfig.cmake")
  set(VALID_INSTALLATION TRUE)
endif()

# Fedora
if(EXISTS "${LLVM_INSTALL_DIR}/lib64/cmake/llvm/LLVMConfig.cmake")
  set(VALID_INSTALLATION TRUE)
endif()

if(NOT ${VALID_INSTALLATION})
  message(FATAL_ERROR
    "LLVM installation directory, (${LLVM_INSTALL_DIR}), is invalid. Couldn't
    find LLVMConfig.cmake.")
endif()

list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/llvm/")

find_package(LLVM CONFIG)

if("${LLVM_VERSION_MAJOR}" VERSION_LESS 17)
  message(FATAL_ERROR "Found LLVM ${LLVM_VERSION_MAJOR}, but need LLVM 17 or above")
endif()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_INSTALL_DIR}")

message("LLVM STATUS:
  Definitions ${LLVM_DEFINITIONS}
  Includes    ${LLVM_INCLUDE_DIRS}
  Libraries   ${LLVM_LIBRARY_DIRS}
  Targets     ${LLVM_TARGETS_TO_BUILD}"
)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

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
if (SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG)
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

