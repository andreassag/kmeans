cmake_minimum_required(VERSION 3.18)

project(kmeans VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(KMEANS_BUILD_TESTS "Build the standalone C library tests" OFF)

if(DEFINED SKBUILD)
  set(KMEANS_BUILD_PYTHON_DEFAULT ON)
else()
  set(KMEANS_BUILD_PYTHON_DEFAULT OFF)
endif()
option(KMEANS_BUILD_PYTHON "Build Python extension module" ${KMEANS_BUILD_PYTHON_DEFAULT})

include_directories(include)

find_library(MATH_LIBRARY m)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# Shared library for C API (optional, for non-Python usage)
add_library(${PROJECT_NAME}_lib SHARED src/2d.c src/3d.c)
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
set_target_properties(${PROJECT_NAME}_lib PROPERTIES
  C_STANDARD 11
  OUTPUT_NAME "kmeans"
)
if(NOT MSVC)
  target_compile_options(${PROJECT_NAME}_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(MATH_LIBRARY)
  target_link_libraries(${PROJECT_NAME}_lib PRIVATE ${MATH_LIBRARY})
endif()

# Tests for the standalone C library (cmake -DKMEANS_BUILD_TESTS=ON)
if(KMEANS_BUILD_TESTS)
  enable_testing()
  add_executable(test_kmeans_lib tests/c/test_kmeans_lib.c)
  target_link_libraries(test_kmeans_lib PRIVATE ${PROJECT_NAME}_lib)
  if(MATH_LIBRARY)
    target_link_libraries(test_kmeans_lib PRIVATE ${MATH_LIBRARY})
  endif()
  add_test(NAME test_kmeans_lib COMMAND test_kmeans_lib)
endif()

# Python extension module
if(KMEANS_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

  python_add_library(_kmeans MODULE
      src/kmeans/_kmeans.c
      WITH_SOABI
  )

  target_include_directories(_kmeans PRIVATE
      ${Python_NumPy_INCLUDE_DIRS}
      ${CMAKE_SOURCE_DIR}/include
  )

  target_link_libraries(_kmeans PRIVATE Python::NumPy)

  if(MATH_LIBRARY)
    target_link_libraries(_kmeans PRIVATE ${MATH_LIBRARY})
  endif()

  # Install the extension module
  install(TARGETS _kmeans DESTINATION kmeans)
endif()
