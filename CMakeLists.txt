cmake_minimum_required(VERSION 3.13)

project(BitVecEngine)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wno-psabi -Wno-sign-compare")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info")

IF (NOT OPTIMIZATION_LEVEL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -mfma")
  add_definitions(-DVAQ_OPTIMIZE)
  add_definitions(-march=haswell)
  add_definitions(-mavx -mavx2)
ELSEIF (OPTIMIZATION_LEVEL STREQUAL "full")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
ELSEIF (OPTIMIZATION_LEVEL STREQUAL "aggressive")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
ELSEIF (OPTIMIZATION_LEVEL STREQUAL "generic")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
ENDIF()

include_directories(external/eigen)

add_subdirectory(bitvecengine)
add_subdirectory(examples)

IF (BUILD_TESTING)
  add_subdirectory(test)
  add_custom_target(test COMMAND Test --durations yes)
ENDIF(BUILD_TESTING)

add_definitions(-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)
add_definitions(-DEIGEN_DONT_PARALLELIZE)

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()
