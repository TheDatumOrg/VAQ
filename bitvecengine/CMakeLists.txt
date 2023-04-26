# add_library(
#   BitVecEngine SHARED
#   BitVecEngine.cpp
# )

add_library(
  VAQ SHARED
  utils/Heap.cpp
  VAQ.cpp
)

# Utils library
add_library(
  FPGrowth SHARED
  "utils/FPGrowth/fptree.cpp"
)

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

