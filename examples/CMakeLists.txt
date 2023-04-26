add_definitions(-DEIGEN_DONT_PARALLELIZE)

include_directories(${PROJECT_SOURCE_DIR}/bitvecengine)
include_directories(${PROJECT_SOURCE_DIR}/external/eigen)

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

# link_directories("/CUSTOM_LIB_DIR/usr/lib")

add_executable(demo_vaq "demo_vaq.cpp")
target_link_libraries(demo_vaq PRIVATE VAQ)
target_link_libraries(demo_vaq PRIVATE FPGrowth)
target_link_libraries(demo_vaq PRIVATE -lglpk)
target_link_libraries(demo_vaq PRIVATE -lblas -llapacke)
target_link_libraries(demo_vaq PRIVATE -larmadillo)

# add_executable(demo_vaq_paramsearch "demo_vaq_paramsearch.cpp")
# target_link_libraries(demo_vaq_paramsearch PRIVATE VAQ)
# target_link_libraries(demo_vaq_paramsearch PRIVATE BitVecEngine)
# target_link_libraries(demo_vaq_paramsearch PRIVATE -lglpk)

# add_executable(demo_bitvecengine "demo_bitvecengine.cpp")
# target_link_libraries(demo_bitvecengine PRIVATE BitVecEngine)

# add_executable(demo_binary "demo_binary.cpp")
# target_link_libraries(demo_binary PRIVATE BitVecEngine)
# target_link_libraries(demo_binary PRIVATE -lglpk)

# add_executable(demo_cluster "demo_cluster.cpp")

# add_executable(demo_boltcent "demo_boltcentroids.cpp")

# add_executable(test_func "test_func.cpp")
# target_link_libraries(test_func PRIVATE FPGrowth)
# target_link_libraries(test_func PRIVATE -lblas -llapacke)

