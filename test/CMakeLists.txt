include_directories(${BitVecEngine_SOURCE_DIR}/external)
add_definitions(-march=native)
add_definitions(-mavx -mavx2)

file(GLOB test_files "test-*.cpp")
add_executable(Test ${test_files})
target_link_libraries(Test BitVecEngine)
