cmake_minimum_required(VERSION 3.16)
project(ann_bruteforce LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ANN_CPP_DIR ${CMAKE_SOURCE_DIR}/ann/cpp)
set(HARNESS_CPP_DIR ${CMAKE_SOURCE_DIR}/harness/cpp)

add_executable(demo examples/main.cpp)
target_include_directories(demo PRIVATE ${ANN_CPP_DIR})

add_executable(run_script harness/cpp/run_script.cpp)
target_include_directories(run_script PRIVATE ${ANN_CPP_DIR} ${HARNESS_CPP_DIR})
