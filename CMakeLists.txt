cmake_minimum_required(VERSION 3.29)
set(CMAKE_CXX_STANDARD 14)

add_subdirectory(lib/depthai-core EXCLUDE_FROM_ALL)

project(luxonis_demo)

add_executable(luxonis_demo main.cpp)
target_link_libraries(luxonis_demo PRIVATE depthai::core)