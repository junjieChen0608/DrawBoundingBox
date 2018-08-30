cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(draw)
set(CMAKE_CXX_STANDARD 11)
FIND_PACKAGE(PCL 1.5 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(draw main.cpp)
target_link_libraries (draw ${PCL_LIBRARIES})