cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(pcl_voxelgridcovar)
set(CMAKE_CXX_STANDARD 11)
FIND_PACKAGE(PCL 1.5 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pcl_voxelgridcovar main.cpp)
target_link_libraries (pcl_voxelgridcovar ${PCL_LIBRARIES})