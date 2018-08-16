cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(pcl_voxelgridcovar)
set(CMAKE_CXX_STANDARD 11)
FIND_PACKAGE(PCL 1.5 REQUIRED)
FIND_PACKAGE(octomap REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${OCTOMAP_INCLUDE_DIRS})

add_executable(pcl_voxelgridcovar main.cpp)
target_link_libraries (pcl_voxelgridcovar ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})