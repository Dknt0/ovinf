cmake_minimum_required(VERSION 3.20.0)
project(ovinf)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If march=native is turned on, you should add the same flag to the root project.
# Otherwise, you may get memory double free error.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/home/dknt/Library/openvino-2025.0.0/runtime/cmake")
# set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/home/dknt/Library/openvino_u24_2025.1.0/runtime/cmake")

find_package(OpenVINO REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/utils SRC_UTILS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/ovinf SRC_OVINF)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(ovinf SHARED
  ${SRC_UTILS}
  ${SRC_OVINF}
)
target_include_directories(ovinf PUBLIC
  ${OpenVINO_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIRS}
  ${yaml-cpp_INCLUDE_DIRS}
)
target_link_libraries(ovinf
  openvino::runtime
  yaml-cpp
  readerwriterqueue

  ${OpenCV_LIBS}
)

set(BUILD_TEST OFF CACHE BOOL "Build test executables")
if (BUILD_TEST) 
  add_subdirectory(test)
endif()

