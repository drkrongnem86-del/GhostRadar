# CMakeLists for Ghost Radar v2.0+ native preprocessing
# YUV_420_888 -> RGB letterbox -> Float32 NCHW (for YOLOv8 input)
cmake_minimum_required(VERSION 3.22.1)
project(ghost_radar_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Native shared library
add_library(ghost_radar_native SHARED
    yuv_processor.cpp
)

# Optimization flags
target_compile_options(ghost_radar_native PRIVATE
    -O3
    -ffast-math
    -fno-exceptions
    -fno-rtti
    -Wall
    -Wno-unused-function
)

# Android log lib
find_library(log-lib log)
target_link_libraries(ghost_radar_native ${log-lib})
