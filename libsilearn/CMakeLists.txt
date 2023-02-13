cmake_minimum_required(VERSION 3.15)
project(SI)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O2 -fPIC")


option(WIN_BUILD "build for windows" OFF)
if (WIN_BUILD)
 add_definitions(-DWIN_BUILD)
endif()

add_library(SI SHARED include/silearn
        src/si.cpp src/si_SL.cpp src/c_extern.cpp)
