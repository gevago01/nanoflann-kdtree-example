cmake_minimum_required(VERSION 3.6)
project(NanoFlannKDTree)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -std=c++11")

set(SOURCE_FILES main.cpp KDTreeVectorOfVectorsAdaptor.h nanoflann.h)
add_executable(NanoFlannKDTree ${SOURCE_FILES})