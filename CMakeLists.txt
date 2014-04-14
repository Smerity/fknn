cmake_minimum_required (VERSION 2.6)
project (fknn)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_executable(fknn knn.cpp)
