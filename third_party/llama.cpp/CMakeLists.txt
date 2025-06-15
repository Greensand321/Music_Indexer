cmake_minimum_required(VERSION 3.15)
project(llama C CXX)

add_library(llama STATIC llama_stub.cpp)

target_include_directories(llama PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
