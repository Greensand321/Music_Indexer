cmake_minimum_required(VERSION 3.15)
project(pybind11_headers LANGUAGES C CXX)

add_library(pybind11_headers INTERFACE)

target_include_directories(pybind11_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Provide pybind11_add_module macro stub
include(CheckLanguage)
macro(pybind11_add_module target)
    add_library(${target} MODULE ${ARGN})
endmacro()
