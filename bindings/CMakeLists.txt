cmake_minimum_required(VERSION 3.15)
project(llama_bindings LANGUAGES C CXX)

add_subdirectory(pybind11)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp/include
)

add_subdirectory(../third_party/llama.cpp llama_cpp_build)

pybind11_add_module(llama_bindings llama_bindings.cpp)
target_link_libraries(llama_bindings PRIVATE llama)
