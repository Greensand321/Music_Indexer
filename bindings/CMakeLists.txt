cmake_minimum_required(VERSION 3.15)
project(llama_bindings LANGUAGES C CXX)

# 1) Point CMake at your Conda Python environment
if(DEFINED ENV{CONDA_PREFIX})
  set(Python3_ROOT_DIR $ENV{CONDA_PREFIX})
endif()

# 2) Locate Python 3 interpreter and development headers/libraries
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 3) Pull in the official pybind11
add_subdirectory(pybind11)

# 4) Build llama.cpp as a static library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp llama_cpp_build)

# 5) Create the Python extension module
pybind11_add_module(llama_bindings
    llama_bindings.cpp
)

# 6) Ensure Python.h (and other headers) are found
target_include_directories(llama_bindings
  PRIVATE
    ${Python3_INCLUDE_DIRS}              # Python dev headers
    ${CMAKE_CURRENT_SOURCE_DIR}          # for llama_c_api_for_bindings.h
)

# 7) Link against llama.cpp static library and the Python library
target_link_libraries(llama_bindings
  PRIVATE
    llama               # the static llama.cpp library
    Python3::Python     # Python interpreter/embed library
)
