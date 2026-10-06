cmake_minimum_required(VERSION 3.18)
project(rtier_engine LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FUSION_NATIVE "Compile host code with -march=native" ON)
option(FUSION_BUILD_TESTS "Build unit tests" ON)
set(FUSION_CUDA "AUTO" CACHE STRING "Build the GPU filtering backend: AUTO, ON or OFF")
set_property(CACHE FUSION_CUDA PROPERTY STRINGS AUTO ON OFF)

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# liburing is optional: without it the engine uses Linux AIO or pread.
find_path(LIBURING_INCLUDE_DIR liburing.h)
find_library(LIBURING_LIBRARY uring)

# ---------------------------------------------------------------------------
# CUDA (optional)
# ---------------------------------------------------------------------------
set(FUSION_WITH_CUDA OFF)
if(NOT FUSION_CUDA STREQUAL "OFF")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
      else()
        set(CMAKE_CUDA_ARCHITECTURES 70 80)
      endif()
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(FUSION_WITH_CUDA ON)
  elseif(FUSION_CUDA STREQUAL "ON")
    message(FATAL_ERROR "FUSION_CUDA=ON but no CUDA compiler was found")
  endif()
endif()

message(STATUS "FusionANNS: CUDA backend = ${FUSION_WITH_CUDA}")
if(LIBURING_INCLUDE_DIR AND LIBURING_LIBRARY)
  message(STATUS "FusionANNS: io_uring backend = ON (${LIBURING_LIBRARY})")
else()
  message(STATUS "FusionANNS: io_uring backend = OFF (liburing not found; AIO/pread only)")
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(fusion STATIC
  src/common.cpp
  src/dataset.cpp
  src/kmeans.cpp
  src/pq.cpp
  src/index_meta.cpp
  src/clustering.cpp
  src/assignment.cpp
  src/layout.cpp
  src/builder.cpp
  src/page_reader.cpp
  src/cpu_filter.cpp
  src/rerank.cpp
  src/engine.cpp
  src/partition.cpp
  src/node_engine.cpp
  src/pq_store.cpp
  src/raw_store.cpp
  src/read_budget.cpp
)
target_include_directories(fusion PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# Vendored hnswlib: SYSTEM so its warnings do not drown ours.
target_include_directories(fusion SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(fusion PUBLIC OpenMP::OpenMP_CXX Threads::Threads)
target_compile_options(fusion PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
if(FUSION_NATIVE)
  target_compile_options(fusion PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()

if(LIBURING_INCLUDE_DIR AND LIBURING_LIBRARY)
  target_include_directories(fusion PRIVATE ${LIBURING_INCLUDE_DIR})
  target_link_libraries(fusion PUBLIC ${LIBURING_LIBRARY})
  target_compile_definitions(fusion PUBLIC FUSION_HAVE_LIBURING=1)
endif()

if(FUSION_WITH_CUDA)
  target_sources(fusion PRIVATE src/gpu_filter.cu)
  target_compile_definitions(fusion PUBLIC FUSION_WITH_CUDA=1)
  target_link_libraries(fusion PUBLIC CUDA::cudart)
  target_compile_options(fusion PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>)
endif()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
foreach(tool fusion_build fusion_search fusion_gt fusion_selftest rtier_segment)
  add_executable(${tool} tools/${tool}.cpp)
  target_link_libraries(${tool} PRIVATE fusion)
  target_compile_options(${tool} PRIVATE -Wall -Wextra)
endforeach()

# ---------------------------------------------------------------------------
# rtier data-node service (wire protocol server around fusion::NodeEngine)
# ---------------------------------------------------------------------------
add_library(rtier_node_lib STATIC node/server.cpp)
target_link_libraries(rtier_node_lib PUBLIC fusion)
target_compile_options(rtier_node_lib PRIVATE -Wall -Wextra)

add_executable(rtier_node tools/rtier_node.cpp)
target_link_libraries(rtier_node PRIVATE rtier_node_lib)
target_compile_options(rtier_node PRIVATE -Wall -Wextra)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(FUSION_BUILD_TESTS)
  enable_testing()
  add_executable(fusion_tests tests/test_main.cpp)
  target_link_libraries(fusion_tests PRIVATE fusion)
  add_test(NAME fusion_tests COMMAND fusion_tests)
  add_executable(rtier_node_tests tests/test_node.cpp)
  target_link_libraries(rtier_node_tests PRIVATE rtier_node_lib)
  add_test(NAME rtier_node_tests COMMAND rtier_node_tests)
endif()
