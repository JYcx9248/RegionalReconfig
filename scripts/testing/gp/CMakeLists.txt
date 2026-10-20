# gp_kaminpar, the partitioning step of make_gp_assignment.py, built against KaMinPar at the commit
# the paper's code pins (gp-ann external/KaMinPar = a9e4b1f6d92b4110d4d4a31ee2e7dee5a0761b05), with
# the options the paper's code sets. KaMinPar needs oneTBB (libtbb-dev) and its kassert submodule.
#
#   git clone https://github.com/KaHIP/KaMinPar && git -C KaMinPar checkout a9e4b1f6 \
#     && git -C KaMinPar submodule update --init external/kassert
#   cmake -S scripts/testing/gp -B /tmp/gp-build -DKAMINPAR_SRC=$PWD/KaMinPar -DCMAKE_BUILD_TYPE=Release
#   cmake --build /tmp/gp-build -j
cmake_minimum_required(VERSION 3.16)
project(gp_kaminpar CXX)

set(KAMINPAR_SRC "" CACHE PATH "KaMinPar source tree (commit a9e4b1f6)")
if (NOT KAMINPAR_SRC)
  message(FATAL_ERROR "set -DKAMINPAR_SRC=<KaMinPar source tree>")
endif ()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# As in gp-ann's CMakeLists.txt; tests and apps are not needed here.
set(KAMINPAR_64BIT_WEIGHTS ON CACHE BOOL "" FORCE)
set(KAMINPAR_64BIT_EDGE_IDS ON CACHE BOOL "" FORCE)
set(KAMINPAR_ENABLE_TIMERS OFF CACHE BOOL "" FORCE)
set(KAMINPAR_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KAMINPAR_BUILD_APPS OFF CACHE BOOL "" FORCE)
add_subdirectory(${KAMINPAR_SRC} kaminpar EXCLUDE_FROM_ALL)

add_executable(gp_kaminpar gp_kaminpar.cpp)
target_link_libraries(gp_kaminpar PRIVATE kaminpar_shm)
