cmake_minimum_required(VERSION 2.8)
include(FetchContent)
project(autoquilt)
find_package(OpenCV REQUIRED)
find_package(CGAL REQUIRED)
# blossom5 source code redistribution prohibited, so we instead fetch it at build
FetchContent_Declare(blossom5
    URL https://pub.ist.ac.at/~vnk/software/blossom5-v2.05.src.tar.gz
    PREFIX blossom5
)
FetchContent_MakeAvailable(blossom5)
include_directories(${blossom5_SOURCE_DIR})
add_executable(autoquilt
    ${blossom5_SOURCE_DIR}/misc.cpp
    ${blossom5_SOURCE_DIR}/PMduals.cpp
    ${blossom5_SOURCE_DIR}/PMexpand.cpp
    ${blossom5_SOURCE_DIR}/PMinit.cpp
    ${blossom5_SOURCE_DIR}/PMinterface.cpp
    ${blossom5_SOURCE_DIR}/PMmain.cpp
    ${blossom5_SOURCE_DIR}/PMrepair.cpp
    ${blossom5_SOURCE_DIR}/PMshrink.cpp
    ${blossom5_SOURCE_DIR}/GEOM/GPMinit.cpp
    ${blossom5_SOURCE_DIR}/GEOM/GPMinterface.cpp
    ${blossom5_SOURCE_DIR}/GEOM/GPMkdtree.cpp
    ${blossom5_SOURCE_DIR}/GEOM/GPMmain.cpp
    ${blossom5_SOURCE_DIR}/MinCost/MinCost.cpp
    main.cpp
)
target_link_libraries(autoquilt ${OpenCV_LIBS} CGAL::CGAL)
