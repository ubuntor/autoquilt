include(ExternalProject)
cmake_minimum_required(VERSION 2.8)
project(autoquilt)
find_package(OpenCV REQUIRED)
find_package(CGAL REQUIRED)
add_executable(autoquilt main.cpp)

# TODO: windows support?
set(BLOSSOM5_MAKE make)
ExternalProject_Add(blossom5
    URL https://pub.ist.ac.at/~vnk/software/blossom5-v2.05.src.tar.gz
    PREFIX blossom5
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${BLOSSOM_MAKE}
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
)
ExternalProject_Get_Property(blossom5 SOURCE_DIR)
include_directories(${SOURCE_DIR})
set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES} ${SOURCE_DIR})
add_dependencies(autoquilt blossom5)

target_link_libraries(autoquilt ${OpenCV_LIBS} CGAL::CGAL)
