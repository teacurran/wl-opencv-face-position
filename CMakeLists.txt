cmake_minimum_required(VERSION 2.6)
project(wlst-ofp)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

SET(CMAKE_BUILD_TYPE testing)
SET(CMAKE_CXX_FLAGS_TESTING "-g -O0")
SET(CMAKE_C_FLAGS_TESTING "-g -O0")

SET(CMAKE_VERBOSE_MAKEFILE ON)

# TODO: get this to find OpenCV properly.
#find_package(CV REQUIRED)
#include_directories(${CV_INCLUDE_DIRS})
#set(LIBS ${LIBS} ${CV_LIBRARIES})

find_library(CV opencv_core)
find_library(CV_HIGHGUI opencv_highgui)
find_library(CV_IMGPROC opencv_imgproc)
find_library(CV_OBJDETECT opencv_objdetect)

include_directories(src)
#include_directories("${CMAKE_SOURCE_DIR}/opencv/include/opencv/")
#include_directories("${CMAKE_SOURCE_DIR}/opencv/src/cv")
include_directories("/usr/local/include/opencv/")

file(GLOB lib_srcs "src/*.cpp")
file(GLOB lib_hdrs "src/*.hpp")

add_executable(wlst-ofp ${lib_srcs} ${lib_hdrs})

LINK_DIRECTORIES(${CV_LIB_DIR} ${CV_HIGHGUI_LIB_DIR} ${CV_IMGPROC_LIB_DIR} ${CV_OBJDETECT_LIB_DIR})

TARGET_LINK_LIBRARIES(wlst-ofp opencv_core opencv_highgui opencv_imgproc opencv_objdetect)

