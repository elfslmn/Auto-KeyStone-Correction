cmake_minimum_required(VERSION 2.8)
set(PROJECT_NAME "keystone" )

project (${PROJECT_NAME})

set(CMAKE_MODULE_PATH "/opt/opencv331/opencv")
set(ROYALE_DIR "/home/user/libroyale-3.11.0.42-LINUX-x86-64Bit")
set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" )
set(SRC_DIR "${PROJECT_SOURCE_DIR}/src" )

#Compilers
include( CheckCXXCompilerFlag )
CHECK_CXX_COMPILER_FLAG( "-std=c++11" COMPILER_SUPPORTS_CXX11 )
CHECK_CXX_COMPILER_FLAG( "-std=c++0x" COMPILER_SUPPORTS_CXX0X )

if( COMPILER_SUPPORTS_CXX11 )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
elseif( COMPILER_SUPPORTS_CXX0X )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x" )
else()
    message( STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler." )
endif()
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w" ) # to surpress warnings

#OpenCv
find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
  message("OpenCV example will not be build as no OpenCV was found!")
  return()
endif()

#Royale
set(CMAKE_PREFIX_PATH "${ROYALE_DIR}/share")
find_package(royale REQUIRED)
link_directories(${royale_LIB_DIR})

#Boost
set( Boost_USE_STATIC_LIBS ON )
set( Boost_USE_STATIC ON )
find_package(Boost 1.58.0 COMPONENTS filesystem REQUIRED)

# Includes
include_directories(
  ${INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

# Project Files
set(
  HEADERS
  ${INCLUDE_DIR}/CamListener.h
  ${INCLUDE_DIR}/FrameObject.h
  ${INCLUDE_DIR}/DetectionParams.h
  ${INCLUDE_DIR}/FramePlane.h
  ${INCLUDE_DIR}/Util.h		
  ${INCLUDE_DIR}/Detector.h
  ${INCLUDE_DIR}/PlaneDetector.h
  ${INCLUDE_DIR}/Version.h
  ${INCLUDE_DIR}/stdafx.h
)

set(
  SOURCES 
  ${SRC_DIR}/CamListener.cpp
  ${SRC_DIR}/Detector.cpp
  ${SRC_DIR}/PlaneDetector.cpp
  ${SRC_DIR}/Util.cpp
  ${SRC_DIR}/FrameObject.cpp
  ${SRC_DIR}/FramePlane.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES} main.cpp)
target_link_libraries(${PROJECT_NAME} "${royale_LIBS}" "${OpenCV_LIBRARIES}" "${Boost_LIBRARIES}") 
COPY_ROYALE_LIBS(${PROJECT_NAME}) 
 



