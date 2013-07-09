cmake_minimum_required(VERSION 2.8)

project(generators)

if(WIN32)
    # For OpenCV installation instructions visit http://docs.opencv.org/doc/tutorials/introduction/windows_install/windows_install.html#windowssetpathandenviromentvariable
    # For Boost C++ libraries installation instructions visit http://andres.jaimes.net/718/how-to-install-the-c-boost-libraries-on-windows/
    set(BOOST_ROOT "C:/boost")
    set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "C:/opencv_src/build/include;C:/boost/lib")
    set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "C:/opencv_src/build/x86/mingw/lib")
endif(WIN32)

if(UNIX)
    # Nothing to do here... yet.
endif(UNIX)

find_package( OpenCV REQUIRED opencv_core opencv_imgproc opencv_highgui opencv_objdetect )
find_package( Boost REQUIRED )
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
    add_definitions( "-DHAS_BOOST" )
endif()

# Debug and release configurations
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -D_DEBUG -Wextra -Wall")
set(CMAKE_CSS_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

# Builds the common objects of my Haar wavelet library
file( GLOB LIB_SOURCES lib/*.cpp )
file( GLOB LIB_HEADERS lib/*.h )
add_library( haarwavelet ${LIB_SOURCES} ${LIB_HEADERS} )

# A simple test program
add_executable( test "test.cpp" )
target_link_libraries(test haarwavelet opencv_core opencv_imgproc opencv_highgui opencv_objdetect)

# The Haar wavelet generator
add_executable( haargen "haargen.cpp" )
target_link_libraries(haargen haarwavelet opencv_core opencv_imgproc opencv_highgui opencv_objdetect)

# The SRFS generator
add_executable( ecrsgen "ecrsgen.cpp" )
if(UNIX)
    target_link_libraries(ecrsgen haarwavelet boost_filesystem boost_system boost_serialization)
endif(UNIX)
if(WIN32)
    target_link_libraries(ecrsgen haarwavelet libboost_filesystem-mgw46-mt-d-1_54.a libboost_system-mgw46-mt-d-1_54.a libboost_serialization-mgw46-mt-d-1_54.a)
endif(WIN32)

# The version number.
# set (test_VERSION_MAJOR 1)
# set (test_VERSION_MINOR 0)
# set (haargen_VERSION_MAJOR 1)
# set (haargen_VERSION_MINOR 0)
# set (ecrsgen_VERSION_MAJOR 1)
# set (ecrsgen_VERSION_MINOR 0)
