cmake_minimum_required(VERSION 2.8)

project(generators)

# The version number.
# set (ecrsgen_VERSION_MAJOR 1)
# set (ecrsgen_VERSION_MINOR 0)

# Debug and release configurations
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -D_DEBUG -Wextra -Wall")
set(CMAKE_CSS_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

if(WIN32)
    # For OpenCV installation instructions visit http://docs.opencv.org/doc/tutorials/introduction/windows_install/windows_install.html#windowssetpathandenviromentvariable
    # For Boost C++ libraries installation instructions visit http://andres.jaimes.net/718/how-to-install-the-c-boost-libraries-on-windows/
    set(BOOST_ROOT "C:/boost")
    set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "C:/opencv_src/build/include;C:/boost/lib")
    set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "C:/opencv_src/build/x86/mingw/lib")
endif(WIN32)



# More on find_package command http://www.cmake.org/Wiki/CMake:How_To_Find_Libraries
find_package( OpenCV REQUIRED COMPONENTS core imgproc highgui objdetect )
find_package( Boost REQUIRED COMPONENTS filesystem system serialization )
#set(Boost_USE_STATIC_LIBS ON)
#set(Boost_USE_MULTITHREADED ON)
message( "OpenCV libraries ${OpenCV_LIBS}" )
message( "OpenCV includes ${OpenCV_INCLUDE_DIRS}" )
message( "Boost libraries ${Boost_LIBRARIES}" )
message( "Boost includes ${Boost_INCLUDE_DIRS}" )



# Builds the common objects of my Haar wavelet library
file( GLOB LIB_SOURCES lib/*.cpp )
file( GLOB LIB_HEADERS lib/*.h )
add_library( haarwavelet ${LIB_SOURCES} ${LIB_HEADERS} )

# A simple test program
add_executable( test "test.cpp" )
target_link_libraries( test haarwavelet ${OpenCV_LIBS} )

# The Haar wavelet generator
add_executable( haargen "haargen.cpp" )
target_link_libraries( haargen haarwavelet ${OpenCV_LIBS} )

# The SRFS generator
add_executable( ecrsgen "ecrsgen.cpp" )
target_link_libraries( ecrsgen haarwavelet ${OpenCV_LIBS} ${Boost_LIBRARIES} )
