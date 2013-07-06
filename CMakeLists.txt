cmake_minimum_required(VERSION 2.8)

project(generators)

# The version number.
# set (ecrsgen_VERSION_MAJOR 1)
# set (ecrsgen_VERSION_MINOR 0)

# Debug and release configurations
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -D_DEBUG -Wextra -Wall")
set(CMAKE_CSS_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
#set(RELATIVE yes)

# Builds the common objects of my Haar wavelet library
file( GLOB LIB_SOURCES lib/*.cpp )
file( GLOB LIB_HEADERS lib/*.h )
add_library( haarwavelet ${LIB_SOURCES} ${LIB_HEADERS} )


# Here is the first executable
add_executable( haargen "haargen.cpp" )
target_link_libraries(haargen haarwavelet opencv_core opencv_imgproc opencv_highgui opencv_objdetect)

# Here is the other one executable
add_executable( ecrsgen "ecrsgen.cpp" )
target_link_libraries(ecrsgen haarwavelet opencv_core opencv_imgproc opencv_highgui opencv_objdetect)
