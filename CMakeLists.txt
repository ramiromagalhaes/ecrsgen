cmake_minimum_required(VERSION 2.8)

project(ecrsgen)
aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

# The version number.
set (ecrsgen_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)

target_link_libraries (ecrsgen opencv_core opencv_imgproc opencv_highgui opencv_objdetect)
