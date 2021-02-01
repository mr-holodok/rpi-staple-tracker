cmake_minimum_required(VERSION 3.8)
project(staple)

set(CMAKE_CXX_STANDARD 11)

aux_source_directory(./src/ SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

# set when runs on Rasbperry Pi
# set(RPi2 TRUE)

# if solution runs on Raspberry Pi then should be used different
# optimizations in code, so we providing special definition "RPi"
# that will be understood as "define RPi" in C++ code
if (RPi2 OR RPi3)
    add_definitions(-DRPi)
endif (RPi2 OR RPi3)


# special optimizations for Raspberry Pi 2/3 or default native optimizations
if (RPi2)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=armv7-a+neon-vfpv4 -mfpu=neon-vfpv4 -mtune=native -ffast-math -funroll-loops -DNDEBUG -DBOOST_DISABLE_ASSERTS" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-mfpu=neon-vfpv4")
elseif (RPi3)
    # TODO: add flags specific to RPi3 (ex. core architecture for armv8)
else ()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -march=native -ffast-math -funroll-loops -DNDEBUG -DBOOST_DISABLE_ASSERTS" CACHE STRING COMPILE_FLAGS FORCE)
endif(RPi2)


find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} pthread)