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
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=armv7-a -mfpu=neon -mtune=native -funroll-loops -DNDEBUG -DBOOST_DISABLE_ASSERTS" CACHE STRING COMPILE_FLAGS FORCE)
elseif (RPi3)
    # TODO: add flags specific to RPi3 (ex. core architecture for armv8)
else ()
    message("CMAKE_CXX_FLAGS_RELEASE are: ${CMAKE_CXX_FLAGS_RELEASE}")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -march=native --fast-math -ffast-math -mtune=native -funroll-loops -DNDEBUG -DBOOST_DISABLE_ASSERTS" CACHE STRING COMPILE_FLAGS FORCE)
    #  set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -march=native -mfpu=native -mtune=native -funroll-loops -DNDEBUG -DBOOST_DISABLE_ASSERTS" CACHE STRING COMPILE_FLAGS FORCE)
endif(RPi2)


if (UNIX)
    find_package(OpenCV REQUIRED)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} pthread)
elseif (WIN32)
    # PAY ATTENTION: if file 'findOpenCV.cmake' is not found by CMake -
    # then provide 'OpenCV_DIR' variable, pointing to 'OpenCVConfig.cmake'
    # and 'OpenCVModules.cmake' [needed in OpenCVConfig.cmake]
    find_package(OpenCV REQUIRED)
    # include_directories(${OpenCV_INCLUDE_DIRS}) # Not needed for CMake >= 2.8.11
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
endif (UNIX)