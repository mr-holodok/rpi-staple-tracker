cmake_minimum_required(VERSION 3.8)

project(tracker LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")

# set when runs on Rasbperry Pi
set(RPi3 TRUE)

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
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mfpu=neon-vfpv4 -mtune=cortex-a53 -ffast-math -funsafe-math-optimizations -funroll-loops -DNDEBUG -DBOOST_DISABLE_ASSERTS" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-mfpu=neon-vfpv4")
else ()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -march=native -ffast-math -funroll-loops -DNDEBUG -DBOOST_DISABLE_ASSERTS" CACHE STRING COMPILE_FLAGS FORCE)
endif(RPi2)

set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-arcs -ftest-coverage")

find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(pigpio REQUIRED)

# set up a mapping so that the Release configuration for the Qt imported target is
# used in the COVERAGE CMake configuration.
set_target_properties(Qt5::Widgets PROPERTIES MAP_IMPORTED_CONFIG_COVERAGE "RELEASE")

aux_source_directory(./src/ SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST} images.qrc)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt5::Widgets
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    pigpio
    pthread
    )
