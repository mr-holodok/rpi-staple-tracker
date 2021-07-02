# STAPLE tracker for Raspberry Pi
-------------------------------------
### Description

Using STAPLE tracker algo from [bertinetto](https://github.com/bertinetto/staple) and some of base code from [xuduo35](https://github.com/xuduo35/STAPLE) this solution was created.

    @article{bertinetto2015staple,
      title={Staple: Complementary Learners for Real-Time Tracking},
      author={Bertinetto, Luca and Valmadre, Jack and Golodetz, Stuart and Miksik, Ondrej and Torr, Philip},
      journal={arXiv preprint arXiv:1512.01355},
      year={2015}
    }

Solution allows you to perform real-time tracking from RPi based on user-selected target (select target with help of bounding box).
Solution has GUI - Qt and servo-motors control.
As camera RPi v1.3 was used.
    
### Prerequisites

Assumed that your target machine has next dependencies:
- OpenCV (Camera capture + all image transformations)
- Qt5 (for GUI)
- pigpio (for servo-motor control)
- OpenMP (for multithreading)

### Build

Project organized with CMake, so build is very easy.
In CMakeLists.txt you may fine tune special compiler flags or just select between available configuration for RPi2 / RPi 3.

So, in order to build project (assuming that you have downloaded all dependencies):

    mkdir build 
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j3
    sudo ./tracker

Note: sudo is necessary because pigpio works in kernel mode; without sudo you will get an error.

### Performance

Performance is dependent on image input size and target size. On average, results are like that:

- 320 x 240, target size 64 x 64 - 27 FPS
- 640 x 480, target size 64 x 64 - 21 FPS
- 1280 x 720, target size 64 x 64 - 12 FPS

When using with VNC (remote desktop) numers might me smaller.

    
### DEMO

![Video-demo](https://drive.google.com/file/d/1W4lLhd5NqbW3O605WRAqg0VEeRi3f2RW/view?usp=sharing)

