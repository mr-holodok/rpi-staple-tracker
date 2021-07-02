#include "videocapturer.hpp"
#include <QDebug>
#include <chrono>


VideoCapturer::VideoCapturer() : ICameraCapturer() {
    bool result = capturer.open(0, cv::CAP_V4L2);
    if (!result) {
        qDebug() << "[ERROR LOG] Open deviceId 0 failed in VideoCapturer!";
    }

    capturer.set(cv::CAP_PROP_FPS, 30);

    captureTread = std::thread([this]() {
        if (!capturer.isOpened()) {
            qDebug() << "[ERROR LOG] Open cam failed! Capture thread exited!";
            return;
        }

        while (!stopFlag) {
            cv::Mat image;
            vc_mutex.lock();
            capturer.read(image);
            vc_mutex.unlock();

            buff.put(image);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });
}

bool VideoCapturer::isOpened() const {
    std::lock_guard<std::mutex> lg(vc_mutex);
    return capturer.isOpened();
}

bool VideoCapturer::set(CaptureSettings s) {
    std::lock_guard<std::mutex> lg(vc_mutex);

    //capturer.release();
    //capturer.open(0, cv::CAP_V4L2);

    bool r1 = capturer.set(cv::CAP_PROP_FRAME_WIDTH, s.width());
    bool r2 = capturer.set(cv::CAP_PROP_FRAME_HEIGHT, s.height());
    bool r3 = capturer.set(cv::CAP_PROP_FPS, s.FPS());
    return r1 && r2 && r3;
}

void VideoCapturer::read(cv::Mat& image) {
    while (buff.empty())
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    auto curr_time = std::chrono::system_clock::now();
    auto delta = curr_time - lastAccessTime; 
    auto waitTime = std::chrono::milliseconds(long (1000 / (capturer.get(cv::CAP_PROP_FPS))));

    if (delta < waitTime) {
        std::this_thread::sleep_for(std::chrono::milliseconds(long (1000 / (capturer.get(cv::CAP_PROP_FPS))) - 15));
    }
    lastAccessTime = curr_time;

    image = buff.get().clone();
    cv::flip(image, image, -1);
}

CaptureSettings VideoCapturer::get() const {
    std::lock_guard<std::mutex> lg(vc_mutex);
    return CaptureSettings(capturer.get(cv::CAP_PROP_FRAME_WIDTH),
                           capturer.get(cv::CAP_PROP_FRAME_HEIGHT),
                           capturer.get(cv::CAP_PROP_FPS));
}

VideoCapturer::~VideoCapturer() {
    stopFlag = true;
    captureTread.join();
    capturer.release();
}
