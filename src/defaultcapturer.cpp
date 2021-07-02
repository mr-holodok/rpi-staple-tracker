#include "defaultcapturer.h"
#include <QDebug>


DefaultCapturer::DefaultCapturer() : ICameraCapturer() {
    bool result = capturer.open(0, cv::CAP_V4L2);
    if (!result) {
        qDebug() << "[ERROR LOG] Open deviceId 0 failed in VideoCapturer!";
    }

    capturer.set(cv::CAP_PROP_FPS, 20);
    capturer.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capturer.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
}

bool DefaultCapturer::isOpened() const {
    return capturer.isOpened();
}

bool DefaultCapturer::set(CaptureSettings s) {
    //capturer.release();
    //capturer.open(0, cv::CAP_GSTREAMER);
    
    bool r1 = capturer.set(cv::CAP_PROP_FRAME_WIDTH, s.width());
    bool r2 = capturer.set(cv::CAP_PROP_FRAME_HEIGHT, s.height());
    bool r3 = capturer.set(cv::CAP_PROP_FPS, s.FPS());
    
    return r1 && r2 && r3;
}

void DefaultCapturer::read(cv::Mat& image) {
    capturer.read(image);
    cv::flip(image, image, -1);
}

CaptureSettings DefaultCapturer::get() const {
    return CaptureSettings(capturer.get(cv::CAP_PROP_FRAME_WIDTH),
                           capturer.get(cv::CAP_PROP_FRAME_HEIGHT),
                           capturer.get(cv::CAP_PROP_FPS));
}

DefaultCapturer::~DefaultCapturer() {
    capturer.release();
}

