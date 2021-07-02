#ifndef VIDEOCAPTURER_H
#define VIDEOCAPTURER_H

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include "icameracapturer.hpp"
#include "tsr_buffer.hpp"
#include <thread>

class VideoCapturer : public ICameraCapturer
{
public:
    VideoCapturer();
    virtual ~VideoCapturer();

    bool isOpened() const override;
    bool set(CaptureSettings) override;
    CaptureSettings get() const override;
    void read(cv::Mat& image) override;

private:
    cv::VideoCapture capturer;
    TSRBuffer<cv::Mat> buff {10};
    std::thread captureTread;
    bool stopFlag {false};
    mutable std::mutex vc_mutex;
    std::chrono::time_point<std::chrono::system_clock> lastAccessTime {std::chrono::system_clock::now()};
};

#endif // VIDEOCAPTURER_H
