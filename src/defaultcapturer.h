#ifndef DEFAULTCAPTURER_H
#define DEFAULTCAPTURER_H


#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include "icameracapturer.hpp"
#include "tsr_buffer.hpp"
#include <thread>

class DefaultCapturer : public ICameraCapturer
{
public:
    DefaultCapturer();
    virtual ~DefaultCapturer();

    bool isOpened() const override;
    bool set(CaptureSettings) override;
    CaptureSettings get() const override;
    void read(cv::Mat& image) override;

private:
    cv::VideoCapture capturer;
};

#endif // DEFAULTCAPTURER_H
