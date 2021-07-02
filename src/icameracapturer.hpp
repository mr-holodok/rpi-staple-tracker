#ifndef ICAMERACAPTURER_H
#define ICAMERACAPTURER_H

#include <opencv2/core/core.hpp>
#include "capturesettings.h"

class ICameraCapturer {
public:
    ICameraCapturer() {};
    virtual ~ICameraCapturer() {};

    virtual bool isOpened() const = 0;
    virtual bool set(CaptureSettings) = 0;
    virtual CaptureSettings get() const = 0;
    virtual void read(cv::Mat&) = 0;
};

#endif // ICAMERACAPTURER_H
