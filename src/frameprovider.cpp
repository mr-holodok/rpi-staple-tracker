#include "frameprovider.hpp"
#include "videocapturer.hpp"
#include "defaultcapturer.h"
#include <QDebug>
#include <chrono>

FrameProvider::FrameProvider(QObject *parent) : QThread(parent), capturer{std::make_unique<DefaultCapturer>()} {
    if (!capturer->isOpened()) {
        emit failedToInitCamera();
        qDebug() << "[ERROR LOG] Camera open failed!";
    }
}

FrameProvider::~FrameProvider() {
    endCapture();
    wait();
}

void FrameProvider::startCapture() {
    this->start();
}

void FrameProvider::run() {
    cv::Mat image;

    auto timeStart = std::chrono::steady_clock::now();
    uint framesCount = 0;

    while (true) {

        lock.lockForWrite();
        capturer->read(image);
        lock.unlock();

        if (image.empty()) {
            qDebug() << "[ERROR LOG] FrameProvider: captured image is empty!";
            continue;
        }

        lock.lockForRead();
        bool isTracking = this->isTracking;
        bool isInitStep = this->isInitStep;
        bool isSelectingTarget = this->isSelectingTarget;
        QRect selectionRect = this->selectionRect;
        lock.unlock();

        if (isTracking) {
            if (isInitStep) {
                // conversion to opencv rect
                cv::Rect r = cv::Rect(selectionRect.topLeft().x(), selectionRect.topLeft().y(), 
                        selectionRect.width(), selectionRect.height());
                emit targetRectUpdated(selectionRect, QSize(image.size().width, image.size().height));

                tracker.trackerInit(image, r);
                isInitStep = false;

                lock.lockForWrite();
                this->isInitStep = false;
                lock.unlock();

                initStepModulo = (framesCount + 1) % 2;

                cv::rectangle(image, r.tl(), r.br(), cv::Scalar(0, 0, 255), 2);
            }
            else {
                static cv::Rect bbox;
                //if (framesCount % 2 == initStepModulo) {
                    bbox = tracker.getNextPos(image);
                    selectionRect.setRect(bbox.x, bbox.y, bbox.width, bbox.height);
                    emit targetRectUpdated(selectionRect, QSize(image.size().width, image.size().height));
                //}
                cv::rectangle(image, bbox.tl(), bbox.br(), cv::Scalar(0, 0, 255), 2); 
            }
        }
        else if (isSelectingTarget) {
            // only drawing bounding box selected by user
            cv::Point tl(selectionRect.topLeft().x(), selectionRect.topLeft().y());
            cv::Point br(selectionRect.bottomRight().x(), selectionRect.bottomRight().y());

            cv::rectangle(image, tl, br, cv::Scalar(255, 0, 0), 2);
        }

        /*cv::line(image, cv::Point(0, image.size().height / 2), 
                cv::Point(image.size().width - 1, image.size().height / 2), 
                cv::Scalar(0, 255, 0), 2);
        cv::line(image, cv::Point(image.size().width / 2, 0),
                cv::Point(image.size().width / 2, image.size().height - 1),
                cv::Scalar(0, 255, 0), 2);*/

        cv::Mat img_copy;
        cv::cvtColor(image, img_copy, cv::COLOR_BGR2RGB);
        
        QImage qimg = QImage(img_copy.data, img_copy.cols, img_copy.rows, img_copy.step, QImage::Format_RGB888);
        // create a deep copy of QImage, because qimg points to buffer on stack
        // yes, signal creates safe copy of it, but this copy highly coupled with this buffer and
        // it still points to buffer which may be already cleaned up from stack and we will get a SEGFAULT

        QImage copy = qimg.copy();
        emit frameReady(copy);

        ++framesCount;
        if (framesCount % 10 == 0) {
            auto timeEnd = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart);
            timeStart = timeEnd;

            int FPS = 10 / (duration.count() / 1000.0);
            emit FPSUpdated(FPS);
        }

        lock.lockForRead();
        bool needExit = stop;
        bool needUpdateSettings = secureSettingUpdate != nullptr;
        lock.unlock();

        if (needExit) {
            break;
        }

        if (needUpdateSettings) {
            lock.lockForWrite();
            bool result = secureSettingUpdate();
            emit settingsUpdated(result);
            secureSettingUpdate = nullptr;
            lock.unlock();
        }
    }
}

void FrameProvider::setTracking(bool state) {
    lock.lockForWrite();
    isTracking = state;
    isInitStep = state;
    isSelectingTarget = false;
    lock.unlock();
}

void FrameProvider::endCapture() {
    lock.lockForWrite();
    stop = true;
    lock.unlock();
}

void FrameProvider::setSelectionRect(QRect rect) {
    lock.lockForWrite();
    selectionRect = rect;
    isSelectingTarget = true;
    lock.unlock();
}

void FrameProvider::setSettings(CaptureSettings s) {
    // fistly, check correctness of the settings
    if (s.width() <= 0 || s.height() <= 0 || s.FPS() < 10) {
        return;
    }
    lock.lockForWrite();
    secureSettingUpdate = std::bind(&FrameProvider::updateSettings, this, s);
    lock.unlock();
}

CaptureSettings FrameProvider::getSettings() const {
    lock.lockForRead();
    auto settings = capturer->get();
    lock.unlock();

    return settings;
}

bool FrameProvider::updateSettings(CaptureSettings s) {
    return capturer->set(s);
}

