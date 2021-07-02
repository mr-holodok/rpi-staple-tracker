#ifndef FRAMEPROVIDER_H
#define FRAMEPROVIDER_H

#include <QThread>
#include <QImage>
#include <QRect>
#include <QPoint>
#include <QReadWriteLock>
#include "icameracapturer.hpp"
#include "tracker.hpp"
#include "capturesettings.h"
#include "qimagedisplay.h"
#include <functional>
#include "pantiltinterface.hpp"

class FrameProvider : public QThread
{
    Q_OBJECT

public:
    FrameProvider(QObject *parent = nullptr);
    ~FrameProvider();

    void setSettings(CaptureSettings);
    CaptureSettings getSettings() const;

public slots:
    void startCapture();
    void setTracking(bool state);
    void endCapture();
    void setSelectionRect(QRect rect);

signals:
    void failedToInitCamera();
    void frameReady(const QImage& qimage);
    void FPSUpdated(const int FPS);
    void settingsUpdated(bool state);
    void targetRectUpdated(QRect, QSize);

private:
    void run() override;
    bool updateSettings(CaptureSettings);

    std::unique_ptr<ICameraCapturer> capturer;
    StapleTracker tracker;
    bool isTracking {false};
    bool isInitStep {false};
    bool isSelectingTarget {false};
    bool stop {false};
    QRect selectionRect;
    mutable QReadWriteLock lock;
    // wrapper for updateSettings to call in loop in safe way
    std::function<bool()> secureSettingUpdate {nullptr};
    PanTiltInterface* panTiltI {nullptr};
    unsigned int initStepModulo {0};
};

#endif // FRAMEPROVIDER_H
