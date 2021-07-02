#ifndef PANTILTCONTROLLER_H
#define PANTILTCONTROLLER_H

#include <QObject>
#include <QRect>
#include <thread>
#include "servo_control.hpp"
#include "frameprovider.hpp"
#include "pantiltinterface.hpp"

class PanTiltHandler : public QObject, public PanTiltInterface
{
    Q_OBJECT
public:
    explicit PanTiltHandler(FrameProvider* fp, QObject *parent = nullptr);
    ~PanTiltHandler();

    void toStartPosition();
    void stop();

    void moveLeft();
    void moveRight();
    void moveUp();
    void moveDown();

public slots:
    void updateTargetPos(QRect pos, QSize backgroundSize) override;

private:
    QRect targetPos, prevPos;
    std::atomic<float> panShift {0}, tiltShift {0};
    PanTiltController cntr{};
    bool stopFlag {false};
    std::thread panController, tiltController;
};

#endif // PANTILTCONTROLLER_H
