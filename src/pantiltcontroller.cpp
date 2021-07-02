#include "pantiltcontroller.h"

PanTiltHandler::PanTiltHandler(FrameProvider *fp, QObject *parent) : QObject(parent) {
    cntr.init();
    toStartPosition();

    panController = std::thread([this]() {
        while (!stopFlag) {
            float shift = panShift.exchange(0);
            cntr.handlePanShift(shift);
        }
    });
    tiltController = std::thread ([this]() {
        while (!stopFlag) {
            float shift = tiltShift.exchange(0);
            cntr.handleTiltShift(shift);
        }
    });

    connect(fp, &FrameProvider::targetRectUpdated, this, &PanTiltHandler::updateTargetPos);
}

int calcPanCenterDelta(const QSize frameSize, const QRect &bbox) {
    int frameCenter = frameSize.width() / 2;
    int bboxCenter = bbox.x() + bbox.width() / 2;
    int diff = frameCenter - bboxCenter;
    //const float CAM_HOR_RANGE = 53.5f;
    //float result = diff * CAM_HOR_RANGE / FRAME_WIDTH;
    return diff;
}

int calcTiltCenterDelta(const QSize frameSize, const QRect &bbox){
    int frameCenter = frameSize.height() / 2;
    int bboxCenter = bbox.y() + bbox.height() / 2;
    int diff = frameCenter - bboxCenter;
    return diff;
}

void PanTiltHandler::updateTargetPos(QRect pos, QSize backgroundSize) {
    targetPos = pos;
    if (prevPos.isNull() || (prevPos.x() != targetPos.x() || prevPos.y() != targetPos.y()) 
		    /* || (std::abs(calcPanCenterDelta(backgroundSize, targetPos)) > 10) */ ) {
        int panDelta = calcPanCenterDelta(backgroundSize, targetPos);
        panShift.store((float) panDelta * 100 / backgroundSize.width());

        int tiltDelta = calcTiltCenterDelta(backgroundSize, targetPos);
        tiltShift.store((float) tiltDelta * 100 / backgroundSize.height());
    }

    prevPos = targetPos;
}

void PanTiltHandler::toStartPosition() {
    const float startAngle = 90.f;
    cntr.setPanAngle(startAngle);
    cntr.setTiltAngle(startAngle);
}

PanTiltHandler::~PanTiltHandler() {
    if (!stopFlag)
        stop();
}

void PanTiltHandler::stop() {
    //for (int i = 0; i < 50; ++i)
    //    cntr.setPanAngle(90.f);

    stopFlag = true;
    panController.join();
    tiltController.join();
    cntr.cleanup();
}

void PanTiltHandler::moveLeft() {
    cntr.moveLeft();
}

void PanTiltHandler::moveRight() {
    cntr.moveRight();
}

void PanTiltHandler::moveUp() {
    cntr.moveUp();
}

void PanTiltHandler::moveDown() {
    cntr.moveDown();
}
