#ifndef PANTILTINTERFACE_HPP
#define PANTILTINTERFACE_HPP

#include <QRect>

class PanTiltInterface
{
public:
    virtual ~PanTiltInterface() = default;
    virtual void updateTargetPos(QRect pos, QSize backgroundSize) = 0;
};

#endif // PANTILTINTERFACE_H
