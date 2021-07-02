#include "qimagedisplay.h"
#include <QDebug>

QRect createRect(QPoint p1, QPoint p2);

QImageDisplay::QImageDisplay(QSize imgResolution, QWidget *parent) : QLabel(parent), imageResolution{imgResolution}
{
    setAlignment(Qt::AlignCenter);
    setMinimumSize(640, 480);
    //setMinimumSize(320, 180);
}

void QImageDisplay::setImage(const QImage& image) 
{
    if (image.isNull()) {
        qDebug() << "QImageDisplay: image in null!";
        return;
    }

    setPixmap(QPixmap::fromImage(image.scaled(width(), height(), Qt::KeepAspectRatio)));
}

void QImageDisplay::mouseMoveEvent(QMouseEvent *ev)
{
    // if left button pressed update second point
    // and when released create rect from 2 points
    if (touchscreenEnabled && isLBtnPressed) {
        // get points coords in sizes of resolution
        int pixmapX = ev->localPos().x() - (width() - pixmap()->width()) / 2;
        float scaleX = imageResolution.width() / float(pixmap()->width());
        selectionEdgePoint2.setX(pixmapX * scaleX);
        int pixmapY = ev->localPos().y() - (height() - pixmap()->height()) / 2;
        float scaleY = imageResolution.height() / float(pixmap()->height());
        selectionEdgePoint2.setY(pixmapY * scaleY);

        // if target size not manual
        if (targetSize.width() > 0 && targetSize.height() > 0)
        {
            // click point is center so create such a rect
            selectionEdgePoint1.setX(selectionEdgePoint2.x() - targetSize.width() / 2);
            selectionEdgePoint1.setY(selectionEdgePoint2.y() - targetSize.height() / 2);
            selectionEdgePoint2.setX(selectionEdgePoint2.x() + targetSize.width() / 2);
            selectionEdgePoint2.setY(selectionEdgePoint2.y() + targetSize.height() / 2);
        }

        emit selectionRectChanged(QRect(selectionEdgePoint1, selectionEdgePoint2));
    }
}

void QImageDisplay::mousePressEvent(QMouseEvent *ev)
{
    // start tracking position of cursor
    if (touchscreenEnabled && ev->button() == Qt::LeftButton) {
        isLBtnPressed = true;
        int pixmapX = ev->localPos().x() - (width() - pixmap()->width()) / 2;
        float scaleX = imageResolution.width() / float(pixmap()->width());
        selectionEdgePoint1.setX(pixmapX * scaleX);
        int pixmapY = ev->localPos().y() - (height() - pixmap()->height()) / 2;
        float scaleY = imageResolution.height() / float(pixmap()->height());
        selectionEdgePoint1.setY(pixmapY * scaleY);
        selectionEdgePoint2 = selectionEdgePoint1;

        // if target size not manual
        if (targetSize.width() > 0 && targetSize.height() > 0)
        {
            // ckicked point is center so create such a rect
            selectionEdgePoint1.setX(selectionEdgePoint1.x() - targetSize.width() / 2);
            selectionEdgePoint1.setY(selectionEdgePoint1.y() - targetSize.height() / 2);
            selectionEdgePoint2.setX(selectionEdgePoint2.x() + targetSize.width() / 2);
            selectionEdgePoint2.setY(selectionEdgePoint2.y() + targetSize.height() / 2);
        }
    }
}

// points can be set chaoticaly, but we must form rect
// from top-left and bottom-right points
QRect createRect(QPoint p1, QPoint p2)
{
    bool x_inc = p1.x() < p2.x();
    bool y_inc = p1.y() < p2.y();
    // ?_inc means increasing order of coords

    if (x_inc && y_inc)
        return QRect(p1.x(), p1.y(), p2.x() - p1.x(), p2.y() - p1.y());
    else if (!x_inc && !y_inc)
        return QRect(p2.x(), p2.y(), p1.x() - p2.x(), p1.y() - p2.y());
    else if (!x_inc && y_inc)
        return QRect(p2.x(), p1.y(), p1.x() - p2.x(), p2.y() - p1.y());
    else
        return QRect(p1.x(), p2.y(), p2.x() - p1.x(), p1.y() - p2.y());
}

void QImageDisplay::mouseReleaseEvent(QMouseEvent *ev)
{
    // left button released, so rectangle is selected
    // calc it by points
    if (touchscreenEnabled && ev->button() == Qt::LeftButton) {
        isLBtnPressed = false;

        QRect area = createRect(selectionEdgePoint1, selectionEdgePoint2);
        emit selectedRect(area);
    }
}

void QImageDisplay::changeImageResolutionSize(QSize newResolution)
{
    imageResolution = newResolution;
    //setMinimumSize(imageResolution.width(), imageResolution.height());
}

void QImageDisplay::resizeEvent(QResizeEvent *event)
{
    if (pixmap()) {
        QPixmap px = pixmap()->scaled(event->size(), Qt::KeepAspectRatio);
        setPixmap(px);
    }
}

int QImageDisplay::heightForWidth( int width ) const
{
    return !pixmap() ? this->height() : ((qreal)pixmap()->height() * width) / pixmap()->width();
}

QSize QImageDisplay::sizeHint() const
{
    int w = width();
    return QSize(w, heightForWidth(w));
}

void QImageDisplay::setTargetSize(QSize targetSz)
{
    targetSize = targetSz;
}

void QImageDisplay::setTouchscreenEnabled(bool state)
{
    touchscreenEnabled = state;
}
