#ifndef QIMAGEDISPLAY_H
#define QIMAGEDISPLAY_H

#include <QLabel>
#include <QMouseEvent>

class QImageDisplay : public QLabel
{
    Q_OBJECT
public:
    QImageDisplay(QSize imgResolution, QWidget *parent=nullptr);

public slots:
    void changeImageResolutionSize(QSize newResolution);
    void setImage(const QImage& img);
    int heightForWidth( int width ) const override;
    QSize sizeHint() const override;
    void setTargetSize(QSize targetSz);
    void setTouchscreenEnabled(bool state);

signals:
    void selectionRectChanged(QRect rect);
    void selectedRect(QRect rect);

protected:
    void mouseMoveEvent(QMouseEvent *ev) override;
    void mousePressEvent(QMouseEvent *ev) override;
    void mouseReleaseEvent(QMouseEvent *ev) override;
    void resizeEvent(QResizeEvent *event) override;

private:
    bool isLBtnPressed{false};
    // resolution needed for a correct signals notifications
    // transform coords of pixmap to coords of resolution
    QSize imageResolution, targetSize;
    QPoint selectionEdgePoint1, selectionEdgePoint2;
    bool touchscreenEnabled;
};

#endif // QIMAGEDISPLAY_H
