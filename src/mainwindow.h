#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "qimagedisplay.h"
#include "frameprovider.hpp"
#include "pantiltcontroller.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(FrameProvider* fp, PanTiltHandler *pth, QWidget *parent = nullptr);
    ~MainWindow();

    QImageDisplay *image;
    bool isTracked {false};

public slots:
    void closeEvent(QCloseEvent *ev) override;
    void keyPressEvent(QKeyEvent *ev) override;
    void configureCaptureSettings();

private slots:
    void startCapture();
    void endCapture();
    void updateFPSEntry(const int FPS);
    void updatedCaptureSettings(bool state);
    void changeTargetSize(int cboxIndex);
    void showTargetSize(QRect);
    void moveCameraLeft();
    void moveCameraRight();
    void moveCameraUp();
    void moveCameraDown();

private:
    Ui::MainWindow *ui;
    QPoint p1, p2;
    QRect rect;
    QSize imageResolution {640, 480};
    FrameProvider* frameProvider;
    QLabel *FPSEntry;
    PanTiltHandler *servoCtrl;
};

#endif // MAINWINDOW_H
