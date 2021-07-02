#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "capturesettingsdialog.h"

#include <QDebug>

MainWindow::MainWindow(FrameProvider* fp, PanTiltHandler* pth, QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow), frameProvider {fp}, servoCtrl {pth}
{
    ui->setupUi(this);

    ui->pbLeft->setIcon(QIcon(":/images/img/arrow-left.png"));
    ui->pbRight->setIcon(QIcon(":/images/img/arrow-right.png"));
    ui->pbUp->setIcon(QIcon(":/images/img/arrow-up.png"));
    ui->pbDown->setIcon(QIcon(":/images/img/arrow-down.png"));

    this->image = new QImageDisplay(QSize(640, 480), ui->verticalLayoutWidget);
    ui->verticalLayout->addWidget(image);
    ui->centralwidget->setLayout(ui->verticalLayout);

    ui->targetSz->addItem("32 x 32", QVariant(QSize(32, 32)));
    ui->targetSz->addItem("64 x 64", QVariant(QSize(64, 64)));
    ui->targetSz->addItem("Ручний вибір", QVariant(QSize(0,0)));

    this->image->setTargetSize(QSize(32,32));
    this->image->setTouchscreenEnabled(true);

    FPSEntry = new QLabel(this);
    FPSEntry->setGeometry(0, 0, 100, 10);
    FPSEntry->setText("FPS: ");
    ui->statusBar->addPermanentWidget(FPSEntry);

    statusBar()->showMessage("Для відстежування оберіть ціль за допомогою лівої кнопки миші");

    connect(this->image, &QImageDisplay::selectionRectChanged, frameProvider, &FrameProvider::setSelectionRect);
    connect(this->image, &QImageDisplay::selectionRectChanged, this, &MainWindow::showTargetSize);
    connect(this->image, &QImageDisplay::selectedRect, frameProvider, &FrameProvider::setSelectionRect);
    connect(this->image, &QImageDisplay::selectedRect, this, &MainWindow::startCapture);
    connect(frameProvider, &FrameProvider::FPSUpdated, this, &MainWindow::updateFPSEntry);
    connect(frameProvider, &FrameProvider::frameReady, this->image, &QImageDisplay::setImage);
    connect(frameProvider, &FrameProvider::finished, frameProvider, &QObject::deleteLater);
    connect(frameProvider, &FrameProvider::settingsUpdated, this, &MainWindow::updatedCaptureSettings);

    connect(ui->pbLeft, &QPushButton::clicked, this, &MainWindow::moveCameraLeft);
    connect(ui->pbRight, &QPushButton::clicked, this, &MainWindow::moveCameraRight);
    connect(ui->pbUp, &QPushButton::clicked, this, &MainWindow::moveCameraUp);
    connect(ui->pbDown, &QPushButton::clicked, this, &MainWindow::moveCameraDown);

    frameProvider->startCapture();
    //qDebug() << "MainWindow: frameProvider started!";
}

void MainWindow::closeEvent(QCloseEvent *ev)
{
    frameProvider->endCapture();
    ev->accept();
    //qDebug() << "MainWindow: close event";
}

MainWindow::~MainWindow()
{
    delete ui;
    delete image;
    delete FPSEntry;
}

void MainWindow::startCapture()
{
    ui->stopBtn->setEnabled(true);
    ui->actionSettings->setEnabled(false);
    ui->targetSz->setEnabled(false);
    image->setTouchscreenEnabled(false);

    ui->pbLeft->setEnabled(false);
    ui->pbRight->setEnabled(false);
    ui->pbUp->setEnabled(false);
    ui->pbDown->setEnabled(false);

    frameProvider->setTracking(true);

    statusBar()->showMessage("Супроводжуємо ціль");
}

void MainWindow::endCapture()
{
    ui->stopBtn->setEnabled(false);
    ui->actionSettings->setEnabled(true);
    ui->targetSz->setEnabled(true);
    image->setTouchscreenEnabled(true);

    ui->pbLeft->setEnabled(true);
    ui->pbRight->setEnabled(true);
    ui->pbUp->setEnabled(true);
    ui->pbDown->setEnabled(true);

    frameProvider->setTracking(false);

    statusBar()->showMessage("Для відстежування оберіть ціль за допомогою лівої кнопки миші");
}

void MainWindow::keyPressEvent(QKeyEvent *ev)
{
    if (ev->key() == Qt::Key_Escape) {
        close();
    }
}

void MainWindow::updateFPSEntry(const int FPS)
{
    FPSEntry->setText("FPS: " + QString::number(FPS));
}

void MainWindow::configureCaptureSettings()
{
    CaptureSettingsDialog diag(this);

    diag.setAvailableResolutions(QList<QSize>({QSize(320, 240), QSize(400, 300), QSize(512, 384), QSize(640, 480), QSize(800, 600), QSize(1280, 720), QSize(1920, 1080)}));
    diag.setAvailableFPS(QList<int>{10, 15, 20, 25, 30, 35, 40});

    auto defaultSettings = frameProvider->getSettings();
    diag.setDefaultResolution(QSize(defaultSettings.width(), defaultSettings.height()));
    diag.SetDefaultFPS(defaultSettings.FPS());

    if (diag.exec()) {
        CaptureSettings settings = diag.getCaptureSettings();
        frameProvider->setSettings(settings);
    }
}

void MainWindow::updatedCaptureSettings(bool state)
{
    if (state) {
        statusBar()->showMessage("Налаштування успішно оновлено!", 5000);
        auto settings = frameProvider->getSettings();
        image->changeImageResolutionSize(QSize(settings.width(), settings.height()));
    }
    else {
        statusBar()->showMessage("Спроба зміни налаштувань зазнала невдачі!", 5000);
    }
}

void MainWindow::changeTargetSize(int cboxIndex)
{
    auto newSize = ui->targetSz->itemData(cboxIndex).toSize();
    this->image->setTargetSize(newSize);
}

void MainWindow::showTargetSize(QRect r)
{
    if (ui->targetSz->currentIndex() == ui->targetSz->count() - 1)
        statusBar()->showMessage("Розмір цілі: " + QString::number(std::abs(r.width())) + "x" + QString::number(std::abs(r.height())));
}

void MainWindow::moveCameraLeft()
{
    servoCtrl->moveLeft();
}

void MainWindow::moveCameraRight()
{
    servoCtrl->moveRight();
}

void MainWindow::moveCameraUp()
{
    servoCtrl->moveUp();
}

void MainWindow::moveCameraDown()
{
    servoCtrl->moveDown();
}
