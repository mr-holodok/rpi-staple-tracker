#include "mainwindow.h"

#include <QApplication>
#include <QDebug>
#include "frameprovider.hpp"
#include "pantiltcontroller.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    std::unique_ptr<FrameProvider> fp = std::make_unique<FrameProvider>();
    PanTiltHandler cntr{fp.get()};

    MainWindow w(fp.get(), &cntr);
    w.show();

    a.exec();

    cntr.stop();
    return 0;
}
