#include "capturesettingsdialog.h"
#include "ui_capturesettingsdialog.h"

CaptureSettingsDialog::CaptureSettingsDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::CaptureSettingsDialog)
{
    ui->setupUi(this);
}

CaptureSettingsDialog::~CaptureSettingsDialog()
{
    delete ui;
}

void CaptureSettingsDialog::setAvailableResolutions(QList<QSize> resolutions)
{
    for (const auto &size : resolutions) {
        QString resolutionStr = QString::number(size.width()) + " x " + QString::number(size.height());
        ui->resolutionBox->addItem(resolutionStr, QVariant(size));
    }
}

void CaptureSettingsDialog::setAvailableFPS(QList<int> fpsList)
{
    for (const auto &fps : fpsList) {
        ui->fpsBox->addItem(QString::number(fps), QVariant(fps));
    }
}

void CaptureSettingsDialog::setDefaultResolution(QSize resolution)
{
    selectComboBoxItem(ui->resolutionBox, resolution);
}

void CaptureSettingsDialog::SetDefaultFPS(int FPS)
{
    selectComboBoxItem(ui->fpsBox, FPS);
}

CaptureSettings CaptureSettingsDialog::getCaptureSettings() const
{
    QSize resolution = boxValue(ui->resolutionBox).toSize();
    int fps = boxValue(ui->fpsBox).toInt();
    return CaptureSettings(resolution.width(), resolution.height(), fps);
}

QVariant CaptureSettingsDialog::boxValue(const QComboBox *box) const
{
    int idx = box->currentIndex();
    if (idx == -1)
        return QVariant();

    return box->itemData(idx);
}

void CaptureSettingsDialog::selectComboBoxItem(QComboBox *box, const QVariant &value)
{
    for (int i = 0; i < box->count(); ++i) {
        if (box->itemData(i) == value) {
            box->setCurrentIndex(i);
            break;
        }
    }
}
