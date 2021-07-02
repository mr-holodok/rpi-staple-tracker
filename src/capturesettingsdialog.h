#ifndef CAPTURESETTINGSDIALOG_H
#define CAPTURESETTINGSDIALOG_H

#include <QDialog>
#include <QComboBox>
#include "capturesettings.h"

namespace Ui {
class CaptureSettingsDialog;
}

class CaptureSettingsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit CaptureSettingsDialog(QWidget *parent = nullptr);
    ~CaptureSettingsDialog();

    void setAvailableResolutions(QList<QSize>);
    void setAvailableFPS(QList<int>);
    void setDefaultResolution(QSize);
    void SetDefaultFPS(int FPS);
    CaptureSettings getCaptureSettings() const;

private:
    Ui::CaptureSettingsDialog *ui;

    QVariant boxValue(const QComboBox *box) const;
    void selectComboBoxItem(QComboBox *box, const QVariant &value);
};

#endif // CAPTURESETTINGSDIALOG_H
