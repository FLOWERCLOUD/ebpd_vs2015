/********************************************************************************
** Form generated from reading UI file 'papagate_dialog.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PAPAGATE_DIALOG_H
#define UI_PAPAGATE_DIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_propagate_compute_dialog
{
public:
    QPushButton *propagate_compute;
    QCheckBox *propagate_front;
    QCheckBox *propagate_back;
    QLabel *center_frame;
    QLabel *front_endframe;
    QLabel *back_endframe;
    QLineEdit *front_endframe_text;
    QLineEdit *back_endframe_text;
    QLineEdit *center_frame_text;

    void setupUi(QWidget *propagate_compute_dialog)
    {
        if (propagate_compute_dialog->objectName().isEmpty())
            propagate_compute_dialog->setObjectName(QStringLiteral("propagate_compute_dialog"));
        propagate_compute_dialog->resize(400, 300);
        QIcon icon;
        icon.addFile(QStringLiteral("Resources/propagate.png"), QSize(), QIcon::Normal, QIcon::Off);
        propagate_compute_dialog->setWindowIcon(icon);
        propagate_compute = new QPushButton(propagate_compute_dialog);
        propagate_compute->setObjectName(QStringLiteral("propagate_compute"));
        propagate_compute->setGeometry(QRect(160, 240, 75, 23));
        propagate_front = new QCheckBox(propagate_compute_dialog);
        propagate_front->setObjectName(QStringLiteral("propagate_front"));
        propagate_front->setGeometry(QRect(30, 20, 191, 16));
        propagate_back = new QCheckBox(propagate_compute_dialog);
        propagate_back->setObjectName(QStringLiteral("propagate_back"));
        propagate_back->setGeometry(QRect(30, 50, 131, 16));
        center_frame = new QLabel(propagate_compute_dialog);
        center_frame->setObjectName(QStringLiteral("center_frame"));
        center_frame->setGeometry(QRect(200, 20, 81, 20));
        front_endframe = new QLabel(propagate_compute_dialog);
        front_endframe->setObjectName(QStringLiteral("front_endframe"));
        front_endframe->setGeometry(QRect(40, 80, 121, 20));
        back_endframe = new QLabel(propagate_compute_dialog);
        back_endframe->setObjectName(QStringLiteral("back_endframe"));
        back_endframe->setGeometry(QRect(40, 110, 121, 20));
        front_endframe_text = new QLineEdit(propagate_compute_dialog);
        front_endframe_text->setObjectName(QStringLiteral("front_endframe_text"));
        front_endframe_text->setGeometry(QRect(200, 80, 113, 20));
        back_endframe_text = new QLineEdit(propagate_compute_dialog);
        back_endframe_text->setObjectName(QStringLiteral("back_endframe_text"));
        back_endframe_text->setGeometry(QRect(200, 110, 113, 20));
        center_frame_text = new QLineEdit(propagate_compute_dialog);
        center_frame_text->setObjectName(QStringLiteral("center_frame_text"));
        center_frame_text->setGeometry(QRect(280, 20, 113, 20));

        retranslateUi(propagate_compute_dialog);

        QMetaObject::connectSlotsByName(propagate_compute_dialog);
    } // setupUi

    void retranslateUi(QWidget *propagate_compute_dialog)
    {
        propagate_compute_dialog->setWindowTitle(QApplication::translate("propagate_compute_dialog", "diaglog", 0));
        propagate_compute->setText(QApplication::translate("propagate_compute_dialog", "compute", 0));
        propagate_front->setText(QApplication::translate("propagate_compute_dialog", "propagate_front", 0));
        propagate_back->setText(QApplication::translate("propagate_compute_dialog", "propagate_back", 0));
        center_frame->setText(QApplication::translate("propagate_compute_dialog", "centerframe", 0));
        front_endframe->setText(QApplication::translate("propagate_compute_dialog", "front_endframe", 0));
        back_endframe->setText(QApplication::translate("propagate_compute_dialog", "back_endframe", 0));
    } // retranslateUi

};

namespace Ui {
    class propagate_compute_dialog: public Ui_propagate_compute_dialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PAPAGATE_DIALOG_H
