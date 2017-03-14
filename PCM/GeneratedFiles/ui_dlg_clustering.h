/********************************************************************************
** Form generated from reading UI file 'dlg_clustering.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DLG_CLUSTERING_H
#define UI_DLG_CLUSTERING_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Dialog
{
public:
    QWidget *layoutWidget;
    QVBoxLayout *vboxLayout;
    QPushButton *okButton;
    QPushButton *cancelButton;
    QSpacerItem *spacerItem;
    QSlider *kClass_Slider;
    QSlider *kNeighbours_Slider;
    QLabel *label;
    QLabel *label_2;
    QLabel *label_3;

    void setupUi(QDialog *Dialog)
    {
        if (Dialog->objectName().isEmpty())
            Dialog->setObjectName(QStringLiteral("Dialog"));
        Dialog->resize(410, 178);
        layoutWidget = new QWidget(Dialog);
        layoutWidget->setObjectName(QStringLiteral("layoutWidget"));
        layoutWidget->setGeometry(QRect(300, 20, 77, 106));
        vboxLayout = new QVBoxLayout(layoutWidget);
        vboxLayout->setSpacing(6);
        vboxLayout->setObjectName(QStringLiteral("vboxLayout"));
        vboxLayout->setContentsMargins(0, 0, 0, 0);
        okButton = new QPushButton(layoutWidget);
        okButton->setObjectName(QStringLiteral("okButton"));

        vboxLayout->addWidget(okButton);

        cancelButton = new QPushButton(layoutWidget);
        cancelButton->setObjectName(QStringLiteral("cancelButton"));

        vboxLayout->addWidget(cancelButton);

        spacerItem = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        vboxLayout->addItem(spacerItem);

        kClass_Slider = new QSlider(Dialog);
        kClass_Slider->setObjectName(QStringLiteral("kClass_Slider"));
        kClass_Slider->setGeometry(QRect(80, 40, 160, 22));
        kClass_Slider->setMaximum(20);
        kClass_Slider->setOrientation(Qt::Horizontal);
        kNeighbours_Slider = new QSlider(Dialog);
        kNeighbours_Slider->setObjectName(QStringLiteral("kNeighbours_Slider"));
        kNeighbours_Slider->setGeometry(QRect(80, 80, 160, 22));
        kNeighbours_Slider->setOrientation(Qt::Horizontal);
        label = new QLabel(Dialog);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(20, 40, 54, 12));
        label_2 = new QLabel(Dialog);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(10, 80, 61, 16));
        label_3 = new QLabel(Dialog);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(30, 140, 54, 12));

        retranslateUi(Dialog);
        QObject::connect(okButton, SIGNAL(clicked()), Dialog, SLOT(accept()));
        QObject::connect(cancelButton, SIGNAL(clicked()), Dialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(Dialog);
    } // setupUi

    void retranslateUi(QDialog *Dialog)
    {
        Dialog->setWindowTitle(QApplication::translate("Dialog", "Dialog", 0));
        okButton->setText(QApplication::translate("Dialog", "OK", 0));
        cancelButton->setText(QApplication::translate("Dialog", "Cancel", 0));
        label->setText(QApplication::translate("Dialog", "#Class", 0));
        label_2->setText(QApplication::translate("Dialog", "#Neighbour", 0));
        label_3->setText(QApplication::translate("Dialog", "TextLabel", 0));
    } // retranslateUi

};

namespace Ui {
    class Dialog: public Ui_Dialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DLG_CLUSTERING_H
