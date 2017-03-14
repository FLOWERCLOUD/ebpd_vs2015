/********************************************************************************
** Form generated from reading UI file 'widget_painting_floats.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WIDGET_PAINTING_FLOATS_H
#define UI_WIDGET_PAINTING_FLOATS_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Painting_floats
{
public:
    QVBoxLayout *verticalLayout;
    QWidget *widget;
    QHBoxLayout *horizontalLayout_2;
    QLabel *lbl_brush_size;
    QDoubleSpinBox *dSpinB_brush_size;
    QCheckBox *checkB_soft_brush;
    QWidget *widget_2;
    QHBoxLayout *horizontalLayout_3;
    QLabel *lbl_strength;
    QDoubleSpinBox *dSpinB_strength;

    void setupUi(QWidget *Painting_floats)
    {
        if (Painting_floats->objectName().isEmpty())
            Painting_floats->setObjectName(QStringLiteral("Painting_floats"));
        Painting_floats->resize(185, 112);
        verticalLayout = new QVBoxLayout(Painting_floats);
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(-1, 0, -1, 0);
        widget = new QWidget(Painting_floats);
        widget->setObjectName(QStringLiteral("widget"));
        horizontalLayout_2 = new QHBoxLayout(widget);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        lbl_brush_size = new QLabel(widget);
        lbl_brush_size->setObjectName(QStringLiteral("lbl_brush_size"));

        horizontalLayout_2->addWidget(lbl_brush_size);

        dSpinB_brush_size = new QDoubleSpinBox(widget);
        dSpinB_brush_size->setObjectName(QStringLiteral("dSpinB_brush_size"));
        dSpinB_brush_size->setDecimals(0);
        dSpinB_brush_size->setMaximum(1000);
        dSpinB_brush_size->setValue(35);

        horizontalLayout_2->addWidget(dSpinB_brush_size);


        verticalLayout->addWidget(widget);

        checkB_soft_brush = new QCheckBox(Painting_floats);
        checkB_soft_brush->setObjectName(QStringLiteral("checkB_soft_brush"));

        verticalLayout->addWidget(checkB_soft_brush);

        widget_2 = new QWidget(Painting_floats);
        widget_2->setObjectName(QStringLiteral("widget_2"));
        horizontalLayout_3 = new QHBoxLayout(widget_2);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        lbl_strength = new QLabel(widget_2);
        lbl_strength->setObjectName(QStringLiteral("lbl_strength"));

        horizontalLayout_3->addWidget(lbl_strength);

        dSpinB_strength = new QDoubleSpinBox(widget_2);
        dSpinB_strength->setObjectName(QStringLiteral("dSpinB_strength"));
        dSpinB_strength->setMinimum(-9999.99);
        dSpinB_strength->setMaximum(9999.99);
        dSpinB_strength->setSingleStep(0.1);

        horizontalLayout_3->addWidget(dSpinB_strength);


        verticalLayout->addWidget(widget_2);


        retranslateUi(Painting_floats);

        QMetaObject::connectSlotsByName(Painting_floats);
    } // setupUi

    void retranslateUi(QWidget *Painting_floats)
    {
        Painting_floats->setWindowTitle(QApplication::translate("Painting_floats", "Form", 0));
        lbl_brush_size->setText(QApplication::translate("Painting_floats", "Brush size: ", 0));
        checkB_soft_brush->setText(QApplication::translate("Painting_floats", "Soft brush", 0));
        lbl_strength->setText(QApplication::translate("Painting_floats", "Strength: ", 0));
    } // retranslateUi

};

namespace Ui {
    class Painting_floats: public Ui_Painting_floats {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_PAINTING_FLOATS_H
