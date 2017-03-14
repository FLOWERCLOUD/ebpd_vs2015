/********************************************************************************
** Form generated from reading UI file 'widget_fitting.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WIDGET_FITTING_H
#define UI_WIDGET_FITTING_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Fitting_toolbuttons
{
public:
    QHBoxLayout *horizontalLayout;
    QToolButton *toolB_fitting;

    void setupUi(QWidget *Fitting_toolbuttons)
    {
        if (Fitting_toolbuttons->objectName().isEmpty())
            Fitting_toolbuttons->setObjectName(QStringLiteral("Fitting_toolbuttons"));
        Fitting_toolbuttons->resize(56, 40);
        horizontalLayout = new QHBoxLayout(Fitting_toolbuttons);
        horizontalLayout->setSpacing(0);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, 0, -1, 0);
        toolB_fitting = new QToolButton(Fitting_toolbuttons);
        toolB_fitting->setObjectName(QStringLiteral("toolB_fitting"));
        toolB_fitting->setMinimumSize(QSize(36, 36));
        toolB_fitting->setIconSize(QSize(32, 32));
        toolB_fitting->setCheckable(true);
        toolB_fitting->setAutoExclusive(true);
        toolB_fitting->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_fitting);


        retranslateUi(Fitting_toolbuttons);

        QMetaObject::connectSlotsByName(Fitting_toolbuttons);
    } // setupUi

    void retranslateUi(QWidget *Fitting_toolbuttons)
    {
        Fitting_toolbuttons->setWindowTitle(QApplication::translate("Fitting_toolbuttons", "Form", 0));
#ifndef QT_NO_TOOLTIP
        toolB_fitting->setToolTip(QApplication::translate("Fitting_toolbuttons", "Enable implicit adjustment.", 0));
#endif // QT_NO_TOOLTIP
    } // retranslateUi

};

namespace Ui {
    class Fitting_toolbuttons: public Ui_Fitting_toolbuttons {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_FITTING_H
