/********************************************************************************
** Form generated from reading UI file 'widget_viewports.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WIDGET_VIEWPORTS_H
#define UI_WIDGET_VIEWPORTS_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Viewports_toolbuttons
{
public:
    QHBoxLayout *horizontalLayout;
    QToolButton *toolB_single;
    QToolButton *toolB_doubleH;
    QToolButton *toolB_doubleV;
    QToolButton *toolB_four;

    void setupUi(QWidget *Viewports_toolbuttons)
    {
        if (Viewports_toolbuttons->objectName().isEmpty())
            Viewports_toolbuttons->setObjectName(QStringLiteral("Viewports_toolbuttons"));
        Viewports_toolbuttons->resize(174, 38);
        horizontalLayout = new QHBoxLayout(Viewports_toolbuttons);
        horizontalLayout->setSpacing(0);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, 0, -1, 0);
        toolB_single = new QToolButton(Viewports_toolbuttons);
        toolB_single->setObjectName(QStringLiteral("toolB_single"));
        toolB_single->setMinimumSize(QSize(36, 36));
        toolB_single->setIconSize(QSize(32, 32));
        toolB_single->setCheckable(true);
        toolB_single->setChecked(true);
        toolB_single->setAutoExclusive(true);
        toolB_single->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_single);

        toolB_doubleH = new QToolButton(Viewports_toolbuttons);
        toolB_doubleH->setObjectName(QStringLiteral("toolB_doubleH"));
        toolB_doubleH->setMinimumSize(QSize(36, 36));
        toolB_doubleH->setIconSize(QSize(32, 32));
        toolB_doubleH->setCheckable(true);
        toolB_doubleH->setAutoExclusive(true);
        toolB_doubleH->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_doubleH);

        toolB_doubleV = new QToolButton(Viewports_toolbuttons);
        toolB_doubleV->setObjectName(QStringLiteral("toolB_doubleV"));
        toolB_doubleV->setMinimumSize(QSize(36, 36));
        toolB_doubleV->setIconSize(QSize(32, 32));
        toolB_doubleV->setCheckable(true);
        toolB_doubleV->setAutoExclusive(true);
        toolB_doubleV->setAutoRaise(true);
        toolB_doubleV->setArrowType(Qt::NoArrow);

        horizontalLayout->addWidget(toolB_doubleV);

        toolB_four = new QToolButton(Viewports_toolbuttons);
        toolB_four->setObjectName(QStringLiteral("toolB_four"));
        toolB_four->setMinimumSize(QSize(36, 36));
        toolB_four->setIconSize(QSize(32, 32));
        toolB_four->setCheckable(true);
        toolB_four->setAutoExclusive(true);
        toolB_four->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_four);


        retranslateUi(Viewports_toolbuttons);

        QMetaObject::connectSlotsByName(Viewports_toolbuttons);
    } // setupUi

    void retranslateUi(QWidget *Viewports_toolbuttons)
    {
        Viewports_toolbuttons->setWindowTitle(QApplication::translate("Viewports_toolbuttons", "Form", 0));
#ifndef QT_NO_TOOLTIP
        toolB_single->setToolTip(QApplication::translate("Viewports_toolbuttons", "Single viewport", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_doubleH->setToolTip(QApplication::translate("Viewports_toolbuttons", "Two viewports (Horizontal split)", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_doubleV->setToolTip(QApplication::translate("Viewports_toolbuttons", "Two viewports (Vertical split)", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_four->setToolTip(QApplication::translate("Viewports_toolbuttons", "Four viewports", 0));
#endif // QT_NO_TOOLTIP
        toolB_four->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class Viewports_toolbuttons: public Ui_Viewports_toolbuttons {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_VIEWPORTS_H
