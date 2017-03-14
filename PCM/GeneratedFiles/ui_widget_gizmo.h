/********************************************************************************
** Form generated from reading UI file 'widget_gizmo.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WIDGET_GIZMO_H
#define UI_WIDGET_GIZMO_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Gizmo_toolbuttons
{
public:
    QHBoxLayout *horizontalLayout;
    QToolButton *toolB_show_gizmo;
    QToolButton *toolB_translate;
    QToolButton *toolB_rotate;
    QToolButton *toolB_trackball;
    QToolButton *toolB_scale;

    void setupUi(QWidget *Gizmo_toolbuttons)
    {
        if (Gizmo_toolbuttons->objectName().isEmpty())
            Gizmo_toolbuttons->setObjectName(QStringLiteral("Gizmo_toolbuttons"));
        Gizmo_toolbuttons->resize(198, 38);
        horizontalLayout = new QHBoxLayout(Gizmo_toolbuttons);
        horizontalLayout->setSpacing(0);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, 0, -1, 0);
        toolB_show_gizmo = new QToolButton(Gizmo_toolbuttons);
        toolB_show_gizmo->setObjectName(QStringLiteral("toolB_show_gizmo"));
        toolB_show_gizmo->setMinimumSize(QSize(36, 36));
        toolB_show_gizmo->setIconSize(QSize(32, 32));
        toolB_show_gizmo->setCheckable(true);
        toolB_show_gizmo->setChecked(true);
        toolB_show_gizmo->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_show_gizmo);

        toolB_translate = new QToolButton(Gizmo_toolbuttons);
        toolB_translate->setObjectName(QStringLiteral("toolB_translate"));
        toolB_translate->setMinimumSize(QSize(36, 36));
        toolB_translate->setIconSize(QSize(32, 32));
        toolB_translate->setCheckable(true);
        toolB_translate->setChecked(true);
        toolB_translate->setAutoExclusive(true);
        toolB_translate->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_translate);

        toolB_rotate = new QToolButton(Gizmo_toolbuttons);
        toolB_rotate->setObjectName(QStringLiteral("toolB_rotate"));
        toolB_rotate->setMinimumSize(QSize(36, 36));
        toolB_rotate->setIconSize(QSize(32, 32));
        toolB_rotate->setCheckable(true);
        toolB_rotate->setAutoExclusive(true);
        toolB_rotate->setAutoRaise(true);
        toolB_rotate->setArrowType(Qt::NoArrow);

        horizontalLayout->addWidget(toolB_rotate);

        toolB_trackball = new QToolButton(Gizmo_toolbuttons);
        toolB_trackball->setObjectName(QStringLiteral("toolB_trackball"));
        toolB_trackball->setMinimumSize(QSize(36, 36));
        toolB_trackball->setIconSize(QSize(32, 32));
        toolB_trackball->setCheckable(true);
        toolB_trackball->setAutoExclusive(true);
        toolB_trackball->setAutoRaise(true);
        toolB_trackball->setArrowType(Qt::NoArrow);

        horizontalLayout->addWidget(toolB_trackball);

        toolB_scale = new QToolButton(Gizmo_toolbuttons);
        toolB_scale->setObjectName(QStringLiteral("toolB_scale"));
        toolB_scale->setMinimumSize(QSize(36, 36));
        toolB_scale->setIconSize(QSize(32, 32));
        toolB_scale->setCheckable(true);
        toolB_scale->setAutoExclusive(true);
        toolB_scale->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_scale);


        retranslateUi(Gizmo_toolbuttons);

        QMetaObject::connectSlotsByName(Gizmo_toolbuttons);
    } // setupUi

    void retranslateUi(QWidget *Gizmo_toolbuttons)
    {
        Gizmo_toolbuttons->setWindowTitle(QApplication::translate("Gizmo_toolbuttons", "Form", 0));
#ifndef QT_NO_TOOLTIP
        toolB_show_gizmo->setToolTip(QApplication::translate("Gizmo_toolbuttons", "Show gizmo", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_translate->setToolTip(QApplication::translate("Gizmo_toolbuttons", "Translate", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_rotate->setToolTip(QApplication::translate("Gizmo_toolbuttons", "Rotate", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_trackball->setToolTip(QApplication::translate("Gizmo_toolbuttons", "<html><head/><body><p>Rotate with trackball</p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_scale->setToolTip(QApplication::translate("Gizmo_toolbuttons", "Scale", 0));
#endif // QT_NO_TOOLTIP
    } // retranslateUi

};

namespace Ui {
    class Gizmo_toolbuttons: public Ui_Gizmo_toolbuttons {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_GIZMO_H
