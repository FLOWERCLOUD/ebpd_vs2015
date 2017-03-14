/********************************************************************************
** Form generated from reading UI file 'widget_frame.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WIDGET_FRAME_H
#define UI_WIDGET_FRAME_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Frame_toolbuttons
{
public:
    QHBoxLayout *horizontalLayout;
    QToolButton *toolB_play;
    QToolButton *toolB_stop;
    QToolButton *toolB_prev;
    QToolButton *toolB_next;

    void setupUi(QWidget *Frame_toolbuttons)
    {
        if (Frame_toolbuttons->objectName().isEmpty())
            Frame_toolbuttons->setObjectName(QStringLiteral("Frame_toolbuttons"));
        Frame_toolbuttons->resize(192, 56);
        horizontalLayout = new QHBoxLayout(Frame_toolbuttons);
        horizontalLayout->setSpacing(0);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, 0, -1, 0);
        toolB_play = new QToolButton(Frame_toolbuttons);
        toolB_play->setObjectName(QStringLiteral("toolB_play"));
        toolB_play->setMinimumSize(QSize(36, 36));
        QIcon icon;
        icon.addFile(QStringLiteral("resource/icons/play.png"), QSize(), QIcon::Normal, QIcon::Off);
        toolB_play->setIcon(icon);
        toolB_play->setIconSize(QSize(32, 32));
        toolB_play->setCheckable(true);
        toolB_play->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_play);

        toolB_stop = new QToolButton(Frame_toolbuttons);
        toolB_stop->setObjectName(QStringLiteral("toolB_stop"));
        toolB_stop->setMinimumSize(QSize(36, 36));
        QIcon icon1;
        icon1.addFile(QStringLiteral("resource/icons/stop.png"), QSize(), QIcon::Normal, QIcon::Off);
        toolB_stop->setIcon(icon1);
        toolB_stop->setIconSize(QSize(32, 32));
        toolB_stop->setCheckable(true);
        toolB_stop->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_stop);

        toolB_prev = new QToolButton(Frame_toolbuttons);
        toolB_prev->setObjectName(QStringLiteral("toolB_prev"));
        toolB_prev->setMinimumSize(QSize(36, 36));
        QIcon icon2;
        icon2.addFile(QStringLiteral("resource/icons/backward.png"), QSize(), QIcon::Normal, QIcon::Off);
        toolB_prev->setIcon(icon2);
        toolB_prev->setIconSize(QSize(32, 32));
        toolB_prev->setCheckable(true);
        toolB_prev->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_prev);

        toolB_next = new QToolButton(Frame_toolbuttons);
        toolB_next->setObjectName(QStringLiteral("toolB_next"));
        toolB_next->setMinimumSize(QSize(36, 36));
        QIcon icon3;
        icon3.addFile(QStringLiteral("resource/icons/forward.png"), QSize(), QIcon::Normal, QIcon::Off);
        toolB_next->setIcon(icon3);
        toolB_next->setIconSize(QSize(32, 32));
        toolB_next->setCheckable(true);
        toolB_next->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_next);


        retranslateUi(Frame_toolbuttons);

        QMetaObject::connectSlotsByName(Frame_toolbuttons);
    } // setupUi

    void retranslateUi(QWidget *Frame_toolbuttons)
    {
        Frame_toolbuttons->setWindowTitle(QApplication::translate("Frame_toolbuttons", "Form", 0));
#ifndef QT_NO_TOOLTIP
        toolB_play->setToolTip(QApplication::translate("Frame_toolbuttons", "<html><head/><body><p>Play/pause animation</p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_stop->setToolTip(QApplication::translate("Frame_toolbuttons", "<html><head/><body><p>Stop animation</p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_prev->setToolTip(QApplication::translate("Frame_toolbuttons", "<html><head/><body><p>Previous animation frame</p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_next->setToolTip(QApplication::translate("Frame_toolbuttons", "<html><head/><body><p>Next animation frame</p></body></html>", 0));
#endif // QT_NO_TOOLTIP
    } // retranslateUi

};

namespace Ui {
    class Frame_toolbuttons: public Ui_Frame_toolbuttons {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_FRAME_H
