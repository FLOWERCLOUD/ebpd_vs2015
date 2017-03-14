/********************************************************************************
** Form generated from reading UI file 'widget_render_mode.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WIDGET_RENDER_MODE_H
#define UI_WIDGET_RENDER_MODE_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Render_mode_toolbuttons
{
public:
    QHBoxLayout *horizontalLayout;
    QToolButton *toolB_wire_transc;
    QToolButton *toolB_wire;
    QToolButton *toolB_solid;
    QToolButton *toolB_tex;

    void setupUi(QWidget *Render_mode_toolbuttons)
    {
        if (Render_mode_toolbuttons->objectName().isEmpty())
            Render_mode_toolbuttons->setObjectName(QStringLiteral("Render_mode_toolbuttons"));
        Render_mode_toolbuttons->resize(174, 38);
        horizontalLayout = new QHBoxLayout(Render_mode_toolbuttons);
        horizontalLayout->setSpacing(0);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, 0, -1, 0);
        toolB_wire_transc = new QToolButton(Render_mode_toolbuttons);
        toolB_wire_transc->setObjectName(QStringLiteral("toolB_wire_transc"));
        toolB_wire_transc->setMinimumSize(QSize(36, 36));
        toolB_wire_transc->setIconSize(QSize(32, 32));
        toolB_wire_transc->setCheckable(true);
        toolB_wire_transc->setChecked(true);
        toolB_wire_transc->setAutoExclusive(true);
        toolB_wire_transc->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_wire_transc);

        toolB_wire = new QToolButton(Render_mode_toolbuttons);
        toolB_wire->setObjectName(QStringLiteral("toolB_wire"));
        toolB_wire->setMinimumSize(QSize(36, 36));
        toolB_wire->setIconSize(QSize(32, 32));
        toolB_wire->setCheckable(true);
        toolB_wire->setAutoExclusive(true);
        toolB_wire->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_wire);

        toolB_solid = new QToolButton(Render_mode_toolbuttons);
        toolB_solid->setObjectName(QStringLiteral("toolB_solid"));
        toolB_solid->setMinimumSize(QSize(36, 36));
        toolB_solid->setIconSize(QSize(32, 32));
        toolB_solid->setCheckable(true);
        toolB_solid->setAutoExclusive(true);
        toolB_solid->setAutoRaise(true);
        toolB_solid->setArrowType(Qt::NoArrow);

        horizontalLayout->addWidget(toolB_solid);

        toolB_tex = new QToolButton(Render_mode_toolbuttons);
        toolB_tex->setObjectName(QStringLiteral("toolB_tex"));
        toolB_tex->setMinimumSize(QSize(36, 36));
        toolB_tex->setIconSize(QSize(32, 32));
        toolB_tex->setCheckable(true);
        toolB_tex->setAutoExclusive(true);
        toolB_tex->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_tex);


        retranslateUi(Render_mode_toolbuttons);

        QMetaObject::connectSlotsByName(Render_mode_toolbuttons);
    } // setupUi

    void retranslateUi(QWidget *Render_mode_toolbuttons)
    {
        Render_mode_toolbuttons->setWindowTitle(QApplication::translate("Render_mode_toolbuttons", "Form", 0));
#ifndef QT_NO_TOOLTIP
        toolB_wire_transc->setToolTip(QApplication::translate("Render_mode_toolbuttons", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Render model with translucent faces and wires for edges.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">(Depth peeling)</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_wire->setToolTip(QApplication::translate("Render_mode_toolbuttons", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Render model with solid faces and wires for edges.</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_solid->setToolTip(QApplication::translate("Render_mode_toolbuttons", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Render model with phong without textures.</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_tex->setToolTip(QApplication::translate("Render_mode_toolbuttons", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Render model with phong and textures.</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
    } // retranslateUi

};

namespace Ui {
    class Render_mode_toolbuttons: public Ui_Render_mode_toolbuttons {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_RENDER_MODE_H
