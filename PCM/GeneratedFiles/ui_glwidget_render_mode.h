/********************************************************************************
** Form generated from reading UI file 'glwidget_render_mode.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GLWIDGET_RENDER_MODE_H
#define UI_GLWIDGET_RENDER_MODE_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_glwidgetRender_mode_toolbuttons
{
public:
    QHBoxLayout *horizontalLayout;
    QToolButton *toolB_wire_transc;
    QToolButton *toolB_wire;
    QToolButton *toolB_solid;
    QToolButton *toolB_tex;
    QToolButton *toolB_video_background_tex;
    QToolButton *toolB_image_resolution;
    QToolButton *toolB_tex_glwidget_resolution;
    QToolButton *toolB_tex_5;
    QToolButton *toolB_tex_6;
    QSpacerItem *horizontalSpacer;

    void setupUi(QWidget *glwidgetRender_mode_toolbuttons)
    {
        if (glwidgetRender_mode_toolbuttons->objectName().isEmpty())
            glwidgetRender_mode_toolbuttons->setObjectName(QStringLiteral("glwidgetRender_mode_toolbuttons"));
        glwidgetRender_mode_toolbuttons->resize(494, 26);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(glwidgetRender_mode_toolbuttons->sizePolicy().hasHeightForWidth());
        glwidgetRender_mode_toolbuttons->setSizePolicy(sizePolicy);
        horizontalLayout = new QHBoxLayout(glwidgetRender_mode_toolbuttons);
        horizontalLayout->setSpacing(2);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, 2, -1, 2);
        toolB_wire_transc = new QToolButton(glwidgetRender_mode_toolbuttons);
        toolB_wire_transc->setObjectName(QStringLiteral("toolB_wire_transc"));
        toolB_wire_transc->setEnabled(true);
        toolB_wire_transc->setMinimumSize(QSize(18, 18));
        toolB_wire_transc->setIconSize(QSize(16, 16));
        toolB_wire_transc->setCheckable(true);
        toolB_wire_transc->setChecked(true);
        toolB_wire_transc->setAutoExclusive(true);
        toolB_wire_transc->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_wire_transc);

        toolB_wire = new QToolButton(glwidgetRender_mode_toolbuttons);
        toolB_wire->setObjectName(QStringLiteral("toolB_wire"));
        toolB_wire->setMinimumSize(QSize(18, 18));
        toolB_wire->setIconSize(QSize(16, 16));
        toolB_wire->setCheckable(true);
        toolB_wire->setAutoExclusive(true);
        toolB_wire->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_wire);

        toolB_solid = new QToolButton(glwidgetRender_mode_toolbuttons);
        toolB_solid->setObjectName(QStringLiteral("toolB_solid"));
        toolB_solid->setMinimumSize(QSize(18, 18));
        toolB_solid->setIconSize(QSize(16, 16));
        toolB_solid->setCheckable(true);
        toolB_solid->setAutoExclusive(true);
        toolB_solid->setAutoRaise(true);
        toolB_solid->setArrowType(Qt::NoArrow);

        horizontalLayout->addWidget(toolB_solid);

        toolB_tex = new QToolButton(glwidgetRender_mode_toolbuttons);
        toolB_tex->setObjectName(QStringLiteral("toolB_tex"));
        toolB_tex->setMinimumSize(QSize(18, 18));
        toolB_tex->setIconSize(QSize(16, 16));
        toolB_tex->setCheckable(true);
        toolB_tex->setAutoExclusive(true);
        toolB_tex->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_tex);

        toolB_video_background_tex = new QToolButton(glwidgetRender_mode_toolbuttons);
        toolB_video_background_tex->setObjectName(QStringLiteral("toolB_video_background_tex"));
        toolB_video_background_tex->setMinimumSize(QSize(18, 18));
        toolB_video_background_tex->setIconSize(QSize(16, 16));
        toolB_video_background_tex->setCheckable(true);
        toolB_video_background_tex->setAutoExclusive(true);
        toolB_video_background_tex->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_video_background_tex);

        toolB_image_resolution = new QToolButton(glwidgetRender_mode_toolbuttons);
        toolB_image_resolution->setObjectName(QStringLiteral("toolB_image_resolution"));
        toolB_image_resolution->setMinimumSize(QSize(18, 18));
        toolB_image_resolution->setIconSize(QSize(16, 16));
        toolB_image_resolution->setCheckable(true);
        toolB_image_resolution->setAutoExclusive(true);
        toolB_image_resolution->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_image_resolution);

        toolB_tex_glwidget_resolution = new QToolButton(glwidgetRender_mode_toolbuttons);
        toolB_tex_glwidget_resolution->setObjectName(QStringLiteral("toolB_tex_glwidget_resolution"));
        toolB_tex_glwidget_resolution->setMinimumSize(QSize(18, 18));
        toolB_tex_glwidget_resolution->setIconSize(QSize(16, 16));
        toolB_tex_glwidget_resolution->setCheckable(true);
        toolB_tex_glwidget_resolution->setAutoExclusive(true);
        toolB_tex_glwidget_resolution->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_tex_glwidget_resolution);

        toolB_tex_5 = new QToolButton(glwidgetRender_mode_toolbuttons);
        toolB_tex_5->setObjectName(QStringLiteral("toolB_tex_5"));
        toolB_tex_5->setMinimumSize(QSize(18, 18));
        toolB_tex_5->setIconSize(QSize(16, 16));
        toolB_tex_5->setCheckable(true);
        toolB_tex_5->setAutoExclusive(true);
        toolB_tex_5->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_tex_5);

        toolB_tex_6 = new QToolButton(glwidgetRender_mode_toolbuttons);
        toolB_tex_6->setObjectName(QStringLiteral("toolB_tex_6"));
        toolB_tex_6->setMinimumSize(QSize(18, 18));
        toolB_tex_6->setIconSize(QSize(16, 16));
        toolB_tex_6->setCheckable(true);
        toolB_tex_6->setAutoExclusive(true);
        toolB_tex_6->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_tex_6);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);


        retranslateUi(glwidgetRender_mode_toolbuttons);

        QMetaObject::connectSlotsByName(glwidgetRender_mode_toolbuttons);
    } // setupUi

    void retranslateUi(QWidget *glwidgetRender_mode_toolbuttons)
    {
        glwidgetRender_mode_toolbuttons->setWindowTitle(QApplication::translate("glwidgetRender_mode_toolbuttons", "Form", 0));
#ifndef QT_NO_TOOLTIP
        toolB_wire_transc->setToolTip(QApplication::translate("glwidgetRender_mode_toolbuttons", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Render model with translucent faces and wires for edges.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">(Depth peeling)</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_wire->setToolTip(QApplication::translate("glwidgetRender_mode_toolbuttons", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Render model with solid faces and wires for edges.</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_solid->setToolTip(QApplication::translate("glwidgetRender_mode_toolbuttons", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Render model with phong without textures.</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_tex->setToolTip(QApplication::translate("glwidgetRender_mode_toolbuttons", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Render model with phong and textures.</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_video_background_tex->setToolTip(QApplication::translate("glwidgetRender_mode_toolbuttons", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Render model with phong and textures.</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_image_resolution->setToolTip(QApplication::translate("glwidgetRender_mode_toolbuttons", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Render model with phong and textures.</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_tex_glwidget_resolution->setToolTip(QApplication::translate("glwidgetRender_mode_toolbuttons", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Render model with phong and textures.</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_tex_5->setToolTip(QApplication::translate("glwidgetRender_mode_toolbuttons", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Render model with phong and textures.</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_tex_6->setToolTip(QApplication::translate("glwidgetRender_mode_toolbuttons", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Render model with phong and textures.</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
    } // retranslateUi

};

namespace Ui {
    class glwidgetRender_mode_toolbuttons: public Ui_glwidgetRender_mode_toolbuttons {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GLWIDGET_RENDER_MODE_H
