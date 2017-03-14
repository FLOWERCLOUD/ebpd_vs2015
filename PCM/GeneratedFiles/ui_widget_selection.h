/********************************************************************************
** Form generated from reading UI file 'widget_selection.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WIDGET_SELECTION_H
#define UI_WIDGET_SELECTION_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Selection_toolbuttons
{
public:
    QHBoxLayout *horizontalLayout;
    QToolButton *toolB_select_point;
    QToolButton *toolB_select_circle;
    QToolButton *toolB_select_square;
    QToolButton *toolB_select_lasso;
    QToolButton *toolB_backface_select;

    void setupUi(QWidget *Selection_toolbuttons)
    {
        if (Selection_toolbuttons->objectName().isEmpty())
            Selection_toolbuttons->setObjectName(QStringLiteral("Selection_toolbuttons"));
        Selection_toolbuttons->resize(198, 38);
        horizontalLayout = new QHBoxLayout(Selection_toolbuttons);
        horizontalLayout->setSpacing(0);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, 0, -1, 0);
        toolB_select_point = new QToolButton(Selection_toolbuttons);
        toolB_select_point->setObjectName(QStringLiteral("toolB_select_point"));
        toolB_select_point->setMinimumSize(QSize(36, 36));
        toolB_select_point->setIconSize(QSize(32, 32));
        toolB_select_point->setCheckable(true);
        toolB_select_point->setChecked(true);
        toolB_select_point->setAutoExclusive(true);
        toolB_select_point->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_select_point);

        toolB_select_circle = new QToolButton(Selection_toolbuttons);
        toolB_select_circle->setObjectName(QStringLiteral("toolB_select_circle"));
        toolB_select_circle->setMinimumSize(QSize(36, 36));
        toolB_select_circle->setIconSize(QSize(32, 32));
        toolB_select_circle->setCheckable(true);
        toolB_select_circle->setAutoExclusive(true);
        toolB_select_circle->setAutoRaise(true);

        horizontalLayout->addWidget(toolB_select_circle);

        toolB_select_square = new QToolButton(Selection_toolbuttons);
        toolB_select_square->setObjectName(QStringLiteral("toolB_select_square"));
        toolB_select_square->setMinimumSize(QSize(36, 36));
        toolB_select_square->setIconSize(QSize(32, 32));
        toolB_select_square->setCheckable(true);
        toolB_select_square->setAutoExclusive(true);
        toolB_select_square->setAutoRaise(true);
        toolB_select_square->setArrowType(Qt::NoArrow);

        horizontalLayout->addWidget(toolB_select_square);

        toolB_select_lasso = new QToolButton(Selection_toolbuttons);
        toolB_select_lasso->setObjectName(QStringLiteral("toolB_select_lasso"));
        toolB_select_lasso->setMinimumSize(QSize(36, 36));
        toolB_select_lasso->setIconSize(QSize(32, 32));
        toolB_select_lasso->setCheckable(true);
        toolB_select_lasso->setAutoExclusive(true);
        toolB_select_lasso->setAutoRaise(true);
        toolB_select_lasso->setArrowType(Qt::NoArrow);

        horizontalLayout->addWidget(toolB_select_lasso);

        toolB_backface_select = new QToolButton(Selection_toolbuttons);
        toolB_backface_select->setObjectName(QStringLiteral("toolB_backface_select"));
        toolB_backface_select->setMinimumSize(QSize(36, 36));
        toolB_backface_select->setIconSize(QSize(32, 32));
        toolB_backface_select->setCheckable(true);
        toolB_backface_select->setAutoRaise(true);
        toolB_backface_select->setArrowType(Qt::NoArrow);

        horizontalLayout->addWidget(toolB_backface_select);


        retranslateUi(Selection_toolbuttons);

        QMetaObject::connectSlotsByName(Selection_toolbuttons);
    } // setupUi

    void retranslateUi(QWidget *Selection_toolbuttons)
    {
        Selection_toolbuttons->setWindowTitle(QApplication::translate("Selection_toolbuttons", "Form", 0));
#ifndef QT_NO_TOOLTIP
        toolB_select_point->setToolTip(QApplication::translate("Selection_toolbuttons", "Mouse selection", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_select_circle->setToolTip(QApplication::translate("Selection_toolbuttons", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Selection with a circle area</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_select_square->setToolTip(QApplication::translate("Selection_toolbuttons", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Selection with a square area</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_select_lasso->setToolTip(QApplication::translate("Selection_toolbuttons", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Selection with a square area</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        toolB_backface_select->setToolTip(QApplication::translate("Selection_toolbuttons", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'MS Shell Dlg 2'; font-size:8pt;\">Enable backface selection</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
    } // retranslateUi

};

namespace Ui {
    class Selection_toolbuttons: public Ui_Selection_toolbuttons {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_SELECTION_H
