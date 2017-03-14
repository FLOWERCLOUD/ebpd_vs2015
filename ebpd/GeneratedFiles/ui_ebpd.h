/********************************************************************************
** Form generated from reading UI file 'ebpd.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_EBPD_H
#define UI_EBPD_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ebpdClass
{
public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *ebpdClass)
    {
        if (ebpdClass->objectName().isEmpty())
            ebpdClass->setObjectName(QStringLiteral("ebpdClass"));
        ebpdClass->resize(600, 400);
        menuBar = new QMenuBar(ebpdClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        ebpdClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(ebpdClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        ebpdClass->addToolBar(mainToolBar);
        centralWidget = new QWidget(ebpdClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        ebpdClass->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(ebpdClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        ebpdClass->setStatusBar(statusBar);

        retranslateUi(ebpdClass);

        QMetaObject::connectSlotsByName(ebpdClass);
    } // setupUi

    void retranslateUi(QMainWindow *ebpdClass)
    {
        ebpdClass->setWindowTitle(QApplication::translate("ebpdClass", "ebpd", 0));
    } // retranslateUi

};

namespace Ui {
    class ebpdClass: public Ui_ebpdClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_EBPD_H
