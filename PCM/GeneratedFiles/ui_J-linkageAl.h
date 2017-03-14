/********************************************************************************
** Form generated from reading UI file 'J-linkageAl.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_J_2D_LINKAGEAL_H
#define UI_J_2D_LINKAGEAL_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Form
{
public:
    QLineEdit *TrajLen;
    QPushButton *pushButton;
    QLabel *TriLen;

    void setupUi(QWidget *Form)
    {
        if (Form->objectName().isEmpty())
            Form->setObjectName(QStringLiteral("Form"));
        Form->resize(392, 330);
        TrajLen = new QLineEdit(Form);
        TrajLen->setObjectName(QStringLiteral("TrajLen"));
        TrajLen->setGeometry(QRect(140, 80, 113, 20));
        pushButton = new QPushButton(Form);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        pushButton->setGeometry(QRect(160, 210, 75, 23));
        TriLen = new QLabel(Form);
        TriLen->setObjectName(QStringLiteral("TriLen"));
        TriLen->setGeometry(QRect(30, 80, 54, 12));

        retranslateUi(Form);

        QMetaObject::connectSlotsByName(Form);
    } // setupUi

    void retranslateUi(QWidget *Form)
    {
        Form->setWindowTitle(QApplication::translate("Form", "Form", 0));
        pushButton->setText(QApplication::translate("Form", "computer", 0));
        TriLen->setText(QApplication::translate("Form", "TextLabel", 0));
    } // retranslateUi

};

namespace Ui {
    class Form: public Ui_Form {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_J_2D_LINKAGEAL_H
