/********************************************************************************
** Form generated from reading UI file 'testUi.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TESTUI_H
#define UI_TESTUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Form
{
public:
    QHBoxLayout *horizontalLayout;
    QSplitter *splitter;
    QPushButton *pushButton;
    QPushButton *pushButton_3;
    QPushButton *pushButton_2;
    QFrame *frame;
    QHBoxLayout *horizontalLayout_2;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout;
    QLabel *label;
    QPushButton *pushButton_4;
    QVBoxLayout *verticalLayout_2;
    QPushButton *pushButton_5;
    QRadioButton *radioButton;

    void setupUi(QWidget *Form)
    {
        if (Form->objectName().isEmpty())
            Form->setObjectName(QStringLiteral("Form"));
        Form->resize(710, 554);
        horizontalLayout = new QHBoxLayout(Form);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        splitter = new QSplitter(Form);
        splitter->setObjectName(QStringLiteral("splitter"));
        splitter->setOrientation(Qt::Vertical);
        pushButton = new QPushButton(splitter);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        splitter->addWidget(pushButton);
        pushButton_3 = new QPushButton(splitter);
        pushButton_3->setObjectName(QStringLiteral("pushButton_3"));
        splitter->addWidget(pushButton_3);
        pushButton_2 = new QPushButton(splitter);
        pushButton_2->setObjectName(QStringLiteral("pushButton_2"));
        splitter->addWidget(pushButton_2);

        horizontalLayout->addWidget(splitter);

        frame = new QFrame(Form);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        horizontalLayout_2 = new QHBoxLayout(frame);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        groupBox = new QGroupBox(frame);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        verticalLayout = new QVBoxLayout(groupBox);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        label = new QLabel(groupBox);
        label->setObjectName(QStringLiteral("label"));
        label->setPixmap(QPixmap(QString::fromUtf8("C:/Users/hehua2015/Pictures/500abe03e3e61.jpg")));

        verticalLayout->addWidget(label);

        pushButton_4 = new QPushButton(groupBox);
        pushButton_4->setObjectName(QStringLiteral("pushButton_4"));

        verticalLayout->addWidget(pushButton_4);


        horizontalLayout_2->addWidget(groupBox);


        horizontalLayout->addWidget(frame);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        pushButton_5 = new QPushButton(Form);
        pushButton_5->setObjectName(QStringLiteral("pushButton_5"));

        verticalLayout_2->addWidget(pushButton_5);

        radioButton = new QRadioButton(Form);
        radioButton->setObjectName(QStringLiteral("radioButton"));

        verticalLayout_2->addWidget(radioButton);


        horizontalLayout->addLayout(verticalLayout_2);


        retranslateUi(Form);

        QMetaObject::connectSlotsByName(Form);
    } // setupUi

    void retranslateUi(QWidget *Form)
    {
        Form->setWindowTitle(QApplication::translate("Form", "Form", 0));
        pushButton->setText(QApplication::translate("Form", "PushButton", 0));
        pushButton_3->setText(QApplication::translate("Form", "PushButton", 0));
        pushButton_2->setText(QApplication::translate("Form", "PushButton", 0));
        groupBox->setTitle(QApplication::translate("Form", "GroupBox", 0));
        label->setText(QString());
        pushButton_4->setText(QApplication::translate("Form", "PushButton", 0));
        pushButton_5->setText(QApplication::translate("Form", "PushButton", 0));
        radioButton->setText(QApplication::translate("Form", "RadioButton", 0));
    } // retranslateUi

};

namespace Ui {
    class Form: public Ui_Form {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TESTUI_H
