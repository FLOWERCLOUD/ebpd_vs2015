/********************************************************************************
** Form generated from reading UI file 'savePlyDialog.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SAVEPLYDIALOG_H
#define UI_SAVEPLYDIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_SavePlyDialog
{
public:
    QVBoxLayout *vboxLayout;
    QHBoxLayout *hboxLayout;
    QLabel *label;
    QLineEdit *outDirLineEdit;
    QPushButton *browseDir;
    QSpacerItem *spacerItem;
    QHBoxLayout *horizontalLayout;
    QLabel *label_2;
    QLineEdit *baseNameLineEdit;
    QSpacerItem *horizontalSpacer_2;
    QGridLayout *gridLayout;
    QCheckBox *checkbox_isadjustview;
    QCheckBox *checkBox_4;
    QCheckBox *checkBox_isoricolor;
    QCheckBox *checkBox_is_labeledcolor;
    QLabel *label_startframe;
    QLineEdit *line_endframe;
    QLineEdit *line_startframe;
    QLabel *label_endframe;
    QSpacerItem *horizontalSpacer;
    QComboBox *comboBox;
    QLabel *label_3;
    QHBoxLayout *hboxLayout1;
    QSpacerItem *spacerItem1;
    QPushButton *cancelButton;
    QPushButton *saveButton;

    void setupUi(QDialog *SavePlyDialog)
    {
        if (SavePlyDialog->objectName().isEmpty())
            SavePlyDialog->setObjectName(QStringLiteral("SavePlyDialog"));
        SavePlyDialog->resize(701, 387);
        vboxLayout = new QVBoxLayout(SavePlyDialog);
        vboxLayout->setSpacing(6);
        vboxLayout->setContentsMargins(11, 11, 11, 11);
        vboxLayout->setObjectName(QStringLiteral("vboxLayout"));
        hboxLayout = new QHBoxLayout();
        hboxLayout->setSpacing(6);
        hboxLayout->setObjectName(QStringLiteral("hboxLayout"));
        label = new QLabel(SavePlyDialog);
        label->setObjectName(QStringLiteral("label"));

        hboxLayout->addWidget(label);

        outDirLineEdit = new QLineEdit(SavePlyDialog);
        outDirLineEdit->setObjectName(QStringLiteral("outDirLineEdit"));

        hboxLayout->addWidget(outDirLineEdit);

        browseDir = new QPushButton(SavePlyDialog);
        browseDir->setObjectName(QStringLiteral("browseDir"));
        browseDir->setMinimumSize(QSize(20, 20));
        browseDir->setMaximumSize(QSize(20, 20));

        hboxLayout->addWidget(browseDir);

        spacerItem = new QSpacerItem(16, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout->addItem(spacerItem);


        vboxLayout->addLayout(hboxLayout);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        label_2 = new QLabel(SavePlyDialog);
        label_2->setObjectName(QStringLiteral("label_2"));

        horizontalLayout->addWidget(label_2);

        baseNameLineEdit = new QLineEdit(SavePlyDialog);
        baseNameLineEdit->setObjectName(QStringLiteral("baseNameLineEdit"));

        horizontalLayout->addWidget(baseNameLineEdit);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);


        vboxLayout->addLayout(horizontalLayout);

        gridLayout = new QGridLayout();
        gridLayout->setSpacing(6);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        checkbox_isadjustview = new QCheckBox(SavePlyDialog);
        checkbox_isadjustview->setObjectName(QStringLiteral("checkbox_isadjustview"));

        gridLayout->addWidget(checkbox_isadjustview, 1, 0, 1, 1);

        checkBox_4 = new QCheckBox(SavePlyDialog);
        checkBox_4->setObjectName(QStringLiteral("checkBox_4"));

        gridLayout->addWidget(checkBox_4, 5, 0, 1, 1);

        checkBox_isoricolor = new QCheckBox(SavePlyDialog);
        checkBox_isoricolor->setObjectName(QStringLiteral("checkBox_isoricolor"));

        gridLayout->addWidget(checkBox_isoricolor, 3, 0, 1, 1);

        checkBox_is_labeledcolor = new QCheckBox(SavePlyDialog);
        checkBox_is_labeledcolor->setObjectName(QStringLiteral("checkBox_is_labeledcolor"));

        gridLayout->addWidget(checkBox_is_labeledcolor, 4, 0, 1, 1);

        label_startframe = new QLabel(SavePlyDialog);
        label_startframe->setObjectName(QStringLiteral("label_startframe"));

        gridLayout->addWidget(label_startframe, 1, 1, 1, 1);

        line_endframe = new QLineEdit(SavePlyDialog);
        line_endframe->setObjectName(QStringLiteral("line_endframe"));

        gridLayout->addWidget(line_endframe, 1, 5, 1, 1);

        line_startframe = new QLineEdit(SavePlyDialog);
        line_startframe->setObjectName(QStringLiteral("line_startframe"));

        gridLayout->addWidget(line_startframe, 1, 3, 1, 1);

        label_endframe = new QLabel(SavePlyDialog);
        label_endframe->setObjectName(QStringLiteral("label_endframe"));

        gridLayout->addWidget(label_endframe, 1, 4, 1, 1);

        horizontalSpacer = new QSpacerItem(60, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer, 1, 6, 1, 1);

        comboBox = new QComboBox(SavePlyDialog);
        comboBox->setObjectName(QStringLiteral("comboBox"));

        gridLayout->addWidget(comboBox, 3, 3, 1, 1);

        label_3 = new QLabel(SavePlyDialog);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout->addWidget(label_3, 3, 1, 1, 1);


        vboxLayout->addLayout(gridLayout);

        hboxLayout1 = new QHBoxLayout();
        hboxLayout1->setSpacing(6);
        hboxLayout1->setObjectName(QStringLiteral("hboxLayout1"));
        spacerItem1 = new QSpacerItem(51, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout1->addItem(spacerItem1);

        cancelButton = new QPushButton(SavePlyDialog);
        cancelButton->setObjectName(QStringLiteral("cancelButton"));
        cancelButton->setMinimumSize(QSize(0, 25));
        cancelButton->setMaximumSize(QSize(16777215, 25));

        hboxLayout1->addWidget(cancelButton);

        saveButton = new QPushButton(SavePlyDialog);
        saveButton->setObjectName(QStringLiteral("saveButton"));
        saveButton->setMinimumSize(QSize(0, 25));
        saveButton->setMaximumSize(QSize(16777215, 25));
        saveButton->setDefault(true);

        hboxLayout1->addWidget(saveButton);


        vboxLayout->addLayout(hboxLayout1);


        retranslateUi(SavePlyDialog);

        QMetaObject::connectSlotsByName(SavePlyDialog);
    } // setupUi

    void retranslateUi(QDialog *SavePlyDialog)
    {
        SavePlyDialog->setWindowTitle(QApplication::translate("SavePlyDialog", "Save PLY", 0));
        label->setText(QApplication::translate("SavePlyDialog", "  Output folder  ", 0));
        browseDir->setText(QApplication::translate("SavePlyDialog", "...", 0));
        label_2->setText(QApplication::translate("SavePlyDialog", "  basename  ", 0));
        checkbox_isadjustview->setText(QApplication::translate("SavePlyDialog", "adjustview", 0));
        checkBox_4->setText(QApplication::translate("SavePlyDialog", "CheckBox", 0));
        checkBox_isoricolor->setText(QApplication::translate("SavePlyDialog", "oricor", 0));
        checkBox_is_labeledcolor->setText(QApplication::translate("SavePlyDialog", "curcolor(labeled)", 0));
        label_startframe->setText(QApplication::translate("SavePlyDialog", "start frame", 0));
        label_endframe->setText(QApplication::translate("SavePlyDialog", "end frame", 0));
        comboBox->clear();
        comboBox->insertItems(0, QStringList()
         << QApplication::translate("SavePlyDialog", "ply", 0)
         << QApplication::translate("SavePlyDialog", "xyz", 0)
         << QApplication::translate("SavePlyDialog", "obj", 0)
        );
        label_3->setText(QApplication::translate("SavePlyDialog", "output format", 0));
        cancelButton->setText(QApplication::translate("SavePlyDialog", "Cancel", 0));
        saveButton->setText(QApplication::translate("SavePlyDialog", "Save", 0));
    } // retranslateUi

};

namespace Ui {
    class SavePlyDialog: public Ui_SavePlyDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SAVEPLYDIALOG_H
