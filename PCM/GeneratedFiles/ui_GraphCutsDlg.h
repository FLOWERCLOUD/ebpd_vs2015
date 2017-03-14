/********************************************************************************
** Form generated from reading UI file 'GraphCutsDlg.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GRAPHCUTSDLG_H
#define UI_GRAPHCUTSDLG_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>

QT_BEGIN_NAMESPACE

class Ui_GraphCuts
{
public:
    QPushButton *okButton;
    QLabel *label;
    QLineEdit *nLabels;
    QLabel *label_2;
    QLabel *label_3;
    QLabel *label_4;
    QLabel *label_5;
    QLabel *label_6;
    QLineEdit *nGraph;
    QLineEdit *nCurv;
    QLineEdit *Expansion;
    QLineEdit *Swap;
    QLineEdit *TradeOff;

    void setupUi(QDialog *GraphCuts)
    {
        if (GraphCuts->objectName().isEmpty())
            GraphCuts->setObjectName(QStringLiteral("GraphCuts"));
        GraphCuts->resize(471, 357);
        okButton = new QPushButton(GraphCuts);
        okButton->setObjectName(QStringLiteral("okButton"));
        okButton->setGeometry(QRect(300, 330, 161, 23));
        label = new QLabel(GraphCuts);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(20, 80, 61, 16));
        nLabels = new QLineEdit(GraphCuts);
        nLabels->setObjectName(QStringLiteral("nLabels"));
        nLabels->setGeometry(QRect(150, 80, 61, 20));
        label_2 = new QLabel(GraphCuts);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(20, 120, 71, 16));
        label_3 = new QLabel(GraphCuts);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(20, 150, 101, 16));
        label_4 = new QLabel(GraphCuts);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setGeometry(QRect(20, 180, 61, 21));
        label_5 = new QLabel(GraphCuts);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setGeometry(QRect(20, 210, 71, 31));
        label_6 = new QLabel(GraphCuts);
        label_6->setObjectName(QStringLiteral("label_6"));
        label_6->setGeometry(QRect(20, 250, 61, 21));
        nGraph = new QLineEdit(GraphCuts);
        nGraph->setObjectName(QStringLiteral("nGraph"));
        nGraph->setGeometry(QRect(150, 120, 61, 20));
        nCurv = new QLineEdit(GraphCuts);
        nCurv->setObjectName(QStringLiteral("nCurv"));
        nCurv->setGeometry(QRect(150, 150, 61, 20));
        Expansion = new QLineEdit(GraphCuts);
        Expansion->setObjectName(QStringLiteral("Expansion"));
        Expansion->setGeometry(QRect(150, 180, 61, 20));
        Swap = new QLineEdit(GraphCuts);
        Swap->setObjectName(QStringLiteral("Swap"));
        Swap->setGeometry(QRect(150, 220, 61, 20));
        TradeOff = new QLineEdit(GraphCuts);
        TradeOff->setObjectName(QStringLiteral("TradeOff"));
        TradeOff->setGeometry(QRect(150, 250, 61, 20));

        retranslateUi(GraphCuts);
        QObject::connect(okButton, SIGNAL(clicked()), GraphCuts, SLOT(accept()));

        QMetaObject::connectSlotsByName(GraphCuts);
    } // setupUi

    void retranslateUi(QDialog *GraphCuts)
    {
        GraphCuts->setWindowTitle(QApplication::translate("GraphCuts", "Dialog", 0));
        okButton->setText(QApplication::translate("GraphCuts", "Single frame refine", 0));
        label->setText(QApplication::translate("GraphCuts", "\345\210\235\345\247\213\347\261\273\344\270\252\346\225\260", 0));
        label_2->setText(QApplication::translate("GraphCuts", "Graph \351\202\273\345\237\237", 0));
        label_3->setText(QApplication::translate("GraphCuts", "Curvature \351\202\273\345\237\237", 0));
        label_4->setText(QApplication::translate("GraphCuts", "Expansion", 0));
        label_5->setText(QApplication::translate("GraphCuts", "Swap", 0));
        label_6->setText(QApplication::translate("GraphCuts", "\345\271\263\346\273\221\351\241\271\346\235\203\351\207\215", 0));
    } // retranslateUi

};

namespace Ui {
    class GraphCuts: public Ui_GraphCuts {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GRAPHCUTSDLG_H
