#ifndef WIDGET_FITTING_HPP__
#define WIDGET_FITTING_HPP__

#include <QIcon>
#include "ui_widget_fitting.h"
#include "global_datas/g_paths.hpp"

class Widget_fitting : public QWidget, public Ui::Fitting_toolbuttons {
    Q_OBJECT
public:
    Widget_fitting(QWidget* parent) : QWidget(parent)
    {
        setupUi(this);
        QIcon icon( (g_icons_theme_dir+"/fitting.svg").c_str() );
        toolB_fitting->setIcon( icon );
    }
};


#endif // WIDGET_FITTING_HPP__
