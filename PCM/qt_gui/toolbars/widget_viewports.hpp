#ifndef WIDGET_VIEWPORTS_HPP__
#define WIDGET_VIEWPORTS_HPP__

#include "ui_widget_viewports.h"
#include "global_datas/g_paths.hpp"

class Widget_viewports : public QWidget, public Ui::Viewports_toolbuttons {
    Q_OBJECT
public:
    Widget_viewports(QWidget* parent) : QWidget(parent)
    {
        setupUi(this);

        QIcon icon( (g_icons_theme_dir+"/viewport_single.svg").c_str() );
        toolB_single->setIcon(icon);

        QIcon icon1( (g_icons_theme_dir+"/viewport_double_horz.svg").c_str() );
        toolB_doubleH->setIcon(icon1);

        QIcon icon2( (g_icons_theme_dir+"/viewport_double_vert.svg").c_str() );
        toolB_doubleV->setIcon(icon2);

        QIcon icon3( (g_icons_theme_dir+"/viewport_four.svg").c_str() );
        toolB_four->setIcon(icon3);

    }
};


#endif // WIDGET_VIEWPORTS_HPP__
