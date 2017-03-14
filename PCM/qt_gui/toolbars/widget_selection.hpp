#ifndef WIDGET_SELECTION_HPP__
#define WIDGET_SELECTION_HPP__

#include "ui_widget_selection.h"
#include "global_datas/g_paths.hpp"

class Widget_selection : public QWidget, public Ui::Selection_toolbuttons {
    Q_OBJECT
public:
    Widget_selection(QWidget* parent) : QWidget(parent)
    {
        setupUi(this);

        QIcon icon( (g_icons_theme_dir+"/select.svg").c_str() );
        toolB_select_point->setIcon(icon);

        QIcon icon1( (g_icons_theme_dir+"/select_circle.svg").c_str() );
        toolB_select_circle->setIcon(icon1);

        QIcon icon2( (g_icons_theme_dir+"/select_square.svg").c_str() );
        toolB_select_square->setIcon(icon2);

        QIcon icon3( (g_icons_theme_dir+"/lasso.svg").c_str() );
        toolB_select_lasso->setIcon(icon3);

        QIcon icon4( (g_icons_theme_dir+"/hidden_faces.svg").c_str() );
        toolB_backface_select->setIcon(icon4);
    }

};


#endif // WIDGET_SELECTION_HPP__
