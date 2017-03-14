#ifndef WIDGET_GIZMO_HPP__
#define WIDGET_GIZMO_HPP__

#include "ui_widget_gizmo.h"
#include "global_datas/g_paths.hpp"

class Widget_gizmo : public QWidget, public Ui::Gizmo_toolbuttons {
    Q_OBJECT
public:
    Widget_gizmo(QWidget* parent) : QWidget(parent)
    {
        setupUi(this);

        // Setup icons
        QIcon icon( (g_icons_dir+"/gizmo.svg").c_str() );
        toolB_show_gizmo->setIcon(icon);

        QIcon icon1( (g_icons_theme_dir+"/translate.svg").c_str() );
        toolB_translate->setIcon(icon1);

        QIcon icon2( (g_icons_theme_dir+"/rotate.svg").c_str() );
        toolB_rotate->setIcon(icon2);

        QIcon icon3( (g_icons_theme_dir+"/trackball.svg").c_str() );
        toolB_trackball->setIcon(icon3);

        QIcon icon4( (g_icons_theme_dir+"/scale.svg").c_str() );
        toolB_scale->setIcon(icon4);
    }

};


#endif // WIDGET_GIZMO_HPP__
