#ifndef WIDGET_RENDER_MODE_HPP__
#define WIDGET_RENDER_MODE_HPP__

#include "ui_widget_render_mode.h"
#include "global_datas/g_paths.hpp"

class Widget_render_mode : public QWidget, public Ui::Render_mode_toolbuttons {
    Q_OBJECT
public:
    Widget_render_mode(QWidget* parent) : QWidget(parent)
    {
        setupUi(this);

        QIcon icon( (g_icons_theme_dir+"/wireframe_transparent.svg").c_str() );
        toolB_wire_transc->setIcon(icon);

        QIcon icon1( (g_icons_theme_dir+"/wireframe.svg").c_str() );
        toolB_wire->setIcon(icon1);

        QIcon icon2( (g_icons_theme_dir+"/solid.svg").c_str() );
        toolB_solid->setIcon(icon2);

        QIcon icon3( (g_icons_theme_dir+"/texture.svg").c_str() );
        toolB_tex->setIcon(icon3);
    }

};


#endif // WIDGET_RENDER_MODE_HPP__
