#ifndef WIDGET_FRAME_HPP__
#define WIDGET_FRAME_HPP__

#include "ui_widget_frame.h"

class Widget_frame : public QWidget, public Ui::Frame_toolbuttons {
    Q_OBJECT
public:
    Widget_frame(QWidget* parent) : QWidget(parent) {
        setupUi(this);
    }

private slots:
    void on_toolB_prev_pressed(){ toolB_prev->setChecked(true); }
    void on_toolB_next_pressed(){ toolB_next->setChecked(true); }
    void on_toolB_play_pressed(){ toolB_play->setChecked(true); }
    void on_toolB_stop_pressed(){ toolB_stop->setChecked(true); }
};

#endif // WIDGET_FRAME_HPP__
