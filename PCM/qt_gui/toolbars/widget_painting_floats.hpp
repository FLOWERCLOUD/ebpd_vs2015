#ifndef WIDGET_PAINTING_HPP__
#define WIDGET_PAINTING_HPP__

#include "ui_widget_painting_floats.h"

class Widget_painting : public QWidget, public Ui::Painting_floats {
    Q_OBJECT
public:
    Widget_painting(QWidget* parent) : QWidget(parent)
    {
        setupUi(this);
    }
};


#endif // WIDGET_PAINTING_HPP__
