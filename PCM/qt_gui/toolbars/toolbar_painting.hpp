#ifndef TOOLBAR_PAINTING_HPP__
#define TOOLBAR_PAINTING_HPP__

#include <QToolBar>
#include <QComboBox>
#include <QCheckBox>

#include "animesh_enum.hpp"
class 	Widget_painting;
class Toolbar_painting : public QToolBar {
    Q_OBJECT

public:
    Toolbar_painting(QWidget *parent);
    ~Toolbar_painting();

    bool is_paint_on() const { return _enable_paint->isChecked(); }

    EAnimesh::Paint_type get_paint_mode() {
        return (EAnimesh::Paint_type)_paint_mode_comboBox->itemData( _paint_mode_comboBox->currentIndex() ).toInt();
    }

    QCheckBox* _enable_paint;
    Widget_painting* _paint_widget;

public slots:

private:
    QComboBox* _paint_mode_comboBox;
};




#endif // TOOLBAR_PAINTING_HPP__
