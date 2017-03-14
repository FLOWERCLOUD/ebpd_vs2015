#include "toolbar_painting.hpp"
#include "animesh_enum.hpp"
#include "widget_painting_floats.hpp"
#include <QLayout>

Toolbar_painting::Toolbar_painting(QWidget *parent) :
    QToolBar(parent)

{
    //Setup layout
    QLayout* layout = this->layout();
    layout->setSpacing(6);
    layout->setObjectName(QString::fromUtf8("_hLayout"));
    //_hLayout->setContentsMargins(-1, 0, -1, 0);
    // -----------------

    _enable_paint = new QCheckBox(this);
    _enable_paint->setText("Enable painting");
    this->addWidget(_enable_paint);

    _paint_mode_comboBox = new QComboBox( this );
    _paint_mode_comboBox->addItem("Implicit skin", (int)EAnimesh::PT_SSD_INTERPOLATION);
    _paint_mode_comboBox->addItem("Cluster"      , (int)EAnimesh::PT_CLUSTER          );
    _paint_mode_comboBox->addItem("Bones weights", (int)EAnimesh::PT_SSD_WEIGHTS      );
    this->addWidget(_paint_mode_comboBox);

    _paint_widget = new Widget_painting(this);
    this->addWidget(_paint_widget);
}

Toolbar_painting::~Toolbar_painting()
{

}
