#include "qt_gui/main_window.h"
#include "qt_gui/toolbars/widget_fitting.hpp"
#include <QToolBar>
// -----------------------------------------------------------------------------

void main_window::enable_animesh(bool state)
{
	ui.toolBar->_wgt_fit->setEnabled( state );
	ui.toolBoxMenu->setItemEnabled(5 /* tab debug     */, state);
	ui.toolBoxMenu->setItemEnabled(4 /* tab bone edit */, state);
	ui.toolBoxMenu->setItemEnabled(3 /* tab blending  */, state);
	ui.toolBoxMenu->setItemEnabled(2 /* tab animation */, state);
	ui.box_animesh_color->setEnabled( state );
	ui.box_skeleton->setEnabled( state );
}

// -----------------------------------------------------------------------------

void main_window::enable_mesh(bool state)
{
	ui.box_mesh_color->setEnabled( state );
}

// -----------------------------------------------------------------------------
