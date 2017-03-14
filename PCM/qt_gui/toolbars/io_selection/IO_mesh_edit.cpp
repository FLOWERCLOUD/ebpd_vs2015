#include "qt_gui/paint_canvas.h"
#include "IO_mesh_edit.hpp"
#include "global_datas/cuda_globals.hpp"
#include "control/cuda_ctrl.hpp"
#include "qt_gui/main_window.h"
using namespace Tbx;

void IO_mesh_edit::unselect(float x, float y)
{
	Cuda_ctrl::_anim_mesh->unselect(*_cam,
		x, y,
		_gl_widget->get_heuristic(),
		_gl_widget->rest_pose());
}

void IO_mesh_edit::select(float x, float y)
{
	Cuda_ctrl::_anim_mesh->select(*_cam,
		x, y,
		_gl_widget->get_heuristic(),
		_gl_widget->rest_pose());
}

void IO_mesh_edit::paint(int x, int y)
{
	//Widget_painting*  paint_wgt  = _main_win->getUI()->toolBar_painting->_paint_widget;
	//Widget_selection* select_wgt = _main_win->getUI()->toolBar->_wgt_select;
	//Selection_circle<int>* h = (Selection_circle<int>*)_gl_widget->get_heuristic();
	//Animesh::Paint_setup setup;
	//setup._brush_radius = h->_rad;
	//setup._rest_pose = _gl_widget->rest_pose();
	//setup._val = paint_wgt->dSpinB_strength->value();
	//setup._backface_cull = !select_wgt->toolB_backface_select->isChecked();
	//setup._x = x;
	//setup._y = y;
	//g_animesh->paint(_main_win->toolBar_painting->get_paint_mode(), setup, *_cam);
}

bool IO_mesh_edit::is_paint_on()
{
	return _main_win->getUI()->toolBar_painting->is_paint_on();
}

void IO_mesh_edit::keyReleaseEvent(QKeyEvent* event)
{
	IO_skeleton::keyReleaseEvent(event);
}

void IO_mesh_edit::keyPressEvent(QKeyEvent* event)
{
	IO_skeleton::keyPressEvent(event);
	if(event->key() == Qt::Key_Tab){
		_is_edit_on = !_is_edit_on;
		Cuda_ctrl::_anim_mesh->set_display_points(_is_edit_on);
	}
}

void IO_mesh_edit::wheelEvent(QWheelEvent* event)
{
	using namespace Cuda_ctrl;
	if(_gl_widget->get_heuristic()->_type == Selection::CIRCLE &&
		_is_ctrl_pushed)
	{
		float numDegrees = event->delta() / 8.f;
		float numSteps   = numDegrees / 15.f;

		Selection_circle<int>* h = (Selection_circle<int>*)_gl_widget->get_heuristic();
		float tmp = h->_rad + 10*numSteps;
		h->_rad   = tmp < 0.f ? 0.f : tmp;
	}else
		IO_skeleton::wheelEvent(event);
}

void IO_mesh_edit::mouseMoveEvent(QMouseEvent* event)
{
	IO_skeleton::mouseMoveEvent(event);

	const int x = event->x();
	const int y = event->y();
	Vec2i m(x, y);

	if( _is_left_pushed && is_paint_on() && (_old_mouse - m).norm() > 0.5f )
		paint(x, y);

	// TODO: update painted attributes when mouse release

	_old_mouse = m;
}

void IO_mesh_edit::mouseReleaseEvent(QMouseEvent* event)
{
	IO_skeleton::mouseReleaseEvent(event);
}

void IO_mesh_edit::mousePressEvent(QMouseEvent* event)
{
	IO_skeleton::mousePressEvent(event);
	using namespace Cuda_ctrl;

	const int x = event->x();
	const int y = event->y();
	_old_mouse = Vec2i(x, y);

	if(_is_left_pushed && _is_edit_on)
	{
		if(_is_ctrl_pushed)
			// Add to previous selection
				select(x, y);
		else if(_is_maj_pushed)
			// Remove from previous
			unselect(x, y);
		else{
			// Discard old selection and add the new one
			_anim_mesh->reset_selection();
			select(x, y);
		}
	}

	if( _is_left_pushed && is_paint_on() )
		paint(x, y);
	//event->ignore();
}

IO_mesh_edit::~IO_mesh_edit()
{
	Cuda_ctrl::_anim_mesh->reset_selection();
}

IO_mesh_edit::IO_mesh_edit(PaintCanvas* gl_widget) :
	IO_skeleton(gl_widget),
	_is_edit_on(false)
{

}
