#include "IO_interface_skin.hpp"
#include "../../animation/animesh.hpp"
#include "../../animation/skeleton.hpp"
#include "../../control/cuda_ctrl.hpp"
#include "../../qt_gui/paint_canvas.h"
#include "../../GlobalObject.h"
#include "../../rendering/camera.hpp"
#include "qt_gui/main_window.h"
extern Skeleton* g_skel;
extern Animesh* g_animesh;
// -------

extern bool g_shooting_state;
extern bool g_save_anim;
using namespace Tbx;

IO_interface_skin::IO_interface_skin(PaintCanvas* gl_widget) :
	_is_ctrl_pushed(false),
	_is_alt_pushed(false),
	_is_tab_pushed(false),
	_is_maj_pushed(false),
	_is_space_pushed(false),
	_is_right_pushed(false),
	_is_left_pushed(false),
	_is_mid_pushed(false),
	_is_gizmo_grabed(false),
	_movement_speed(1.f),
	_rot_speed(0.01f),
	_gizmo_tr(),
	_gl_widget(gl_widget),

	_main_win( Global_Window)
{
		_cam =  new Tbx::Camera( gl_widget->camera());
}

void IO_interface_skin::update_gl_matrix()
{
	glGetIntegerv(GL_VIEWPORT, _viewport);
	glGetDoublev(GL_MODELVIEW_MATRIX,  _modelview);
	glGetDoublev(GL_PROJECTION_MATRIX, _projection);
}

Kinematic* IO_interface_skin::kinec()
{
	return g_skel->_kinec;
}

Gizmo* IO_interface_skin::gizmo()
{
	return _gl_widget->gizmo();
}

void IO_interface_skin::update_frame_gizmo()
{

}

void IO_interface_skin::keyReleaseEvent(QKeyEvent* event)
{
	// Check special keys
	switch( event->key() ){
	case Qt::Key_Control: _is_ctrl_pushed = false; break;
	case Qt::Key_Alt:     _is_alt_pushed  = false; break;
	case Qt::Key_Tab:     _is_tab_pushed  = false; break;
	case Qt::Key_Shift:   _is_maj_pushed  = false; break;
	case Qt::Key_Space:   _is_space_pushed= false; break;
	}
}

void IO_interface_skin::keyPressEvent(QKeyEvent* event)
{
	using namespace Cuda_ctrl;

	// Check special keys
	switch( event->key() ){
	case Qt::Key_Control: _is_ctrl_pushed = true; break;
	case Qt::Key_Alt:     _is_alt_pushed  = true; break;
	case Qt::Key_Tab:     _is_tab_pushed  = true; break;
	case Qt::Key_Shift:   _is_maj_pushed  = true; break;
	case Qt::Key_Space:   _is_space_pushed= true; break;
	case Qt::Key_Up :
		_cam->update_pos(Tbx::Vec3(0.f, 0.f,  2.*_movement_speed*1.f));
		break;
	case Qt::Key_Down :
		_cam->update_pos(Tbx::Vec3(0.f, 0.f, -2.*_movement_speed* 1.f));
		break;
	case Qt::Key_Left : //screenshot();
		break;
	}

	const std::vector<int>& set = Cuda_ctrl::_skeleton.get_selection_set();

	//IBL::Ctrl_shape shape;
	//if( set.size() > 0 )
	//	shape = Cuda_ctrl::_skeleton.get_joint_controller(set[set.size()-1]);
	//else
	//	shape = Blending_env::get_global_ctrl_shape();


	// Check standard characters :
	QString t = event->text();
	QChar c = t[0];

	bool view_changed = true;
	switch (c.toLatin1()) {
		// ''''''''''''''''''''''''''''''''''''''''''''
		// Change operator parameters :

		// Change abscissa :
	//case 'E': shape.set_p0( increase(shape.p0(),  0.1f, true) );break;
	//case 'e': shape.set_p0( increase(shape.p0(), -0.1f, true) );break;
	//case 'Z': shape.set_p1( increase(shape.p1(),  0.1f, true) );break;
	//case 'z': shape.set_p1( increase(shape.p1(), -0.1f, true) );break;
	//case 'A': shape.set_p2( increase(shape.p2(),  0.1f, true) );break;
	//case 'a': shape.set_p2( increase(shape.p2(), -0.1f, true) );break;
	//	// Change ordinate :
	//case 'D': shape.set_p0( increase(shape.p0(),  0.02f, false) );break;
	//case 'd': shape.set_p0( increase(shape.p0(), -0.02f, false) );break;
	//case 'S': shape.set_p1( increase(shape.p1(),  0.02f, false) );break;
	//case 's': shape.set_p1( increase(shape.p1(), -0.02f, false) );break;
	//case 'Q': shape.set_p2( increase(shape.p2(),  0.02f, false) );break;
	//case 'q': shape.set_p2( increase(shape.p2(), -0.02f, false) );break;
	//	// Change Stifness
	//case 'X': shape.set_s0( shape.s0() +  0.2f );break;
	//case 'x': shape.set_s0( shape.s0() + -0.2f );break;
	//case 'W': shape.set_s1( shape.s1() +  0.2f );break;
	//case 'w': shape.set_s1( shape.s1() + -0.2f );break;
		// ''''''''''''''''''''''''''''''''''''''''''''

		// OTHER STUFF
	case 'u' : _anim_mesh->update_base_potential(); push_msge("update base potential"); break;
		// Camera angles :
	case '8': top_view(); push_msge("top");       break;
	case '2': bottom_view(); push_msge("bottom"); break;
	case '6': right_view(); push_msge("right");   break;
	case '4': left_view(); push_msge("left");     break;
	case '1': front_view(); push_msge("front");   break;
	case '3': rear_view(); push_msge("rear");     break;
	case '5': _cam->set_ortho(!_cam->is_ortho()); break;

	default: view_changed = false; break;
	}
	// This updates the raytracing drawing and sets back _raytrace_again to false
	Cuda_ctrl::_display._raytrace_again = (view_changed || Cuda_ctrl::_display._raytrace_again);

	//// Update the controler function
	//if(set.size() > 0)
	//	Cuda_ctrl::_skeleton.set_joint_controller(set[set.size()-1], shape);
	//else
	//	Blending_env::set_global_ctrl_shape(shape);

	//_main_win->update_ctrl_spin_boxes(shape);

	//Cuda_ctrl::_operators.update_displayed_operator_texture();

	// Ok this is a bit hacky but it enables main window to see keypress events
	// and handle them. Be aware that a conflict will appear if the same shortcut
	// is used here and inside MainWindow.
	event->ignore();
}

Vec2 IO_interface_skin::increase(Vec2 val, float incr, bool t)
{
	if(t) val.x += incr;
	else  val.y += incr;
	return val;
}

void IO_interface_skin::rear_view()
{
	_cam->set_dir_and_up(Tbx::Vec3::unit_z(),
		Tbx::Vec3::unit_y());

	Tbx::Vec3 p = _cam->get_pos();
	Tbx::Vec3 v = _gl_widget->pivot();
	_cam->set_pos( Tbx::Vec3(v.x, v.y, v.z - (p-v).norm()) );
}

void IO_interface_skin::front_view()
{
	_cam->set_dir_and_up(-Tbx::Vec3::unit_z(),
		Tbx::Vec3::unit_y());

	Tbx::Vec3 p = _cam->get_pos();
	Tbx::Vec3 v = _gl_widget->pivot();
	_cam->set_pos( Tbx::Vec3(v.x, v.y, v.z + (p-v).norm()) );
}

void IO_interface_skin::bottom_view()
{
	_cam->set_dir_and_up(Tbx::Vec3::unit_y(),
		Tbx::Vec3::unit_z());

	Tbx::Vec3 p = _cam->get_pos();
	Tbx::Vec3 v = _gl_widget->pivot();
	_cam->set_pos( Tbx::Vec3(v.x, v.y - (p-v).norm(), v.z) );
}

void IO_interface_skin::top_view()
{
	_cam->set_dir_and_up(-Tbx::Vec3::unit_y(),
		-Tbx::Vec3::unit_z());

	Tbx::Vec3 p = _cam->get_pos();
	Tbx::Vec3 v = _gl_widget->pivot();
	_cam->set_pos( Tbx::Vec3(v.x, v.y + (p-v).norm(), v.z) );
}

void IO_interface_skin::left_view()
{
	_cam->set_dir_and_up(Tbx::Vec3::unit_x(),
		Tbx::Vec3::unit_y());

	Tbx::Vec3 p = _cam->get_pos();
	Tbx::Vec3 v = _gl_widget->pivot();
	_cam->set_pos( Tbx::Vec3(v.x - (p-v).norm(), v.y, v.z) );
}

void IO_interface_skin::right_view()
{
	_cam->set_dir_and_up( -Tbx::Vec3::unit_x(),
		Tbx::Vec3::unit_y() );

	Tbx::Vec3 p = _cam->get_pos();
	Tbx::Vec3 v = _gl_widget->pivot();
	_cam->set_pos( Tbx::Vec3( v.x + (p-v).norm(), v.y, v.z) );
}

void IO_interface_skin::wheelEvent(QWheelEvent* event)
{
	float numDegrees = event->delta() / 8.;
	float numSteps = numDegrees / 15.;

	if(event->buttons() == Qt::NoButton )
	{
		float sign  = numSteps > 0 ? 1.f : -1.f;
		float width = _cam->get_ortho_zoom();
		float new_width = std::max(width - sign * _movement_speed, 0.3f);

		if(_cam->is_ortho())
			_cam->set_ortho_zoom(new_width);
		else
			_cam->update_pos(Tbx::Vec3(0.f, 0.f,  _movement_speed*1.f*sign));
	}
}

void IO_interface_skin::mouseMoveEvent(QMouseEvent* event)
{
	using namespace Cuda_ctrl;
	const int x = event->x();
	const int y = event->y();

	_gizmo_tr = gizmo()->slide(*_cam, x, y);

	if(_is_right_pushed && 0 )
	{
		int dx = x - _cam_old_x;
		int dy = _cam->height() - y - _cam_old_y;
		//rotation around x axis (to manage the vertical angle of vision)
		const float pitch = dy * _rot_speed / M_PI;
		//rotation around the y axis (vertical)
		const float yaw = -dx * _rot_speed / M_PI;
		//no roll
		_cam->update_dir(yaw, pitch, 0.f);

		if( _gl_widget->pivot_mode() != EOGL_widget::FREE )
		{
			// rotate around the pivot
			_gl_widget->update_pivot();
			Tbx::Vec3 pivot = _gl_widget->pivot();
			Tbx::Vec3 tcam  = _cam->get_pos();
			float d = (pivot - tcam).norm();
			_cam->set_pos(pivot - _cam->get_dir() * d);
		}
	}

	// camera straff
	if( _is_mid_pushed && _is_maj_pushed && 0)
	{
		int dx = x;
		int dy = (_cam->height() - y);
		Point3 p     = _cam->un_project( Point3(        dx,         dy, 0.f) );
		Point3 old_p = _cam->un_project( Point3(_cam_old_x, _cam_old_y, 0.f) );
		Tbx::Vec3 v = old_p - p;

		_cam->set_pos( _cam->get_pos() + v );
		_gl_widget->set_pivot_user(  _gl_widget->pivot() + v );
	}

	_cam_old_x = x;
	_cam_old_y = _cam->height() - y;
}

void IO_interface_skin::mouseReleaseEvent(QMouseEvent* event)
{
	_is_right_pushed = (event->button() == Qt::RightButton) ? false : _is_right_pushed;
	_is_left_pushed  = (event->button() == Qt::LeftButton)  ? false : _is_left_pushed;
	_is_mid_pushed   = (event->button() == Qt::MidButton)   ? false : _is_mid_pushed;

	if(!_is_left_pushed){
		gizmo()->reset_constraint();
		_is_gizmo_grabed = false;
	}
	_main_win->updateGL();
}

void IO_interface_skin::mousePressEvent(QMouseEvent* event)
{
	const int x = event->x();
	const int y = event->y();

	_old_x = x;
	_old_y = _cam->height() - y;

	_cam_old_x = x;
	_cam_old_y = _cam->height() - y;

	_is_right_pushed = (event->button() == Qt::RightButton);
	_is_left_pushed  = (event->button() == Qt::LeftButton);
	_is_mid_pushed   = (event->button() == Qt::MidButton);

	if( _gl_widget->_draw_gizmo && _is_left_pushed )
		_is_gizmo_grabed = gizmo()->select_constraint(*_cam, x, y);
}

void IO_interface_skin::push_msge(const QString& str)
{
	//_gl_widget->_msge_stack->push(str, true);
}

IO_interface_skin::~IO_interface_skin()
{
	delete _cam;
}
