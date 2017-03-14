#include "../../qt_gui/paint_canvas.h"
#include "IO_skeleton.hpp"
#include "../../control/cuda_ctrl.hpp"
#include "../../global_datas/cuda_globals.hpp"
#include "../../animation/skeleton.hpp"

#include <iostream>
using std::cout;
using std::endl;
static bool debug = 1;
//#include "../../global_datas/cuda_globals.hpp"
//#include "../../global_datas/toolglobals.hpp"

using namespace Cuda_ctrl;
using namespace Tbx;






IO_skeleton::IO_skeleton(PaintCanvas* gl_widget) :
	IO_interface_skin(gl_widget),
	_janim_on(true),
	_mouse_z(0.f),
	_joint(-1),
	_last_vertex(-1),
	_nb_selected_vert(0),
	_move_joint_mode(false),
	_selection_hysteresis(8.f)
{
	using namespace Cuda_ctrl;

	_gl_widget->set_draw_skeleton( true );

	update_moved_vertex();

	// Setup contextual menu
	_menu = new QMenu(_gl_widget);
	{
		QMenu* sub_menu_1 = _menu->addMenu("convert bone to");
		sub_menu_1->addAction("Hermite RBF");
		_map_menutext_enum["Hermite RBF"] = EBone::HRBF;
		sub_menu_1->addAction("Implicit cylinder");
		_map_menutext_enum["Implicit cylinder"] = EBone::CYLINDER;
		sub_menu_1->addAction("SSD");
		_map_menutext_enum["SSD"] = EBone::SSD;
		sub_menu_1->addAction("Precomputed");
		_map_menutext_enum["Precomputed"] = EBone::PRECOMPUTED;

		QMenu* sub_menu_2 = _menu->addMenu("Blending type to");
		sub_menu_2->addAction("Arc gradient controlled");
		_map_menutext_enum["Arc gradient controlled"] = EJoint::GC_ARC_CIRCLE_TWEAK;
		sub_menu_2->addAction("Union with Max");
		_map_menutext_enum["Union with Max"] = EJoint::MAX;
		sub_menu_2->addAction("Bulge");
		_map_menutext_enum["Bulge"] = EJoint::BULGE;
	}
}

void IO_skeleton::mousePressEvent(QMouseEvent* event)
{
	IO_interface_skin::mousePressEvent(event);
	using namespace Cuda_ctrl;

	//const int x = event->x();
	//const int y = event->y();

	update_gl_matrix();

	if(event->button() == Qt::LeftButton)
	{
		if(0){}
			//_potential_plane._setup = false;
		else
		{
			if(!_is_gizmo_grabed)
			{
				const bool rest = _gl_widget->rest_pose();
				if( _is_ctrl_pushed )
					_skeleton.select_joint(*_cam, _old_x, _old_y, rest);
				else if( _is_maj_pushed )
					_skeleton.unselect(*_cam, _old_x, _old_y, rest);
				else
					_skeleton.select_safely(*_cam, _old_x, _old_y, rest);
			}

			update_moved_vertex();

			if(_move_joint_mode)
			{
				Tbx::Vec3 v = Cuda_ctrl::_skeleton.joint_pos(_last_vertex);
				GLdouble vx, vy, vz;
				gluProject(v.x, v.y, v.z,
					_modelview, _projection, _viewport,
					&vx, &vy, &vz);
				_mouse_z = vz;
			}
		}
	}
}

void IO_skeleton::mouseMoveEvent(QMouseEvent* event)
{
	IO_interface_skin::mouseMoveEvent(event);
	using namespace Cuda_ctrl;

	const int x = event->x();
	const int y = event->y();

	update_gl_matrix();

	if(_is_mid_pushed && _joint > -1)
	{
		//...
	}

	if(_is_left_pushed)
	{
		Vec2 m_pos    ((float)x     , (float)y     );
		Vec2 m_clicked((float)_old_x, (float)_old_y);

		if(_joint > -1 &&  _janim_on && (m_pos - m_clicked).norm() > 1.5f )
		{
			// Get local transformation of the gizmo
			TRS giz_tr = _gizmo_tr;
			if(debug)
			{
				cout<<"joint "<<_joint<<"_gizmo_tr "<<endl;
				giz_tr.to_transfo().print();
			}

			//                if( giz_tr._angle > 0.01f)

			float acc_angle = 0.f;
			const float step = 0.5f;
			float sign = giz_tr._angle > 0.f ? 1.f : -1.f;
			giz_tr._angle *= sign;
			while( acc_angle < giz_tr._angle )
			{
				acc_angle += (acc_angle + step) > giz_tr._angle ? giz_tr._angle - acc_angle : step;

				// Compute rotation
				int pid = g_skel->parent(_joint) > -1  ? g_skel->parent(_joint) : _joint;

				Transfo pframe     = g_skel->joint_anim_frame( pid );
				Transfo pframe_inv = pframe.fast_invert();

				// Axis in world coordinates
				Tbx::Vec3 world_axis = gizmo()->old_frame() * giz_tr._axis;

				// Joint origin in its parent joint coordinate system
				Point3 org = pframe_inv * _curr_joint_org.to_point3();

				// Rotation of the joint in its parent joint coordinate system
				Transfo tr = Transfo::rotate(org.to_vec3(), pframe_inv * world_axis, (sign * acc_angle)/*giz_tr._angle*/);


				// Compute Translation

				// Translation in world coordinates
				Tbx::Vec3 world_trans = gizmo()->old_frame() * giz_tr._translation;

				// Translation of the joint in its parent coordinate system
				tr = Transfo::translate( pframe_inv * world_trans ) * tr;
				if(debug)
				{
					cout<<"joint "<<_joint<<"tr "<<endl;
					tr.print();
					cout<<"_curr_joint_lcl "<<endl;
					_curr_joint_lcl.print();
				}


				// Concatenate last user defined transformation
				Transfo usr = tr * _curr_joint_lcl;


				// Update the skeleton position
				kinec()->set_user_lcl_parent( _joint, usr );

				// TODO: ensure transformation is small or do a loop to update progressively (maybe interpolate linearly in the deformer between transformations ...)
				if( _anim_mesh->_incremental_deformation )
					_anim_mesh->deform_mesh(); // Do one step of incr algo

			}

			{
				update_moved_vertex();
				if( dynamic_cast<Gizmo_rot*>( gizmo() ) != 0 )
				{
					Gizmo_rot* g = (Gizmo_rot*)gizmo();
					g->_clicked.x = x;
					g->_clicked.y = _gl_widget->height() - y;
				}
			}

			// Update gizmo orientation
			//update_frame_gizmo();
		}

		if(_move_joint_mode && _last_vertex > -1)
		{
			GLdouble ccx, ccy, ccz;
			float cx = x, cy = _cam->height()-y, cz = _mouse_z;
			gluUnProject(cx, cy, cz,
				_modelview, _projection, _viewport,
				&ccx, &ccy, &ccz);
			Tbx::Vec3 pos = Tbx::Vec3(ccx, ccy, ccz);
			_graph.set_vertex( _last_vertex, pos);
			_skeleton.set_joint_pos( _last_vertex, pos );
		}
		//if(_is_gizmo_grabed)
		//{
		//	TRS gizmo_tr = gizmo()->slide(*_cam, x, y);

		//	Transfo res = global_transfo( gizmo_tr );

		//	_picker->transform( _selection, res );
		//	Transfo new_frame = res * gizmo()->frame();
		//	gizmo()->set_frame( new_frame );
		//	gizmo()->slide_from( new_frame, pos);
		//}
	}

	_old_x = x;
	_old_y = _cam->height() - y;
}

void IO_skeleton::wheelEvent(QWheelEvent* event)
{
	using namespace Cuda_ctrl;
	const std::vector<int>& set = _skeleton.get_selection_set();
	float numDegrees = event->delta() / 8.f;
	float numSteps   = numDegrees / 15.f;

	if(0/*_potential_plane._setup*/)
	{
		//if(event->buttons() == Qt::NoButton )
		//{
		//	Tbx::Vec3 org  = _potential_plane._org;
		//	Tbx::Vec3 n    = _potential_plane._normal;
		//	_potential_plane._org = org + n * numSteps;
		//}
	}
	else if(_is_maj_pushed && set.size() > 0)
	{
		int bone_id = set[set.size()-1];
		_anim_mesh->incr_junction_rad(bone_id, numSteps/16.f);
		_display._raytrace_again = true;
	}
	else
		IO_interface_skin::wheelEvent(event);
}

void IO_skeleton::keyPressEvent(QKeyEvent* event)
{
	using namespace Cuda_ctrl;

	if(event == QKeySequence::SelectAll)
		Cuda_ctrl::_skeleton.select_all();

	IO_interface_skin::keyPressEvent(event);

	QString t = event->text();
	QChar c = t[0];

	switch( c.toLatin1() )
	{
	case 'j':
		{
			if(!_move_joint_mode && !event->isAutoRepeat())
			{
				_move_joint_mode = true;
				_gl_widget->set_rest_pose( true );
				push_msge("Hold the key to move the joint and left click");
			}
		}break;

	}//END_SWITCH

	if(_is_space_pushed)
		trigger_menu();
}

void IO_skeleton::trigger_menu()
{
	using namespace Cuda_ctrl;

	// Print and wait for the contextual menu
	QAction* selected_item = _menu->exec(QCursor::pos());
	// Key space release event is unfortunately eaten by the
	// QMenu widget we have to set it back by hand...
	_is_space_pushed = false;

	if(selected_item == 0) return;

	QMenu* menu = (QMenu*)selected_item->parentWidget();
	const std::vector<int>& set = _skeleton.get_selection_set();
	if(set.size() <= 0) {
		push_msge("Error : please select a least one bone to convert");
		return;
	}

	std::string item_text = selected_item->text().toStdString();

	// TODO: check if the entry doesn't exists

	bool type_changed = false;
	for(unsigned i = 0; i < set.size(); i++)
	{
		if( menu->title().compare("convert bone to") == 0  )
		{
			int type = _map_menutext_enum[item_text];
			int bone_type = _skeleton.get_bone_type(set[i]);

			if(bone_type == EBone::PRECOMPUTED &&
				type == EBone::PRECOMPUTED)
			{
				std::cout << "Error : bone (" << i  << ") is a precomputed bone, it can't be precomputed" << std::endl;
			}
			else if(bone_type == EBone::SSD &&
				type      == EBone::PRECOMPUTED)
			{
				std::cout << "Error : bone (" << i << ") is an SSD bone, it can't be precomputed" << std::endl;
			}
			else
			{
				_anim_mesh->set_bone_type(set[i], type);
				type_changed = true;
			}
		}
		else if( menu->title().compare("Blending type to") == 0  )
		{
			EJoint::Joint_t type = (EJoint::Joint_t)_map_menutext_enum[item_text];
			_skeleton.set_joint_blending(set[i], type);
			type_changed = true;
		}
		Cuda_ctrl::_display._raytrace_again = true;
	}

	if(type_changed) _anim_mesh->update_base_potential();
}

void IO_skeleton::keyReleaseEvent(QKeyEvent* event)
{
	IO_interface_skin::keyReleaseEvent(event);

	QString t = event->text();
	QChar c = t[0];

	switch( c.toLatin1() )
	{
	case 'j':
		{
			if( _move_joint_mode && !event->isAutoRepeat())
			{
				_move_joint_mode = false;
				_gl_widget->set_rest_pose( false );
			}
		}break;

	}//END_SWITCH
}

void IO_skeleton::update_moved_vertex()
{
	using namespace Cuda_ctrl;
	const std::vector<int>& set = _skeleton.get_selection_set();

	_joint = -1;
	_last_vertex  = -1;
	gizmo()->show( false );
	if(set.size() > 0)
	{
		int idx = set[set.size()-1];
		_last_vertex  = idx;
		_joint = idx;

		gizmo()->show( true );

		_curr_joint_lcl = kinec()->get_user_lcl_parent( _joint );
		_curr_joint_org = g_skel->joint_anim_frame(_joint).get_translation();
	}

	//update_frame_gizmo();

	_nb_selected_vert = set.size();

	//IBL::Ctrl_shape shape;
	//if(_joint != -1)
	//	shape = _skeleton.get_joint_controller(_joint);
	//else
	//	shape = _operators.get_global_controller();

	//_main_win->update_ctrl_spin_boxes(shape);
}

void IO_skeleton::mouseReleaseEvent(QMouseEvent* event)
{
	IO_interface_skin::mouseReleaseEvent(event);

	using namespace Cuda_ctrl;
	if( _move_joint_mode )
	{
		// update selected vertex
		Cuda_ctrl::_anim_mesh->update_caps(_last_vertex , true, true);
		//update caps every sons
		const std::vector<int>& sons = _skeleton.get_sons( _last_vertex );
		for(unsigned i = 0; i < sons.size(); i++)
			Cuda_ctrl::_anim_mesh->update_caps(sons[i], true, true);

		Cuda_ctrl::_anim_mesh->update_base_potential();
		//Cuda_ctrl::_anim_mesh->update_clusters();
	}
}

void IO_skeleton::update_frame_gizmo()
{
	if(_joint > -1 ) {
		Transfo tr = g_skel->joint_anim_frame( _joint );
		gizmo()->set_transfo( tr );
	}
}

/// Gizmo global 'TRS' to 4*4 matrix 'Transfo'
Tbx::Transfo IO_skeleton::global_transfo(const TRS& gizmo_tr)
{
	Transfo tr = Transfo::translate( gizmo_tr._translation );

	Tbx::Vec3 raxis = gizmo_tr._axis;
	Tbx::Vec3 org = gizmo()->frame().get_org();
	Transfo rot = Transfo::rotate(org, raxis, gizmo_tr._angle);
	Transfo sc  =  Transfo::scale(org, gizmo_tr._scale );

	return tr * rot * sc;
}