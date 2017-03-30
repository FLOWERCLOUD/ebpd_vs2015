#include "qt_gui/paint_canvas.h"
#include "select_tool.h"
#include "sample_set.h"
#include "vertex.h"
#include "windows.h"
#include <gl/gl.h>
#include <gl/glu.h>
#include "globals.h"
#include "color_table.h"
#include "GlobalObject.h"
#include <QMenu>
#include <QAction>
#include <deque>
#include "LBS_Control.h"
extern std::vector<MeshControl*> g_MeshControl;


#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE 0x809D
#endif
void SelectTool::move(QMouseEvent *e, qglviewer::Camera* camera )
{
	if (left_mouse_button_ == true)
	{
			drag(e);
	}

}

void SelectTool::drag(QMouseEvent *e, qglviewer::Camera* camera )
{

	rectangle_.setBottomRight( e->pos() );
	canvas_->updateGL();

}
void SelectTool::getKCloestPoint(qglviewer::Vec point, int k, std::vector<int>& selected_idx, std::vector<float>* selected_idx_distance)
{
	SampleSet& set = (*Global_SampleSet);
	Sample& sample = set[cur_sample_to_operate_];
	PointType query_point(point.x, point.y, point.z);
	int cloest_idx = sample.closest_vtx(query_point);
	selected_idx.clear();
	selected_idx.push_back(cloest_idx);
	return;


	if (k <= 0)
		return;
	float min_distance = 10000.0f;
	std::deque<float> k_small;
	std::deque<int> k_small_idx;

	LOCK(sample);

	for (int i = 0; i < m_input_objs_.size(); ++i)
	{
		qreal x, y, z;
		qglviewer::Vec cur_point = m_input_objs_[i]->getWorldPosition();
		//		m_input_objs[i]->frame.getPosition(x, y, z);
		//		qglviewer::Vec cur_point( x,y,z );
		float distance = (point - cur_point).squaredNorm();
		if (k_small.size() < k)
		{
			if (k_small.size() && distance <= k_small.front())
			{
				k_small.push_front(distance);
				k_small_idx.push_front(i);

			}
			else if (k_small.size() && distance >= k_small.back())
			{
				k_small.push_back(distance);
				k_small_idx.push_back(i);
			}
			else if (!k_small.size())
			{
				k_small.push_front(distance);
				k_small_idx.push_front(i);
			}

		}
		else
		{
			if (k_small.size() && distance <= k_small.front())
			{
				k_small.push_front(distance);
				k_small_idx.push_front(i);
				k_small.pop_back();
				k_small_idx.pop_back();

			}
			else if (k_small.size() && distance < k_small.back())
			{

				k_small.pop_back();
				k_small_idx.pop_back();
				k_small.push_back(distance);
				k_small_idx.push_back(i);

				for (int m = k_small.size() - 1; m < k_small.size(); ++m)
				{
					float tmp = k_small[m];
					int tmp_idx = k_small_idx[m];
					int n = m - 1;
					while (n > -1 && tmp < k_small[n])
					{
						k_small[n + 1] = k_small[n];
						k_small_idx[n + 1] = k_small_idx[n];
						n--;
					}
					k_small[n + 1] = tmp;
					k_small_idx[n + 1] = tmp_idx;
				}

			}


		}


	}
	selected_idx.clear();
	if (selected_idx_distance)
		selected_idx_distance->clear();
	while (k_small.size())
	{
		selected_idx.push_back(k_small_idx.front());
		if (selected_idx_distance)
			selected_idx_distance->push_back(k_small.front());
		k_small_idx.pop_front();
		k_small.pop_front();
	}
	UNLOCK(sample);
}

void SelectTool::getCloestHandle(qglviewer::Vec point, std::vector<int>& selected_handle_idx ,int frame_idx)
{
	if ( frame_idx < g_MeshControl.size() )
	{


		MeshControl* p_mesh_ctrl = g_MeshControl[frame_idx];
		auto handles = p_mesh_ctrl->handles_;
		float min_distance = 1000;
		int handle_idx = -1;
		for (int i = 0; i < handles.size(); ++i)
		{
			Handle* handle = handles[i];
			qglviewer::Vec position = handle->getWorldPosition();

			float cur_dis = (point - position).norm();
			if (cur_dis < min_distance)
			{
				min_distance = cur_dis;
				handle_idx = i;
			}

		}
		if (handle_idx != -1)
			selected_handle_idx.push_back(handle_idx);
	}
	


}

void SelectTool::click(QMouseEvent *e, qglviewer::Camera* camera /*= NULL*/)
{
///	Logger << "SelectTool click" << e->pos().x() << e->pos().y() << e->pos().z() << std::endl;
	//Logger << "clicked point" << worldX << worldY << worldZ << std::endl;
	//if (stencilValue == 1)
	//{
	left_mouse_pressed_pos_[0]= e->x();
	left_mouse_pressed_pos_[1] = e->y();

	GLdouble		model[16];
	glGetDoublev(GL_MODELVIEW_MATRIX, model);

	GLdouble proj[16];
	glGetDoublev(GL_PROJECTION_MATRIX, proj);

	GLint view[4];
	glGetIntegerv(GL_VIEWPORT, view);

	int winX = e->x();
	int winY = view[3] - 1 - e->y();

	float zValue;
	glReadPixels(winX, winY, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &zValue);

	GLubyte stencilValue;
	glReadPixels(winX, winY, 1, 1, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, &stencilValue);

	//GLdouble worldX, worldY, worldZ;
	//gluUnProject(winX, winY, zValue, model, proj, view, &worldX, &worldY, &worldZ);
	//qglviewer::Vec queryPos(worldX, worldY, worldZ);
	QPoint targetpoint((float)e->x(), (float)e->y());
	bool isFound;
	qglviewer::Vec target3dVec = camera->pointUnderPixel(targetpoint, isFound);

	SampleSet& smpset = (*Global_SampleSet);
	Sample& smp = smpset[cur_sample_to_operate_];
	qglviewer::Frame& curFrame = smp.getFrame();


	if (isFound)
	{
		switch (manipulateObjectType_)
		{
		case  ManipulatedObject::VERTEX:
		{
			getKCloestPoint(target3dVec, 1, m_selected_idx_, &m_selected_idx_distance_);
			//}
			for (auto iter = m_selected_objs_.begin(); iter != m_selected_objs_.end(); ++iter)
			{
				delete *iter;
			}
			m_selected_objs_.clear();
			for (int i : m_selected_idx_)
			{
				ManipulatedObject* obj = new ManipulatedObjectIsVertex(cur_sample_to_operate_, &curFrame);
				((ManipulatedObjectIsVertex*)obj)->setControlVertexIdx(cur_sample_to_operate_, i);
				m_selected_objs_.push_back(obj);
			}
		}
		break;
		case  ManipulatedObject::EDGE:
		{

		}
		break;
		case  ManipulatedObject::FACE:
		{

		}
		break;
		case  ManipulatedObject::OBJECT:
		{
			for (auto iter = m_selected_objs_.begin(); iter != m_selected_objs_.end(); ++iter)
			{
				delete *iter;
			}
			m_selected_objs_.clear();
			ManipulatedObject* obj = new ManipulatedObjectIsOBJECT(cur_sample_to_operate_, &curFrame);
			m_selected_objs_.push_back(obj);
		}
		break;
		case  ManipulatedObject::HANDLE:
		{
			for (auto iter = m_selected_objs_.begin(); iter != m_selected_objs_.end(); ++iter)
			{
				delete *iter;
			}
			std::vector<int> selected_handle;
			getCloestHandle(target3dVec, selected_handle, cur_sample_to_operate_);
			m_selected_objs_.clear();
			if (selected_handle.size())
			{
				ManipulatedObject* obj = new ManipulatedObjectIsHANDLE(
					g_MeshControl, selected_handle[0], cur_sample_to_operate_, 
					&smp.getFrame());
				m_selected_objs_.push_back(obj);
			}
		}
		break;

		}

	}


}

void SelectTool::release(QMouseEvent *e, qglviewer::Camera* camera )
{
	if (left_mouse_button_ == true)
	{
		// Possibly swap left/right and top/bottom to make rectangle_ valid.
		rectangle_ = rectangle_.normalized();
		qglviewer::Vec release_pos(e->pos().x(), e->pos().y(), 0.0f);
		if (qglviewer::Vec(release_pos - left_mouse_pressed_pos_).norm() < 0.001)
		{
			click(e, camera);
		}
		else
		{
			select();
			rectangle_ = QRect(e->pos(), e->pos());
		}
		postSelect();
		canvas_->updateGL();
		left_mouse_button_ = false;
	}
	if (e->button() == Qt::RightButton)
	{
		qglviewer::Vec release_pos(e->pos().x(), e->pos().y(), 0.0f);
		if (qglviewer::Vec(release_pos - right_mouse_pressed_pos_).norm() < 0.001)
		{
			popupMenu = new QMenu("&Menu", (QWidget*)canvas_);

			QAction* action_1 = new QAction("&Object", popupMenu);
			QAction* action_2 = new QAction("&Vertex", popupMenu);
			QAction* action_3 = new QAction("&Handle", popupMenu);
			action_1->setCheckable(true);
			action_2->setCheckable(true);
			action_3->setCheckable(true);
			switch (manipulateObjectType_)
			{
			case  ManipulatedObject::VERTEX:
			{
				action_2->setChecked(true);
			}
			break;
			case  ManipulatedObject::EDGE:
			{

			}
			break;
			case  ManipulatedObject::FACE:
			{

			}
			break;
			case  ManipulatedObject::OBJECT:
			{
				action_1->setChecked(true);
			}
			break;
			case  ManipulatedObject::HANDLE:
			{
				action_3->setChecked(true);
			}
			break;

			}
			popupMenu->addAction(action_1);
			popupMenu->addAction(action_2);
			popupMenu->addAction(action_3);

			connect(action_1, SIGNAL(triggered()), this, SLOT(slot_action_OBJECT()));
			connect(action_2, SIGNAL(triggered()), this, SLOT(slot_action_VERTEX()));
			connect(action_3, SIGNAL(triggered()), this, SLOT(slot_action_HANDLE()));
		}
		canvas_->updateGL();
		right_mouse_button_ = false;


	}

}

void SelectTool::press(QMouseEvent* e, qglviewer::Camera* camera )
{
	if (e->button() == Qt::LeftButton)
	{

		left_mouse_button_ = true;
		left_mouse_pressed_pos_ = qglviewer::Vec( e->pos().x(), e->pos().y(),0.0f);
		rectangle_ = QRect(e->pos(), e->pos());
		canvas_->updateGL();
	}
	if (e->button() == Qt::RightButton)
	{

		right_mouse_button_ = true;
		right_mouse_pressed_pos_ = qglviewer::Vec(e->pos().x(), e->pos().y(), 0.0f);
	}
}

void SelectTool::draw()
{
	draw_rectangle();

//	Sample& sample = (*Global_SampleSet)[cur_sample_to_operate_];	
//	LOCK(sample);
//	glPushMatrix();
//	glMultMatrixd(sample.getFrame().matrix());
//
//	//draw hightlight vertex
//
////	Matrix44 adjust_mat = sample.matrix_to_scene_coord();
//	Matrix44 adjust_mat = Matrix44::Identity(4,4);
//	for (int i = 0; i < 4; ++i)
//	{
//		adjust_mat(i, i) = 1.0f;
//	}
//	glEnable(GL_MULTISAMPLE);
//	glEnable(GL_DEPTH_TEST);
//	glPointSize(Paint_Param::g_point_size);
//	glBegin(GL_POINTS);
//	for ( IndexType v_idx = 0; v_idx < sample.num_vertices(); v_idx++ )
//	{
//		ColorType c;
//		if (sample[v_idx].is_selected() == true )
//		{
//			c = SELECTED_COLOR;
//		}
//		else
//		{
//			ColorType color2 = Color_Utility::span_color_from_table( sample[v_idx].label()); 
//			c = color2;
//		}
//		
//		glColor4f(  c(0), c(1), c(2),c(3) );
//		sample[v_idx].draw_without_color(adjust_mat);
//	}
//	glEnd();
//	glDisable(GL_DEPTH_TEST);
//	glDisable(GL_MULTISAMPLE);
//	UNLOCK(sample);
//	glPopMatrix();
}

void SelectTool::draw_rectangle()
{
	canvas_->startScreenCoordinatesSystem();

	glDisable(GL_LIGHTING);

	glLineWidth(2.0);
	glColor4f(0.0f, 1.0f, 1.0f, 0.5f);
	glBegin(GL_LINE_LOOP);
	glVertex2i(rectangle_.left(),  rectangle_.top());
	glVertex2i(rectangle_.right(), rectangle_.top());
	glVertex2i(rectangle_.right(), rectangle_.bottom());
	glVertex2i(rectangle_.left(),  rectangle_.bottom());
	glEnd();	

	glEnable(GL_BLEND);
	glDepthMask(GL_FALSE);
	glColor4f(0.0, 0.0, 0.4f, 0.3f);
	glBegin(GL_QUADS);
	glVertex2i(rectangle_.left(),  rectangle_.top());
	glVertex2i(rectangle_.right(), rectangle_.top());
	glVertex2i(rectangle_.right(), rectangle_.bottom());
	glVertex2i(rectangle_.left(),  rectangle_.bottom());
	glEnd();
	glDisable(GL_BLEND);
	glDepthMask(GL_TRUE);

	glEnable(GL_LIGHTING);
	canvas_->stopScreenCoordinatesSystem();
}

void SelectTool::begin_select()
{
	
	canvas_->makeCurrent();
	initialize_select_buffer();

	switch (manipulateObjectType_)
	{
	case  ManipulatedObject::VERTEX:
	{

	}
	break;
	case  ManipulatedObject::EDGE:
	{

	}
	break;
	case  ManipulatedObject::FACE:
	{

	}
	break;
	case  ManipulatedObject::OBJECT:
	{

	}
	break;
	case  ManipulatedObject::HANDLE:
	{

	}
	break;

	}







	if ( select_buffer_==nullptr )
	{
		return;
	}

	glSelectBuffer( select_buffer_size_, select_buffer_ );
	glRenderMode(GL_SELECT);
	glInitNames();

	// Loads the matrices
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	static GLint viewport[4];
	canvas_->camera()->getViewport(viewport);
	gluPickMatrix(rectangle_.center().x(), rectangle_.center().y(), 
		rectangle_.width(), rectangle_.height(), viewport);

	// loadProjectionMatrix() first resets the GL_PROJECTION matrix with a glLoadIdentity().
	// The false parameter prevents this and hence multiplies the matrices.
	canvas_->camera()->loadProjectionMatrix(false);
	// Reset the original (world coordinates) modelview matrix
	canvas_->camera()->loadModelViewMatrix();
}

void SelectTool::end_select()
{
	
	glFlush();
	GLint nbHits = glRenderMode(GL_RENDER);

	SampleSet& set = (*Global_SampleSet);
	
	//reset selected vertex
	Sample& smp = set[cur_sample_to_operate_];
	LOCK(smp);
	for (IndexType i = 0; i < smp.num_vertices(); i++)
	{
		smp[i].set_selected(false);
	}

	UNLOCK(smp);

	// Interpret results : each object created 4 values in the selectBuffer().
	// (selectBuffer())[4*i+3] is the id pushed on the stack.
	selected_vertex_indices_.clear();
	for (int i=0; i<nbHits; ++i)
	{
		IndexType index = select_buffer_[4*i+3];
		if(set[cur_sample_to_operate_][index].is_visible() )//only select visible point
			selected_vertex_indices_.push_back(index);

	}

	
	std::sort( selected_vertex_indices_.begin(), selected_vertex_indices_.end() );

	LOCK(set[cur_sample_to_operate_]);
		
		for (IndexType i = 0; i < selected_vertex_indices_.size(); i++)
		{
			set(cur_sample_to_operate_, selected_vertex_indices_[i]).set_selected(true);
		}

	UNLOCK(set[cur_sample_to_operate_]);


	delete [] select_buffer_;
	select_buffer_ = nullptr;
	select_buffer_size_ = 0;
	return;
}

void SelectTool::select()
{
	begin_select();
	
	SampleSet& set = (*Global_SampleSet);
	LOCK(set[cur_sample_to_operate_]);
	
		set[cur_sample_to_operate_].draw_with_name();	

	UNLOCK(set[cur_sample_to_operate_]);
	
	end_select();
	
}


void SelectTool::initialize_select_buffer()
{
	if ( select_buffer_!=nullptr )
	{
		delete [] select_buffer_;
	}

	SampleSet& set = (*Global_SampleSet);
	LOCK(set[cur_sample_to_operate_]);
		select_buffer_size_ = (set[cur_sample_to_operate_].num_vertices())*4;
	UNLOCK(set[cur_sample_to_operate_]);


	select_buffer_ = new unsigned int[select_buffer_size_];
}