#ifndef _SELECT_TOOL_H
#define _SELECT_TOOL_H
#include "tool.h"
#include "basic_types.h"
#include "manipulate_object.h"
#include <QGLViewer/vec.h>
#include <vector>
#include <QMenu>
class PaintCanvas;
class ManipulateTool;
class ManipulatedObject;
/*	Rectangle Select Tool	*/
class SelectTool :public QObject, public Tool 
{
	Q_OBJECT
public:
	friend class ManipulateTool;
	

	SelectTool(PaintCanvas* canvas , ManipulatedObject::ManipulatedObjectType _manipulateObjectType = ManipulatedObject::OBJECT):Tool(canvas),
							select_buffer_(nullptr),
							select_buffer_size_(0),
							popupMenu(NULL),
							manipulateObjectType_(_manipulateObjectType){}
	~SelectTool(){
	}

	virtual void move(QMouseEvent *e, qglviewer::Camera* camera = NULL);
	virtual void drag(QMouseEvent *e, qglviewer::Camera* camera = NULL);
	virtual void click(QMouseEvent *e, qglviewer::Camera* camera = NULL);
	virtual void release(QMouseEvent *e, qglviewer::Camera* camera = NULL);
	virtual void press(QMouseEvent* e, qglviewer::Camera* camera = NULL);
	virtual void draw();
signals:
	void postSelect();
public:
	const std::vector<IndexType>& get_selected_vertex_idx()
	{
		return selected_vertex_indices_;
	}


protected:
	void getKCloestPoint(qglviewer::Vec point, int k, std::vector<int>& selected_idx, std::vector<float>* selected_idx_distance = NULL);
	void getCloestHandle(qglviewer::Vec point, std::vector<int>& selected_handle_idx, int frame_idx);
	inline void select();
	inline void begin_select();
	inline void end_select();

protected:
	void draw_rectangle();
	inline void initialize_select_buffer();

protected:

	qglviewer::Vec	left_mouse_pressed_pos_;
	qglviewer::Vec	left_mouse_move_pos_;
	qglviewer::Vec	right_mouse_pressed_pos_;
	qglviewer::Vec	right_mouse_move_pos_;

	std::vector<IndexType> selected_vertex_indices_;
	int		select_buffer_size_;
	unsigned int*	select_buffer_;
	QRect	rectangle_;
protected:
	std::vector<ManipulatedObject*> m_selected_objs_;
	std::vector<ManipulatedObject*> m_input_objs_;
	std::vector<int> m_selected_idx_;
	std::vector<float> m_selected_idx_distance_;
	ManipulatedObject::ManipulatedObjectType manipulateObjectType_;
protected slots:
	void slot_action_OBJECT()
	{
		manipulateObjectType_ = ManipulatedObject::OBJECT;
	}
	void slot_action_VERTEX()
	{
		manipulateObjectType_ = ManipulatedObject::VERTEX;
	}
	void slot_action_HANDLE()
	{
		manipulateObjectType_ = ManipulatedObject::HANDLE;
	}
protected:
	QMenu* popupMenu;
};


#endif