#ifndef _TOOL_H
#define _TOOL_H
#include <QPoint>
#include <QMouseEvent>

class PaintCanvas;
namespace qglviewer
{
	class Camera;
}


class Tool
{
public:
	Tool(){}
	enum ToolType { EMPTY_TOOL, SELECT_TOOL, MANIPULATE_TOOL };

	Tool( PaintCanvas* canvas ):canvas_(canvas),
								left_mouse_button_(false),
								right_mouse_button_(false){}
	virtual ~Tool(){}

public:
	virtual void postManipulateToolSelection() {}
	virtual void press(QMouseEvent *e ,qglviewer::Camera* camera = NULL) = 0;
	virtual void move(QMouseEvent *e, qglviewer::Camera* camera = NULL) = 0;
	virtual void release(QMouseEvent *e, qglviewer::Camera* camera = NULL) = 0;
	virtual void drag(QMouseEvent *e, qglviewer::Camera* camera = NULL) = 0;
	virtual void keyPressEvent(QKeyEvent *e){ }
	virtual void draw() = 0;
public:
	virtual ToolType	tool_type () const { return tool_type_; }
	virtual void	set_tool_type( ToolType type ){ tool_type_ = type; }
	virtual unsigned int	cur_sample_to_operate() const { return cur_sample_to_operate_; }
	virtual void set_cur_smaple_to_operate( unsigned int sample_idx )
	{
		cur_sample_to_operate_ = sample_idx;
	}
protected:
	ToolType	tool_type_;
	unsigned int	cur_sample_to_operate_;

	bool	left_mouse_button_;
	bool	right_mouse_button_;

	PaintCanvas*	canvas_;
};

#endif