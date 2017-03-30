#pragma once
#include <QGLViewer/qglviewer.h>
#include "select_tool.h"
#include "basic_types.h"
#include "manipulate_object.h"
#include "QGLViewer/manipulatedFrame.h"
#include <vector>
namespace qglviewer
{
	class AxisPlaneConstraint;
}


enum EnumConstraint
{
	Local_Constraint,
	World_Constraint,
	Camera_Constraint
};
enum EnumConstraintObj
{
	NONE_OBJECT = -1,
	X_AXIS , Y_AXIS, Z_AXIS,
	X_ROTATE, Y_ROTATE, Z_ROTATE,
	YZ_PLANE, XZ_PLAINE, XY_PLAIN,
	SCREEN_PLAIN
};

void drawAxisWithNames(qreal length, int select = -1, bool isWithName = false);

/*	click select ,Rectangle Select Tool ,and manipulate the select things	*/
class ManipulateTool :public qglviewer::ManipulatedFrame,public Tool
{
	Q_OBJECT
public:


	ManipulateTool(PaintCanvas* canvas,ManipulatedObject::ManipulatedObjectType _manipulateObjectType = ManipulatedObject::OBJECT);
	~ManipulateTool() {
		for (ManipulatedObject* obj : select_tool.m_selected_objs_)
		{
			delete obj;
		}
		select_tool.m_selected_objs_.clear();
	}

	virtual void move(QMouseEvent *e, qglviewer::Camera* camera = NULL);
	virtual void drag(QMouseEvent *e,  qglviewer::Camera* camera = NULL);
	virtual void release(QMouseEvent *e,  qglviewer::Camera* camera = NULL);
	virtual void press(QMouseEvent* e,  qglviewer::Camera* camera = NULL);
	virtual void keyPressEvent(QKeyEvent *e);
	virtual void draw();
public slots:
	virtual void postSelect();
	void postManipulateToolSelection();
public :
	ToolType	tool_type() const { return select_tool.tool_type_; }
	void	set_tool_type(ToolType type) { select_tool.tool_type_ = type; }
	unsigned int	cur_sample_to_operate() const { return select_tool.cur_sample_to_operate_; }
	void set_cur_smaple_to_operate(unsigned int sample_idx)
	{
		select_tool.cur_sample_to_operate_ = sample_idx;
	}
protected slots:
	void startManipulation();
	/*
		when frame is manipulate and mouse release
	*/
	void afterManipulate();
	void manipulatedFrameHasChanged();
protected:
	
	
	ManipulatedFrame* manipulatedFrame()
	{
		return this;
	}

//	qglviewer::ManipulatedFrame manipulated_frame;
	void changeToConstraint(EnumConstraint curstraint);
	void setTranslationConstraintType(qglviewer::AxisPlaneConstraint::Type type);
	void setRotationConstraintType(qglviewer::AxisPlaneConstraint::Type type);
	void setTranslationConstraintDirection(int transDir);
	void setRotationConstraintDirection(int rotDir);

	void select(const QPoint& point);

	float camera_dist;
	float axis_length;
	float rotate_circle_radius;
	EnumConstraintObj selected_name;
	qglviewer::AxisPlaneConstraint* cur_constraint;
	EnumConstraint activeConstraint;
	SelectTool select_tool;
	bool isManipultedFrameChanged;
protected slots:
	void slot_action_LocalConstraint()
	{
		changeToConstraint(Local_Constraint);
	}
	void slot_action_WorldConstraint()
	{
		changeToConstraint(World_Constraint);
	}
	void slot_action_CameraConstraint()
	{
		changeToConstraint(Camera_Constraint);
	}


};
