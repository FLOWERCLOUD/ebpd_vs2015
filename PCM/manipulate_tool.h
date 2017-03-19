#pragma once

#include "select_tool.h"
#include "basic_types.h"
#include "manipulate_object.h"
#include "QGLViewer/manipulatedFrame.h"
#include <QGLViewer/qglviewer.h>

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
class ManipulateTool : public SelectTool,public qglviewer::ManipulatedFrame
{
public:


	ManipulateTool(PaintCanvas* canvas);
	~ManipulateTool() {
		for (ManipulatedObject* obj : m_input_objs)
		{
			delete obj;
		}
		m_input_objs.clear();
		m_selected_objs.clear();	
	}

	virtual void move(QMouseEvent *e, qglviewer::Camera* camera = NULL);
	virtual void drag(QMouseEvent *e,  qglviewer::Camera* camera = NULL);
	virtual void release(QMouseEvent *e,  qglviewer::Camera* camera = NULL);
	virtual void press(QMouseEvent* e,  qglviewer::Camera* camera = NULL);
	virtual void draw();
	void postSelection();
protected:
	void getKCloestPoint(qglviewer::Vec point, int k, std::vector<int>& selected_idx, std::vector<float>* selected_idx_distance = NULL );
	void startManipulation();
	ManipulatedFrame* manipulatedFrame()
	{
		return this;
	}
	std::vector<ManipulatedObject*> m_selected_objs;
	std::vector<ManipulatedObject*> m_input_objs;
	std::vector<int> m_selected_idx;
	std::vector<float> m_selected_idx_distance;
//	qglviewer::ManipulatedFrame manipulated_frame;
	void changeConstraint(EnumConstraint curstraint);
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



};
