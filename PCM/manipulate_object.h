#pragma once

#include "QGLViewer/frame.h"
#include "GlobalObject.h"
#include "sample_set.h"

#include <vector>
class ManipulatedObject
{
public:
	enum ManipulatedObjectType
	{
		VERTEX,
		EDGE,
		FACE,
		OBJECT,
		HANDLE
	};
	int idx;
	virtual void draw() const;

	ManipulatedObject(int smp_idx,qglviewer::Frame* _frame) :smp_idx_(smp_idx), frame_(_frame)
	{}
	virtual ~ManipulatedObject()
	{}
	virtual  qglviewer::Vec getWorldPosition() = 0;

	virtual void setWorldPosition(qglviewer::Vec pos) = 0;

	inline qglviewer::Frame& getFrame()
	{
		return *frame_;
	}


	inline void setFrame(qglviewer::Frame* frame)
	{
		this->frame_ = frame;
	}

	inline ManipulatedObjectType getType()
	{
		return type;
	}

protected:
	ManipulatedObjectType type;
	qglviewer::Frame* frame_;
	int smp_idx_;

};
class ManipulatedObjectIsVertex : public ManipulatedObject
{
public:
	ManipulatedObjectIsVertex(int smp_idx, qglviewer::Frame* _frame)
		:ManipulatedObject(smp_idx, _frame)
	{
		type = VERTEX;
	}
	inline virtual void draw() const
	{
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		SampleSet& set = (*Global_SampleSet);
		Sample& smp = set[smp_idx_];
		Vertex& vtx = smp[vertex_idx_];
		glPushMatrix();
		glMultMatrixd(frame_->matrix());
		glPointSize(10.0f);
		glColor3f(1.0f, 1.0f, 0.0f);
		glBegin(GL_POINTS);
		Matrix44 mat = smp.matrix_to_scene_coord();
		Vec4	tmp(vtx.x(), vtx.y(), vtx.z(), 1.);
		Vec4	point_to_show = mat * tmp;
		glNormal3f(normal_.x, normal_.y, normal_.z);
		glVertex3f((GLfloat)point_to_show(0), (GLfloat)point_to_show(1), (GLfloat)point_to_show(2));
		glEnd();
		glPopMatrix();
		glPopAttrib();
	}

	virtual  qglviewer::Vec getWorldPosition()
	{
		SampleSet& set = (*Global_SampleSet);
		Sample& smp = set[smp_idx_];
		Vertex& vtx = smp[vertex_idx_];
		Matrix44 mat = smp.matrix_to_scene_coord();
		Vec4	tmp(vtx.x(), vtx.y(), vtx.z(), 1.);
		Vec4	point_to_show = mat * tmp;
		qglviewer::Vec pos(point_to_show.x(), point_to_show.y(), point_to_show.z());
		return frame_->inverseCoordinatesOf(pos);
	}

	virtual void setWorldPosition(qglviewer::Vec pos)
	{
		SampleSet& set = (*Global_SampleSet);
		Sample& smp = set[smp_idx_];
		Vertex& vtx = smp[vertex_idx_];
		qglviewer::Vec pos_local = frame_->coordinatesOf(pos);
		Matrix44 mat = smp.inverse_matrix_to_scene_coord();
		Vec4	tmp(pos_local.x, pos_local.y, pos_local.z, 1.);
		Vec4	ori_point = mat * tmp;
		vtx.set_position(pcm::PointType(ori_point.x(), ori_point.y(), ori_point.z()));

	}

	void setControlVertexIdx(int frame_idx, int vertex_idx)
	{
		vertex_idx_ = vertex_idx;
		smp_idx_ = frame_idx;
	}
private:
	inline qglviewer::Vec getPosition() //local pos
	{
		return position_;
	}
	inline qglviewer::Vec getNormal()
	{
		return normal_;
	}
	inline qglviewer::Vec getColor()
	{
		return color_;
	}
	inline void setPosition(qglviewer::Vec position) //local pos
	{
		this->position_ = position;
	}
	inline void setNormal(qglviewer::Vec normal)
	{
		this->normal_ = normal;
	}
	inline void setColor(qglviewer::Vec color)
	{
		this->color_ = color;
	}
private:
	int vertex_idx_;
	qglviewer::Vec position_;
	qglviewer::Vec normal_;
	qglviewer::Vec color_;
};

class ManipulatedObjectIsOBJECT : public ManipulatedObject
{
public:
	ManipulatedObjectIsOBJECT(int smp_idx, qglviewer::Frame* _frame)
		:ManipulatedObject(smp_idx,_frame)
	{
		type = OBJECT;
	}
	virtual void draw() const
	{

	}
	virtual  qglviewer::Vec getWorldPosition()
	{
		return frame_->position();
	}
	virtual void setWorldPosition(qglviewer::Vec pos)
	{
		frame_->setPosition(pos);
	}

};

class ManipulatedObjectIsHANDLE : public ManipulatedObject
{
public:
	ManipulatedObjectIsHANDLE(int smp_idx, qglviewer::Frame* _frame)
		:ManipulatedObject(smp_idx, _frame)
	{
		type = HANDLE;
	}
	virtual void draw() const
	{

	}
	virtual  qglviewer::Vec getWorldPosition()
	{
		return frame_->position();
	}
	virtual void setWorldPosition(qglviewer::Vec pos)
	{
		frame_->setPosition(pos);
	}

};