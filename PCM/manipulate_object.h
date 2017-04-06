#pragma once
#include <CustomGL\glew.h>
#include "QGLViewer/frame.h"
#include "GlobalObject.h"
#include "sample_set.h"
#include <vector>
#include "LBS_Control.h"

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

	ManipulatedObject(int smp_idx,qglviewer::Frame* _frame) :smp_idx_(smp_idx), sample_frame_(_frame)
	{}
	virtual ~ManipulatedObject()
	{}
	virtual  qglviewer::Vec getWorldPosition() = 0;

	virtual void setWorldPosition(qglviewer::Vec pos) = 0;

	virtual qglviewer::Quaternion getWorldOrientation() = 0;

	virtual void setWorldOrientation(qglviewer::Quaternion& _q) = 0;

	inline qglviewer::Frame& getFrame()
	{
		return *sample_frame_;
	}


	inline void setFrame(qglviewer::Frame* frame)
	{
		this->sample_frame_ = frame;
	}

	inline ManipulatedObjectType getType()
	{
		return type;
	}

protected:
	ManipulatedObjectType type;
	qglviewer::Frame* sample_frame_;
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
		glMultMatrixd(sample_frame_->matrix());
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
		//注意还要通过sample标架 转换 到世界坐标
		return sample_frame_->inverseCoordinatesOf(pos);
	}

	virtual void setWorldPosition(qglviewer::Vec pos)
	{
		SampleSet& set = (*Global_SampleSet);
		Sample& smp = set[smp_idx_];
		Vertex& vtx = smp[vertex_idx_];
		qglviewer::Vec pos_local = sample_frame_->coordinatesOf(pos);
		Matrix44 mat = smp.inverse_matrix_to_scene_coord();
		Vec4	tmp(pos_local.x, pos_local.y, pos_local.z, 1.);
		Vec4	ori_point = mat * tmp;
		vtx.set_position(pcm::PointType(ori_point.x(), ori_point.y(), ori_point.z()));

	}

	virtual qglviewer::Quaternion getWorldOrientation()
	{
		return qglviewer::Quaternion();
	}

	virtual void setWorldOrientation(qglviewer::Quaternion& _q)
	{

	}
	void setControlVertexIdx(int frame_idx, int vertex_idx)
	{
		vertex_idx_ = vertex_idx;
		smp_idx_ = frame_idx;
	}
public:
	inline qglviewer::Vec getLocalPosition() //local pos
	{
		SampleSet& set = (*Global_SampleSet);
		Sample& smp = set[smp_idx_];
		Vertex& vtx = smp[vertex_idx_];
		qglviewer::Vec local_pos(vtx.x(), vtx.y(), vtx.z());
		return local_pos;
	}
	inline qglviewer::Vec getNormal()
	{
		return normal_;
	}
	inline qglviewer::Vec getColor()
	{
		return color_;
	}
	inline void setLocalPosition(qglviewer::Vec position) //local pos
	{
		SampleSet& set = (*Global_SampleSet);
		Sample& smp = set[smp_idx_];
		Vertex& vtx = smp[vertex_idx_];
		vtx.set_position(pcm::PointType(position.x, position.y, position.z));
	}
	inline void setNormal(qglviewer::Vec normal)
	{
		this->normal_ = normal;
	}
	inline void setColor(qglviewer::Vec color)
	{
		this->color_ = color;
	}
	inline int getVertexIdx()
	{
		return vertex_idx_;
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
		return sample_frame_->position();
	}
	virtual void setWorldPosition(qglviewer::Vec pos)
	{
		sample_frame_->setPosition(pos);
	}

	virtual qglviewer::Quaternion getWorldOrientation()
	{

		return sample_frame_->orientation();
	}

	virtual void setWorldOrientation(qglviewer::Quaternion& _q)
	{
		sample_frame_->setOrientation(_q);
	}

};

class MeshControl;
class Handle;

class ManipulatedObjectIsHANDLE : public ManipulatedObject
{
public:
	ManipulatedObjectIsHANDLE(std::vector<MeshControl*>& mesh_control,int handle_idx, int smp_idx, qglviewer::Frame* _sample_frame)
		:ManipulatedObject(smp_idx, _sample_frame), mesh_control_(mesh_control), handle_idx_(handle_idx)
	{
		type = HANDLE;
		handle_frame_ = &mesh_control[smp_idx]->handles_[handle_idx]->frame_;
	}
	virtual void draw() const
	{

	}
	//返回世界坐标中的值，便于放置manipulator
	virtual  qglviewer::Vec getWorldPosition()
	{
		//Handle* handle = mesh_control_[smp_idx_]->handles_[handle_idx_]; 

		SampleSet& set = (*Global_SampleSet);
		Sample& smp = set[smp_idx_];
		Matrix44 mat = smp.matrix_to_scene_coord();
		qglviewer::Vec ori_pos = handle_frame_->translation();
		Vec4	tmp(ori_pos.x, ori_pos.y, ori_pos.z, 1.);
		Vec4	point_to_show = mat * tmp;
		qglviewer::Vec pos(point_to_show.x(), point_to_show.y(), point_to_show.z());
		return sample_frame_->inverseCoordinatesOf(pos);


	}
	virtual void setWorldPosition(qglviewer::Vec pos)
	{

		SampleSet& set = (*Global_SampleSet);
		Sample& smp = set[smp_idx_];
		qglviewer::Vec pos_local = sample_frame_->coordinatesOf(pos);
		Matrix44 mat = smp.inverse_matrix_to_scene_coord();
		Vec4	tmp(pos_local.x, pos_local.y, pos_local.z, 1.);
		Vec4	ori_point = mat * tmp;
		handle_frame_->setTranslation(qglviewer::Vec(ori_point(0), ori_point(1), ori_point(2)) );
	}

	virtual qglviewer::Quaternion getWorldOrientation()
	{
		qglviewer::Quaternion local_rotate = handle_frame_->rotation();
		return sample_frame_->rotation() * local_rotate;
	}

	virtual void setWorldOrientation(qglviewer::Quaternion& _q)
	{
		qglviewer::Quaternion local_rotate = _q * sample_frame_->rotation().inverse();
		sample_frame_->setRotation(local_rotate);
	}


	std::vector<MeshControl*>& mesh_control_;
	int handle_idx_;
	qglviewer::Frame* handle_frame_;
};