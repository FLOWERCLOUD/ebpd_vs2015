#pragma once
#include "sample.h"
#include <QGLViewer\frame.h>
#include <QGLViewer\vec.h>

#include <vector>
namespace LBS_Control
{
	void drawFrame(int length);
}


class Handle
{
public:
	enum HandleType
	{
		POINT_HANDLE,
		SKELTON_HANDEL,
		CAGE_HANDLE
	};
	Handle(Sample& _sample,int handle_idx = -1):
		sample_(_sample) ,handle_idx_(handle_idx), isSelected(true), isShowFrame(true)
	{

	}
	//应该要返回世界坐标下的标架
	qglviewer::Vec getWorldPosition()
	{
		Matrix44 mat = sample_.matrix_to_scene_coord();
		Vec4	tmp(frame_.position().x, frame_.position().y, frame_.position().z, 1.);
		Vec4	point_to_show = mat * tmp;
		qglviewer::Vec global_pos = sample_.getFrame().localInverseCoordinatesOf( qglviewer::Vec(point_to_show(0), point_to_show(1), point_to_show(2)) );
		return global_pos;
	}
	void setWorldPosition(qglviewer::Vec& position)
	{
		Matrix44 mat = sample_.inverse_matrix_to_scene_coord();
		Vec4	tmp(position.x, position.y, position.z, 1.);
		Vec4	point_to_show = mat * tmp;
		qglviewer::Vec local_pos = sample_.getFrame().localCoordinatesOf(qglviewer::Vec(point_to_show(0), point_to_show(1), point_to_show(2)));
		frame_.setPosition(local_pos);
	}

	qglviewer::Vec getLocalPosition()
	{
		return frame_.translation();
	}
	void setLocalPosition(qglviewer::Vec& position)
	{
		frame_.setTranslation(position);
	}
	void draw(const Matrix44& adjust_matrix , bool isWithName = false);

	float get_wi()
	{
		return handle_idx_;
	}

	bool is_root()
	{
		return is_root_;
	}
	const Handle * get_parent() const
	{
		return this->parent;
	}
	const std::vector<Handle*> & get_children() const
	{
		return children;
	}
	Eigen::Vector3d rest_tip()
	{
		return Eigen::Vector3d();
	}
	// Parent Bone, NULL only if this bone is a root
	Handle * parent;
	std::vector<Handle*> children;
	bool is_root_;
	int handle_idx_;
	bool isSelected;
	bool isShowFrame;
	qglviewer::Frame frame_;//handle 相对于sample标架下的标架
	Sample& sample_;

};
class Sample;
class TransAndRotation
{
public:
	qglviewer::Vec translation_;
	qglviewer::Quaternion rotation_;

};
class MeshControl
{
public:
	MeshControl(Sample& smp) :smp_(smp)
	{

	}
	~MeshControl()
	{
		for (int i = 0; i < handles_.size(); ++i)
		{
			delete handles_[i];
		}
	}
	void updateSample();

	void draw(bool isWithName = false);
/*
bind_frame 的父标架为sample标架，bind_frame 的计算方法是对原始点，求各个分块的最大权重的位置，然后就将
各个bind_frame 设置在 该位置，初始朝向都一样。
然后因为handle 的父标架也为sample标架。在LBS模型中的变换矩阵就可以通过计算 handle的标架与bind_frame标架之间的相对变换来求得

*/



	void bindControl( std::vector<float>& _inputVertices,/*ori points*/
	//	std::vector<qglviewer::Frame>& _bind_frame,
		std::vector<TransAndRotation>& _transfo,
		std::vector<float>& _boneWeights,
		std::vector<int>& _boneWightIdx,
		int numIndices,int numBone,int numVertices, bool _isQlerp);
	std::vector<float> inputVertices_;
	std::vector<qglviewer::Frame> bind_frame_;// the bind position and orientation defined in the sample's local coordinate
	std::vector<TransAndRotation> transfo_;
	int numIndices_;
	int numBone_ ;
	int numVertices_ ;
	bool isQlerp_;
	std::vector<float> boneWeights_;
	std::vector<int> boneWightIdx_;
	std::vector<Handle*> handles_;
	Sample& smp_;
};

