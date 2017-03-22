#pragma once

#include "QGLViewer/constraint.h"
class ManipulatedObject;
namespace qglviewer
{
	class Camera;
}
class ManipulatedFrameSetConstraint : public qglviewer::Constraint
{
public:
	void clearSet();
	void addObjectToSet(ManipulatedObject* o);

	virtual void constrainTranslation(qglviewer::Vec &translation, qglviewer::Frame *const frame);
	virtual void constrainRotation(qglviewer::Quaternion &rotation, qglviewer::Frame *const frame);

private:
	std::vector<ManipulatedObject*> objects_;
};


class ManipulatedFrameSetLocalConstraint : public qglviewer::LocalConstraint
{
public:
	void clearSet();
	void addObjectToSet(ManipulatedObject* o);

	virtual void constrainTranslation(qglviewer::Vec &translation, qglviewer::Frame *const frame);
	virtual void constrainRotation(qglviewer::Quaternion &rotation, qglviewer::Frame *const frame);
	std::vector<ManipulatedObject*>& getManipulateObjs()
	{
		return objects_;
	}
	void setManipulateObjs(std::vector<ManipulatedObject*>& objs)
	{
		objects_ = objs;
	}

private:
	std::vector<ManipulatedObject*> objects_;
};

class ManipulatedFrameSetWorldConstraint : public qglviewer::WorldConstraint
{
public:
	void clearSet();
	void addObjectToSet(ManipulatedObject* o);

	virtual void constrainTranslation(qglviewer::Vec &translation, qglviewer::Frame *const frame);
	virtual void constrainRotation(qglviewer::Quaternion &rotation, qglviewer::Frame *const frame);
	std::vector<ManipulatedObject*>& getManipulateObjs()
	{
		return objects_;
	}
	void setManipulateObjs(std::vector<ManipulatedObject*>& objs)
	{
		objects_ = objs;
	}
private:
	std::vector<ManipulatedObject*> objects_;
};

class ManipulatedFrameSetCameraConstraint : public qglviewer::CameraConstraint
{
public:
	explicit ManipulatedFrameSetCameraConstraint(const qglviewer::Camera* const camera);
	void clearSet();
	void addObjectToSet(ManipulatedObject* o);

	virtual void constrainTranslation(qglviewer::Vec &translation, qglviewer::Frame *const frame);
	virtual void constrainRotation(qglviewer::Quaternion &rotation, qglviewer::Frame *const frame);
	std::vector<ManipulatedObject*>& getManipulateObjs()
	{
		return objects_;
	}
	void setManipulateObjs(std::vector<ManipulatedObject*>& objs)
	{
		objects_ = objs;
	}
private:
	std::vector<ManipulatedObject*> objects_;
};



