#pragma once

#include "QGLViewer/constraint.h"
class ManipulatedObject;

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