
#include "manipulatedFrameSetConstraint.h"
#include "QGLViewer/frame.h"
#include "manipulate_object.h"

using namespace qglviewer;

void ManipulatedFrameSetConstraint::clearSet()
{
	objects_.clear();
}

void ManipulatedFrameSetConstraint::addObjectToSet(ManipulatedObject* o)
{
	objects_.push_back(o);
}

void ManipulatedFrameSetConstraint::constrainTranslation(qglviewer::Vec &translation, Frame *const)
{
	for ( auto it = objects_.begin(), end = objects_.end(); it != end; ++it)
		(*it)->getFrame().translate(translation);
}

void ManipulatedFrameSetConstraint::constrainRotation(qglviewer::Quaternion &rotation, Frame *const frame)
{
	// A little bit of math. Easy to understand, hard to guess (tm).
	// rotation is expressed in the frame local coordinates system. Convert it back to world coordinates.
	const Vec worldAxis = frame->inverseTransformOf(rotation.axis());
	const Vec pos = frame->position();
	const float angle = rotation.angle();

	for (std::vector<ManipulatedObject*>::iterator it = objects_.begin(), end = objects_.end(); it != end; ++it)
	{
		// Rotation has to be expressed in the object local coordinates system.
		Quaternion qObject((*it)->getFrame().transformOf(worldAxis), angle);
		(*it)->getFrame().rotate(qObject);

		// Comment these lines only rotate the objects
		Quaternion qWorld(worldAxis, angle);
		// Rotation around frame world position (pos)
		(*it)->getFrame().setPosition(pos + qWorld.rotate((*it)->getFrame().position() - pos));
	}
}