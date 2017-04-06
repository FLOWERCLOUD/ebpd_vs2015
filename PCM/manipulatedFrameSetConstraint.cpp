#include <CustomGL\glew.h>
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




	for (auto it = objects_.begin(), end = objects_.end(); it != end; ++it)
	{
		switch ((*it)->getType())
		{
			case  ManipulatedObject::VERTEX:
			{
				qglviewer::Vec prevPos = (*it)->getWorldPosition();
				prevPos += translation;
				(*it)->setWorldPosition(prevPos);

			}
			break;
			case  ManipulatedObject::EDGE:
			{

			}
			break;
			case  ManipulatedObject::FACE:
			{

			}
			break;
			case  ManipulatedObject::OBJECT:
			{
				(*it)->getFrame().translate(translation);
			}
			break;
			case  ManipulatedObject::HANDLE:
			{
				(*it)->getFrame().translate(translation);
			}
			break;

		}


	}
		
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

void ManipulatedFrameSetLocalConstraint::clearSet()
{
	objects_.clear();
}

void ManipulatedFrameSetLocalConstraint::addObjectToSet(ManipulatedObject* o)
{
	objects_.push_back(o);
}

void ManipulatedFrameSetLocalConstraint::constrainTranslation(qglviewer::Vec &translation, Frame *const frame)
{
	LocalConstraint::constrainTranslation(translation, frame);
	for (auto it = objects_.begin(), end = objects_.end(); it != end; ++it)
	{
		switch ((*it)->getType())
		{
		case  ManipulatedObject::VERTEX:
		{
			qglviewer::Vec prevPos = (*it)->getWorldPosition();
			prevPos += translation;
			(*it)->setWorldPosition(prevPos);

		}
		break;
		case  ManipulatedObject::EDGE:
		{

		}
		break;
		case  ManipulatedObject::FACE:
		{

		}
		break;
		case  ManipulatedObject::OBJECT:
		{
			(*it)->getFrame().translate(translation);
		}
		break;
		case  ManipulatedObject::HANDLE:
		{
			qglviewer::Vec prevPos =  ((ManipulatedObjectIsHANDLE*)(*it))->getWorldPosition();
			prevPos += translation;
			((ManipulatedObjectIsHANDLE*)(*it))->setWorldPosition(prevPos);
//			(*it)->getFrame().translate(translation);
		}
		break;

		}


	}
}

void ManipulatedFrameSetLocalConstraint::constrainRotation(qglviewer::Quaternion &rotation, Frame *const frame)
{
	LocalConstraint::constrainRotation(rotation, frame);
	// A little bit of math. Easy to understand, hard to guess (tm).
	// rotation is expressed in the frame local coordinates system. Convert it back to world coordinates.
	for (auto it = objects_.begin(), end = objects_.end(); it != end; ++it)
	{
		switch ((*it)->getType())
		{
		case  ManipulatedObject::VERTEX:
		{


		}
		break;
		case  ManipulatedObject::EDGE:
		{

		}
		break;
		case  ManipulatedObject::FACE:
		{

		}
		break;
		case  ManipulatedObject::OBJECT:
		{
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
		break;
		case  ManipulatedObject::HANDLE:
		{

			for (std::vector<ManipulatedObject*>::iterator it = objects_.begin(), end = objects_.end(); it != end; ++it)
			{
				qglviewer::Quaternion local_rotate = ((ManipulatedObjectIsHANDLE*)(*it))->handle_frame_->rotation();
				qglviewer::Quaternion new_local_rotate = local_rotate*rotation;
				((ManipulatedObjectIsHANDLE*)(*it))->handle_frame_->setRotation(new_local_rotate);
			}
		}
		break;

		}


	}
}


void ManipulatedFrameSetWorldConstraint::clearSet()
{
	objects_.clear();
}

void ManipulatedFrameSetWorldConstraint::addObjectToSet(ManipulatedObject* o)
{
	objects_.push_back(o);
}

void ManipulatedFrameSetWorldConstraint::constrainTranslation(qglviewer::Vec &translation, Frame *const frame)
{
	WorldConstraint::constrainTranslation(translation, frame);
	for (auto it = objects_.begin(), end = objects_.end(); it != end; ++it)
	{
		switch ((*it)->getType())
		{
		case  ManipulatedObject::VERTEX:
		{
			qglviewer::Vec prevPos = (*it)->getWorldPosition();
			prevPos += translation;
			(*it)->setWorldPosition(prevPos);

		}
		break;
		case  ManipulatedObject::EDGE:
		{

		}
		break;
		case  ManipulatedObject::FACE:
		{

		}
		break;
		case  ManipulatedObject::OBJECT:
		{
			(*it)->getFrame().translate(translation);
		}
		break;
		case  ManipulatedObject::HANDLE:
		{
			qglviewer::Vec prevPos = ((ManipulatedObjectIsHANDLE*)(*it))->getWorldPosition();
			prevPos += translation;
			((ManipulatedObjectIsHANDLE*)(*it))->setWorldPosition(prevPos);
		}
		break;

		}


	}
}

void ManipulatedFrameSetWorldConstraint::constrainRotation(qglviewer::Quaternion &rotation, Frame *const frame)
{
	WorldConstraint::constrainRotation(rotation, frame);
	// A little bit of math. Easy to understand, hard to guess (tm).
	// rotation is expressed in the frame local coordinates system. Convert it back to world coordinates.

	for (auto it = objects_.begin(), end = objects_.end(); it != end; ++it)
	{
		switch ((*it)->getType())
		{
		case  ManipulatedObject::VERTEX:
		{


		}
		break;
		case  ManipulatedObject::EDGE:
		{

		}
		break;
		case  ManipulatedObject::FACE:
		{

		}
		break;
		case  ManipulatedObject::OBJECT:
		{
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
		break;
		case  ManipulatedObject::HANDLE:
		{

			for (std::vector<ManipulatedObject*>::iterator it = objects_.begin(), end = objects_.end(); it != end; ++it)
			{
				qglviewer::Quaternion local_rotate = ((ManipulatedObjectIsHANDLE*)(*it))->handle_frame_->rotation();
				qglviewer::Quaternion new_local_rotate = local_rotate*rotation;
				((ManipulatedObjectIsHANDLE*)(*it))->handle_frame_->setRotation(new_local_rotate);
			}
		}
		break;

		}


	}



}

ManipulatedFrameSetCameraConstraint::ManipulatedFrameSetCameraConstraint(const qglviewer::Camera* const camera)
	:CameraConstraint(camera)
{



}
void ManipulatedFrameSetCameraConstraint::clearSet()
{
	objects_.clear();
}

void ManipulatedFrameSetCameraConstraint::addObjectToSet(ManipulatedObject* o)
{
	objects_.push_back(o);
}

void ManipulatedFrameSetCameraConstraint::constrainTranslation(qglviewer::Vec &translation, Frame *const frame)
{
	CameraConstraint::constrainTranslation(translation, frame);
	for (auto it = objects_.begin(), end = objects_.end(); it != end; ++it)
	{
		switch ((*it)->getType())
		{
		case  ManipulatedObject::VERTEX:
		{
			qglviewer::Vec prevPos = (*it)->getWorldPosition();
			prevPos += translation;
			(*it)->setWorldPosition(prevPos);

		}
		break;
		case  ManipulatedObject::EDGE:
		{

		}
		break;
		case  ManipulatedObject::FACE:
		{

		}
		break;
		case  ManipulatedObject::OBJECT:
		{
			(*it)->getFrame().translate(translation);
		}
		break;
		case  ManipulatedObject::HANDLE:
		{
			qglviewer::Vec prevPos = ((ManipulatedObjectIsHANDLE*)(*it))->getWorldPosition();
			prevPos += translation;
			((ManipulatedObjectIsHANDLE*)(*it))->setWorldPosition(prevPos);
		}
		break;

		}


	}
}

void ManipulatedFrameSetCameraConstraint::constrainRotation(qglviewer::Quaternion &rotation, Frame *const frame)
{
	CameraConstraint::constrainRotation(rotation, frame);
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