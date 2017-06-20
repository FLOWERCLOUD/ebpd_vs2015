#include "TranslateManipulator.h"
#include "VideoEDCamera.h"

#include <QVector>
#include <QVector3D>
#include <QQuaternion>
#include <QSharedPointer>
#include <vector>
#include <unordered_map>
using namespace std;

namespace videoEditting
{
	extern std::vector<QVector3D>     g_init_vertices;
	extern std::vector<int>           g_faces_;
	extern QVector3D                  g_init_translation;
	extern QQuaternion                g_init_rotation;
	extern std::vector<QVector3D>     g_translations;
	extern std::vector<QQuaternion>   g_rotations;
	extern std::vector<std::vector<QVector3D> > g_simulated_vertices;
	extern std::vector<std::unordered_map<int, QVector3D>> g_position_constraint; //this constraint the position of vertices of frames
	extern int g_total_frame;//total frame 
	extern int g_current_frame;
	extern float g_time_step;
}

namespace videoEditting
{
	TranslateManipulator::TranslateManipulator(void)
	{
		setSize(1.0f);
	}


	TranslateManipulator::~TranslateManipulator(void)
	{
	}

	void TranslateManipulator::draw(Camera& camera)
	{
		if (curObject.isNull())
			return;

		QVector3D center = curObject.data()->getCenter();
		QVector3D delta = camera.getOrigin() - center;
		if (!isWorking)
		{
			setSize(delta.length() * M_SIZE_FACTOR);
		}
		glLineWidth(2);
		glDisable(GL_LIGHTING);
		glDisable(GL_DEPTH_TEST);
		for (char axis = 0; axis < 3; ++axis)
		{
			if (axis == curSelectedAxis)
				glColor3f(1, 1, 0);
			else
				glColor3fv(axisColor[axis]);

			float centerV[] = { center.x(),center.y(),center.z() };
			float v[3] = { center.x(),center.y(),center.z() };
			v[axis] += axisLength;
			glBegin(GL_LINES);
			glVertex3fv(centerV);
			glVertex3fv(v);
			glEnd();

			float arrowVertex[3] = { center.x(),center.y(),center.z() };
			arrowVertex[axis] += arrowBegin;
			float arrowNormal[3];
			arrowNormal[axis] = 0.1f;
			glBegin(GL_TRIANGLE_FAN);
			glVertex3fv(v);
			for (char i = 0; i < 13; ++i)
			{
				arrowVertex[otherAxis[axis][0]] = arrowRadius * cosTable[i] + centerV[otherAxis[axis][0]];
				arrowVertex[otherAxis[axis][1]] = arrowRadius * sinTable[i] + centerV[otherAxis[axis][1]];
				glVertex3fv(arrowVertex);
				arrowNormal[otherAxis[axis][0]] = cosTable[i];
				arrowNormal[otherAxis[axis][1]] = sinTable[i];
				glNormal3fv(arrowNormal);
			}
			glEnd();
		}
		glEnable(GL_DEPTH_TEST);
	}

	char TranslateManipulator::intersect(const QVector3D& rayOri, const QVector3D& rayDir)
	{
		if (!curObject.isNull())
		{
			QVector3D ori = rayOri - curObject.data()->getCenter();
			float t;
			for (int axis = 0; axis < 3; ++axis)
			{
				if (intersectOriginCylinder(ori, rayDir, axis, detectRadius, 0, axisLength, t))
				{
					return axis;
				}
			}
		}
		return -1;
	}

	void TranslateManipulator::beginManipulate(const QVector3D& rayOri, const QVector3D& rayDir, char axis)
	{
		if (curObject.isNull())
			return;

		QVector3D center = curObject.data()->getCenter();
		isWorking = true;
		curSelectedAxis = axis;
		beginProjPos = projRayToOriginAxis(rayOri - center, rayDir, curSelectedAxis);
		beginCenter = center;
		lastPos = 0;

		oldTransform = curObject.data()->getTransform();
	}

	void TranslateManipulator::goOnManipulate(const QVector3D& rayOri, const QVector3D& rayDir)
	{
		M_CHECK_OBJECT(curObject)

			QVector3D center = curObject.data()->getCenter();
		float pos = projRayToOriginAxis(rayOri - beginCenter, rayDir, curSelectedAxis);

		float offset = pos - beginProjPos;
		float offsetV[] = { 0,0,0 };
		offsetV[curSelectedAxis] = offset - lastPos;
		lastPos = offset;
		getTransform()->translate(QVector3D(offsetV[0], offsetV[1], offsetV[2]));
		if (curObject)
		{
			if (curObject.data()->getType() == RenderableObject::OBJ_CAMERA)
			{
				((Camera*)curObject.data())->updateCameraPose();
			}
			//update curpose
			g_translations[g_current_frame] = getTransform()->getTranslate();
//			g_rotations[g_current_frame] = getTransform()->getRotate();
		}
	}

	void TranslateManipulator::setSize(float size)
	{
		axisLength = TRANS_M_AXIS_LENGTH * size;
		arrowBegin = TRANS_M_ARROW_BEGIN * size;
		arrowRadius = TRANS_M_ARROW_RADIUS * size;
		detectRadius = TRANS_M_DETECT_RADIUS * size;
	}
}


