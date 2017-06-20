#include "basic_types.h"
#include "VideoEDCamera.h"
#include "ObjReader.h"
#include "drawcamera.h"
#include <math.h>
#include <QtOpenGL/QGLFunctions>

using namespace std;

namespace videoEditting
{
	float CAMERA_FOV_Y_RAD(float fov_degree)
	{
		return fov_degree* M_PI / 180.0f;
	}


	Camera::Camera(void)
	{
		type = OBJ_CAMERA;
		fov_degree = 60;
		nearplane = 0.2;
		farplane = 100;
		setupCameraShape();
		resetCamera();
	}


	Camera::~Camera(void)
	{

	}

	void Camera::setupCameraShape()
	{
		QString fileName = "resource/meshes/scene/camera/cameraball.obj";
		ObjReader reader;
		if (!reader.read(fileName))
			return;
		for (int i = 0; i < reader.getNumMeshes(); ++i)
		{
			QSharedPointer<Mesh> pM(reader.getMesh(i)); //reader 读的网格会后面会自己销毁
//			pM->init();
//			common_scene->objectArray.push_back(pM);
			vertices = pM->getVertices();
			normals  = pM->getNormals();
			texcoords =  pM->gettexcoords();
			faces = pM->getFaces();
		}
	}
	void Camera::computeViewParam()
	{
		QVector3D xRotAxis = QVector3D(0, 0, direction[1].z());
		QQuaternion xRotQuat = QQuaternion::fromAxisAndAngle(xRotAxis, deltaAngle[0]);
		direction[0] = xRotQuat.rotatedVector(direction[0]);
		QQuaternion yRotQuat = QQuaternion::fromAxisAndAngle(direction[0], deltaAngle[1]);
		QQuaternion rotQuat = yRotQuat * xRotQuat;

		direction[1] = rotQuat.rotatedVector(direction[1]);
		direction[2] = rotQuat.rotatedVector(direction[2]);

		origin = target - direction[2] * length;

		viewMatrix.setToIdentity();
		viewMatrix.lookAt(origin, target, direction[1]);

	}
	void Camera::computeAttachedTransfrom()
	{
		float tanAngle = tan(CAMERA_FOV_Y_RAD(fov_degree) / 2.0f);
		double h = tanAngle * length;
		double w = h * aspectRatio;
		float xOffset = w * attachOffset[0];
		float yOffset = h * attachOffset[1];
		float zOffset = length * attachOffset[2] + 1;
		QQuaternion localQuat = QQuaternion::fromAxisAndAngle(0, 0, 1, attachRotation);
		QVector3D trans; QQuaternion rot;
		this->getCameraTransform(trans, rot);
		for (int i = 0; i < attachedObjects.size(); ++i)
		{
			if (attachedObjects[i])
			{
				RenderableObject* obj = attachedObjects[i].data();
				obj->getTransform().setRotate(localQuat);
				obj->getTransform().rotate(rot, target);
				obj->getTransform().setTranslate(origin + direction[0] * xOffset + direction[1] * yOffset + direction[2] * zOffset);
				obj->getTransform().setScale(QVector3D(attachScale[0] * w, attachScale[1] * h, 1));
			}
		}
	}

	void Camera::computeProjParam()
	{
		projMatrix.setToIdentity();
		if (projType == Camera::GLCAMERA_PERSP)
		{
			projMatrix.perspective(fov_degree, aspectRatio, nearplane, farplane);
		}
		else
		{
			float tanAngle = tan(CAMERA_FOV_Y_RAD(fov_degree) / 2.0f);
			double h = tanAngle * length;
			double w = h * aspectRatio;
			projMatrix.ortho(-w, w, -h, h, 0, 1000);
		}
	}


	void Camera::setScreenResolution(int x, int y, int glwidget_width, int glwidget_height)
	{
		screenRes[0] = x; screenRes[1] = y;
		glwidgetRes[0] = glwidget_width; glwidgetRes[1] = glwidget_height;
//		glViewport(0, 0, x, y);
		viewoffeset[0] = glwidgetRes[0] / 2 - screenRes[0] / 2;
		viewoffeset[1] = glwidgetRes[1] / 2 - screenRes[1] / 2;
		glViewport(viewoffeset[0], viewoffeset[1],
			screenRes[0], screenRes[1]);
		aspectRatio = screenRes[0] / (double)screenRes[1];
		updateAll();
	}
	void Camera::getCameraViewport(int& x_offset, int& y_offset, int& width, int& height)
	{
		x_offset = viewoffeset[0];
		y_offset = viewoffeset[1];
		width = screenRes[0];
		height = screenRes[1];
	}


	void Camera::rotateCamera(double dx, double dy)
	{
		deltaAngle[0] -= dx / 5.0;
		deltaAngle[1] -= dy / 5.0;
		updateAll();
		deltaAngle[0] = deltaAngle[1] = 0.0f;
	}

	void Camera::moveCamera(double dx, double dy)
	{
		dx /= 100; dy /= 100;
		target -= direction[0] * dx - direction[1] * dy;
		updateAll();
	}

	void Camera::zoomCamera(double dz)
	{
		float newLength = max(0.1, length - dz / 90.0f);
		deltaLength = length - newLength;
		length = newLength;
		updateAll();
		deltaLength = 0;
	}

	void Camera::getRay(int x, int y, QVector3D& ori, QVector3D& dir)
	{

		int width, height;
		int x_offset, y_offset;//y_offset 相对于左下角
		int glwidget_width, glwidget_height;
		//	camera->getScreenResolution(width, height);
		getCameraViewport(x_offset, y_offset, width, height);
		getGLWidgetResoluiont(glwidget_width, glwidget_height);
		//float xRatio = x / float(width);
		//float yRatio = y / float(height);
//		float xRatio = (x - x_offset) / float(width);
		float y_offset_reverse = glwidget_height - y_offset - height;
//		float yRatio = (y - y_offset_reverse) / float(height);
		
		
		double xRatio = (x-x_offset )/ (double)screenRes[0] * 2.0 - 1.0;
		double yRatio = 1.0 - (y- y_offset_reverse) / (double)screenRes[1] * 2.0;

		if (projType == Camera::GLCAMERA_ORTHO)
		{
			double h = tan(CAMERA_FOV_Y_RAD(fov_degree) / 2.0f) * length;
			double w = h * aspectRatio;
			xRatio *= w;
			yRatio *= h;
			ori.setX(origin.x() + xRatio * direction[0].x() + yRatio * direction[1].x());
			ori.setY(origin.y() + xRatio * direction[0].y() + yRatio * direction[1].y());
			ori.setZ(origin.z() + xRatio * direction[0].z() + yRatio * direction[1].z());
			dir = direction[2];
		}
		else
		{
			double h = tan(CAMERA_FOV_Y_RAD(fov_degree) / 2.0f);
			double w = h * aspectRatio;
			xRatio *= w;
			yRatio *= h;
			QVector3D vd = direction[2] + direction[0] * xRatio + direction[1] * yRatio;
			vd.normalize();
			ori = origin;
			dir = vd;
		}
	}

	void Camera::setProjMode(ProjType type)
	{
		projType = type;
		computeProjParam();
	}


	Camera::ProjType Camera::getProjtype()
	{
		return projType;
	}

	void Camera::setFovAngle(float angle)
	{
		fov_degree = angle;
		computeProjParam();
	}


	float Camera::getFovAngle()
	{
		return fov_degree;
	}


	void Camera::setAspectRatio(float _aspectRatio)
	{
		aspectRatio = _aspectRatio;
		computeProjParam();
	}


	float Camera::getAspectRatio()
	{
		return aspectRatio;
	}


	void Camera::setNearPlane(float _nearplane)
	{
		nearplane = _nearplane;
		computeProjParam();
	}


	float Camera::getNearPlane()
	{
		return nearplane;
	}


	void Camera::setFarPlane(float _farplane)
	{
		farplane = _farplane;
		computeProjParam();
	}


	float Camera::getFarPlane()
	{
		return farplane;
	}

	void Camera::resetCamera()
	{
		projType = GLCAMERA_PERSP;
		screenRes[0] = 640;
		screenRes[1] = 480;
		viewoffeset[0] = 0.0f;
		viewoffeset[1] = 0.0f;
		aspectRatio = screenRes[1] / (double)screenRes[0];
//		origin = QVector3D(10, 0, 0);
		origin = getCenter();
		target = QVector3D(0, 0, 0);
		direction[0] = QVector3D(0, 1, 0);
		direction[1] = QVector3D(0, 0, 1);
		direction[2] = QVector3D(-1, 0, 0);
		deltaAngle[0] = CAMERA_INIT_ALPHA_DEGREE;
		deltaAngle[1] = CAMERA_INIT_BETA_DEGREE;
		deltaLength = 0;
		length = 10.0f;
		attachOffset[0] = attachOffset[1] = attachOffset[2] = 0;
		attachRotation = 0;
		attachScale[0] = attachScale[1] = 1;

		//updateAll();
		updateAllnotMesh();
	}

	void Camera::applyGLMatrices()
	{
		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(viewMatrix.constData());
		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(projMatrix.constData());
	}


	void Camera::setCenter(QVector3D& center, float distance /*= 10*/)
	{
		target = center;
		length = distance;
		updateAll();
	}

	QVector3D Camera::getWorldPos(float xRatio, float yRatio, float zRatio)const
	{
		float xR = xRatio * 2.0 - 1.0;
		float yR = 1.0 - yRatio * 2.0;
		QVector3D ori, dir;

		if (projType == Camera::GLCAMERA_ORTHO)
		{
			double h = tan(CAMERA_FOV_Y_RAD(fov_degree) / 2.0f) * length;
			double w = h * aspectRatio;
			xR *= w;
			yR *= h;
			ori.setX(origin.x() + xR * direction[0].x() + yR * direction[1].x());
			ori.setY(origin.y() + xR * direction[0].y() + yR * direction[1].y());
			ori.setZ(origin.z() + xR * direction[0].z() + yR * direction[1].z());
			dir = direction[2];
		}
		else
		{
			double h = tan(CAMERA_FOV_Y_RAD(fov_degree) / 2.0f);
			double w = h * aspectRatio;
			xR *= w;
			yR *= h;
			QVector3D vd = direction[2] + direction[0] * xR + direction[1] * yR;
			ori = origin;
			dir = vd;
		}
		dir.normalize();
		// 	float z = depthRatio  / (1 - depthRatio);
		// 	return ori + z * dir;
		return ori + zRatio * dir;
	}

	QVector3D Camera::getWorldNormalFromView(const QVector3D& viewNormal)
	{
		return QVector3D(
			viewNormal.x() * direction[0].x() + viewNormal.y() * direction[1].x() + viewNormal.z() * -direction[2].x(),
			viewNormal.x() * direction[0].y() + viewNormal.y() * direction[1].y() + viewNormal.z() * -direction[2].y(),
			viewNormal.x() * direction[0].z() + viewNormal.y() * direction[1].z() + viewNormal.z() * -direction[2].z());
	}

	void Camera::getCameraTransform(QVector3D& trans, QQuaternion& rot)
	{
		trans = origin + direction[2];

		float xAngle = atan2(direction[0].y(), direction[0].x()) / M_PI * 180.0f;
		QQuaternion rotX = QQuaternion::fromAxisAndAngle(0, 0, 1, xAngle);
		float x = -direction[2].z();
		float y = direction[1].z();
		float yAngle = atan2(y, x) / M_PI * 180.0f;
		QVector3D newX = rotX.rotatedVector(QVector3D(1, 0, 0));
		QQuaternion rotY = QQuaternion::fromAxisAndAngle(newX, yAngle);
		rot = rotY * rotX;

	}

	void Camera::getScreenSize(float&w, float&h)
	{
		if (projType == Camera::GLCAMERA_ORTHO)
		{
			h = 2 * tan(CAMERA_FOV_Y_RAD(fov_degree) / 2.0f) * length;
		}
		else
			h = 2 * tan(CAMERA_FOV_Y_RAD(fov_degree) / 2.0f);
		w = h * aspectRatio;
	}
	void Camera::updateCameraPose()
	{

		const QQuaternion& rotation = getTransform().getRotate();
		const QVector3D& translation = getTransform().getScale();
		const QVector3D& scale = getTransform().getTranslate();


	}

	void Camera::drawGeometry()
	{
		Mesh::drawGeometry();
	}


	void Camera::drawAppearance()
	{
		Mesh::drawAppearance();
		appearProgram->release();

		glPushMatrix();

		glMultMatrixf(transform.getTransformMatrix().constData());
		//QGLFunctions glFuncs(QGLContext::currentContext());
		//glFuncs.glEnableVertexAttribArray(PROGRAM_VERTEX_ATTRIBUTE);
		//glFuncs.glEnableVertexAttribArray(PROGRAM_NORMAL_ATTRIBUTE);
		glEnable(GL_COLOR_MATERIAL);
//		drawCamera();
      // 再画透明的物体
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glDepthMask(GL_FALSE);


		float fov_half_rad = CAMERA_FOV_Y_RAD(fov_degree / 2);
		drawFrustum(0, -nearplane*tan(fov_half_rad)*aspectRatio, nearplane*tan(fov_half_rad)*aspectRatio,
			-nearplane*tan(fov_half_rad), nearplane*tan(fov_half_rad),
			nearplane, farplane);
		glDepthMask(GL_TRUE);
		glDisable(GL_BLEND);

		glPopMatrix();
		glDisable(GL_COLOR_MATERIAL);
		//glFuncs.glDisableVertexAttribArray(PROGRAM_VERTEX_ATTRIBUTE);
		//glFuncs.glDisableVertexAttribArray(PROGRAM_NORMAL_ATTRIBUTE);
		appearProgram->bind();
	}

	QDataStream& operator<<(QDataStream& out, const Camera& mesh)
	{
		out << mesh.vertices << mesh.normals << mesh.texcoords << mesh.faces;
			
		//	<< mesh.canvas;
		return out;
	}

	QDataStream& operator >> (QDataStream& in, Camera& mesh)
	{
		in >> mesh.vertices >> mesh.normals >> mesh.texcoords >> mesh.faces;
		//		>> mesh.canvas;
		mesh.init();
		return in;
	}


	void Camera::addAttachedObjects(const QWeakPointer<RenderableObject> obj)
	{
		attachedObjects.push_back(obj);
		updateAll();
	}

	void Camera::clearAttachedObjects()
	{
		attachedObjects.clear();
	}

	void Camera::setAttachObjectOffset(float x, float y, float z)
	{
		attachOffset[0] = x;
		attachOffset[1] = y;
		attachOffset[2] = z;
		updateAll();
	}

	void Camera::setAttachObjectRotation(float angle)
	{
		attachRotation = angle;
		updateAll();
	}

	void Camera::setAttachObjectScale(float x, float y)
	{
		attachScale[0] = x;
		attachScale[1] = y;
		updateAll();
	}

	void Camera::updateAll()
	{

		computeViewParam();
		computeProjParam();
		computeAttachedTransfrom();
		deltaAngle[0] = deltaAngle[1] = 0;
		deltaLength = 0;
		updateMesh();

	}
	void  Camera::updateMesh()
	{
		QVector3D trans;
		QQuaternion rot;
		getCameraTransform(trans, rot);
		getTransform().setTranslate(trans);
		getTransform().setRotate(rot);
	}
	void Camera::updateAllnotMesh()
	{
		computeViewParam();
		computeProjParam();
		computeAttachedTransfrom();
		deltaAngle[0] = deltaAngle[1] = 0;
		deltaLength = 0;
	}
}




