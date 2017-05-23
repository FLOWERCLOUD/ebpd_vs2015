#pragma once
#include <QVector>
#include <qmatrix4x4.h>
#include <qvector2d.h>
#include <QQuaternion>
#include "qpointer.h"
#include <math.h>
#include <qmath.h>
#include "RenderableObject.h"

#define CAMERA_INIT_ALPHA_DEGREE                     45.0f
#define CAMERA_INIT_BETA_DEGREE                      -45.0f

#define CAMERA_FOV_Y_DEGREE 60
#define CAMERA_FOV_Y_RAD    CAMERA_FOV_Y_DEGREE * M_PI / 180.0f
namespace videoEditting
{
	class Camera
	{
	public:
		Camera(void);
		~Camera(void);

		enum ProjType { GLCAMERA_ORTHO = 0, GLCAMERA_PERSP = 1 };
		void resetCamera();
		void setScreenResolution(int x, int y);
		void getScreenResolution(int&x, int&y) { x = screenRes[0]; y = screenRes[1]; }
		void setProjMode(ProjType type);

		// ����3������ֻ�����������������û�а���Щ����Ӧ�õ�openGL
		void rotateCamera(double dx, double dy);
		void moveCamera(double dx, double dy);
		void zoomCamera(double dz);

		void applyGLMatrices();

		void getRay(int x, int y, QVector3D& ori, QVector3D& dir);

		QMatrix4x4 getViewMatrix() { return viewMatrix; }
		QMatrix4x4 getProjMatrix() { return projMatrix; }

		// ������Ļһ���λ���Լ����ֵ(zRatio)���������ռ�ĵ�
		QVector3D  getWorldPos(float xRatio, float yRatio, float zRatio)const;
		// �ѹ��ڹ۲�����ϵ�ķ��߱任����������
		QVector3D  getWorldNormalFromView(const QVector3D& viewNormal);
		// ��λ����Ļ����
		QVector2D  getScreenRatio(const QVector2D& screenCoord)
		{
			return QVector2D(screenCoord.x() / screenRes[0], screenCoord.y() / screenRes[1]);
		}

		void setCenter(QVector3D& center, float length = 10);

		const QVector3D& getOrigin() { return origin; }

		void addAttachedObjects(const QWeakPointer<RenderableObject> obj);
		void clearAttachedObjects();
		void setAttachObjectOffset(float x, float y, float z);
		void setAttachObjectRotation(float angle);
		void setAttachObjectScale(float x, float y);

		void getCameraTransform(QVector3D& trans, QQuaternion& rot);
		void getScreenSize(float&w, float&h);
	private:

		void updateAll();
		void computeViewParam();
		void computeProjParam();
		void computeAttachedTransfrom();
		// ����ΪͶӰ������һ���ı���Щ������Ҫ��������openGLͶӰ����
		ProjType projType;
		int     screenRes[2];
		double  aspectRatio;        // width / height

									// ����Ϊ�۲������һ���ı���Ҫ�������ù۲����
		QVector3D target;
		double  deltaAngle[2];
		double  deltaLength;
		double  length;

		// ������3����Ա���,�������һ����Ա�ı䣬��Ҫ������һ��
		QVector3D direction[3];
		QVector3D origin;
		QMatrix4x4 viewMatrix;
		QMatrix4x4 projMatrix;

		QVector<QWeakPointer<RenderableObject>> attachedObjects;
		float attachOffset[3];
		float attachRotation;
		float attachScale[2];
	};

}


