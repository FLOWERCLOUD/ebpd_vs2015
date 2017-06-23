#pragma once
#include "VideoEDMesh.h"
#include <QVector>
#include <qmatrix4x4.h>
#include <qvector2d.h>
#include <QQuaternion>
#include "qpointer.h"
#include <math.h>
#include <qmath.h>


#define CAMERA_INIT_ALPHA_DEGREE                     45.0f
#define CAMERA_INIT_BETA_DEGREE                      -45.0f

//#define CAMERA_FOV_Y_DEGREE 60
//#define CAMERA_FOV_Y_RAD    CAMERA_FOV_Y_DEGREE * M_PI / 180.0f

namespace videoEditting
{
	class Camera :public Mesh
	{
	public:
		Camera(void);
		~Camera(void);
		void setupCameraShape();
		enum ProjType { GLCAMERA_ORTHO = 0, GLCAMERA_PERSP = 1 };
		void resetCamera();
		void setScreenResolution(int x, int y,int glwidget_width = 10,int glwidget_height =10);
		void getCameraViewport(int& x_offset, int& y_offset, int& width, int& height);
		void getScreenResolution(int&x, int&y) { x = screenRes[0]; y = screenRes[1]; }
		void getGLWidgetResoluiont(int& width, int& height)
		{
			width = glwidgetRes[0];
			height = glwidgetRes[1];
		}
		void setProjMode(ProjType type);
		ProjType getProjtype();

		void setFovAngle(float angle);
		float getFovAngle();
		void setAspectRatio(float aspectRatio);
		float getAspectRatio();
		void setNearPlane(float nearplane);
		float getNearPlane();
		void setFarPlane(float farplane);
		float getFarPlane();


		// ����3������ֻ�����������������û�а���Щ����Ӧ�õ�openGL
		void rotateCamera(double dx, double dy);
		void moveCamera(double dx, double dy);
		void zoomCamera(double dz);

		void applyGLMatrices();
		void applyViewMatrices();

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
		/*
		center �����Χ�Ƶ����ģ�����������ĵľ���
		*/
		void setCenter(QVector3D& center, float length = 10);

		const QVector3D& getOrigin() { return origin; }
		void setOrigin(QVector3D& in)
		{
			origin = in;
		}
		void getViewDirection(QVector3D dirs[3])
		{
			dirs[0] = direction[0];
			dirs[1] = direction[1];
			dirs[2] = direction[2];
		}
		void setViewDirection(QVector3D dirs[3])
		{
			direction[0] = dirs[0] ;
			direction[1] = dirs[1] ;
			direction[2] = dirs[2] ;
		}
		const QVector3D& getTarget()
		{
			return target;
		}
		void setTarget(QVector3D& in)
		{
			target = in;
		}
		void setLength(float in)
		{
			length = in;
		}
		void addAttachedObjects(const QWeakPointer<RenderableObject> obj);
		void clearAttachedObjects();
		void setAttachObjectOffset(float x, float y, float z);
		void setAttachObjectRotation(float angle);
		void setAttachObjectScale(float x, float y);

		void getCameraTransform(QVector3D& trans, QQuaternion& rot);
		void getScreenSize(float&w, float&h);
		void updateCameraPose();//���Ӧֻ��manipulator����
		void drawGeometry();
		void drawAppearance();
		void updateCameraLazy()
		{
			cameraShouldUpdate = true;
		}
		friend QDataStream& operator<<(QDataStream& out, const Camera&mesh);
		friend QDataStream& operator >> (QDataStream& in, Camera&mesh);
	private:
		
		void updateAll();
		void updateMesh();
		void updateAllnotMesh();
		void computeViewParam();
		void computeProjParam();
		void computeAttachedTransfrom();
		// ����ΪͶӰ������һ���ı���Щ������Ҫ��������openGLͶӰ����
		ProjType projType;
		int     screenRes[2];
		int     glwidgetRes[2];
		int     viewoffeset[2];
		float  aspectRatio;        // width / height
		float fov_degree;
		float nearplane;
		float farplane;
									// ����Ϊ�۲������һ���ı���Ҫ�������ù۲����
		QVector3D target;
		double  deltaAngle[2]; //delta angle[0] ��Ӧ��derection[1],delta angle[1] ��Ӧ��derection[0]
		double  deltaLength;
		double  length;

		// ������3����Ա���,�������һ����Ա�ı䣬��Ҫ������һ��
		QVector3D direction[3]; //���������
		QVector3D origin; //�������������
		QMatrix4x4 viewMatrix;
		QMatrix4x4 projMatrix;

		QVector<QWeakPointer<RenderableObject>> attachedObjects;
		float attachOffset[3];
		float attachRotation;
		float attachScale[2];
		bool cameraShouldUpdate;
	};

}


