#pragma once
#include "basic_types.h"
#include "RenderableObject.h"
#include <qpointer.h>
#include <math.h>
#include <QVector>

#define M_SIZE_FACTOR  0.08
#define M_CHECK_OBJECT(obj) if(!obj){return;}
namespace videoEditting
{
	class Camera;
	class Manipulator2
	{
	public:
		enum ManipulateAxis { M_AXIS_X = 0x1, M_AXIS_Y = 0x1 << 1, M_AXIS_Z = 0x1 << 2 };
		Manipulator2(void);
		virtual ~Manipulator2(void);

		// �жϹ�����������Ƿ��ཻ�������ཻ���ᣬ���û�н��㣬����-1
		virtual char intersect(const QVector3D& rayOri, const QVector3D& rayDir) = 0;

		// ѡ�������ʾ����
		void selectAxis(char axis) { curSelectedAxis = axis; }

		// �Ѳ��������ŵ�һ�������ϣ�֮��������������
		virtual void captureObject(QWeakPointer<RenderableObject> object);
		virtual void releaseObject();

		// ����3����������ʵ�ֲ������ı任����
		// ����������£����ָ�����������һ�������ཻʱ������beginManipulate
		// ������϶�ʱ������ goOnManipulate
		// ���ɿ����ʱ������ endManipulate
		virtual void beginManipulate(const QVector3D& rayOri, const QVector3D& rayDir, char axis) = 0;
		virtual void goOnManipulate(const QVector3D& rayOri, const QVector3D& rayDir) = 0;
		virtual void endManipulate();

		virtual void draw(Camera& camera) = 0;
		bool isManipulating() { return isWorking; }
		//QMatrix4x4 getTransformMatrix(){return m_transformMatrix;}
		QWeakPointer<RenderableObject> getCurObject() { return curObject; }

		inline ObjectTransform& getOldTransform() { return oldTransform; }
		inline ObjectTransform& getNewTransform() { return newTransform; }
	protected:
		virtual void setSize(float size) = 0;

		inline ObjectTransform* getTransform() { return &(curObject.data()->getTransform()); }
		// ���ñ��������������任����
		void setObjectTransformMatrix();

		// ��λ��ԭ���Բ������,���ڼ������Ƿ�λ��ĳһ��������
		bool intersectOriginCylinder(
			const QVector3D& rayOri, const QVector3D& rayDir,
			const char axis, const float radius,
			const float start, const float end, float& t);

		// ���������ĳһ�������ṫ�����ڸ����ϵĴ���
		float projRayToOriginAxis(
			const QVector3D& rayOri, const QVector3D& rayDir,
			const char axis);

		// ���������ĳһ��ֱ�ߵĹ������ڸ�ֱ���ϵĴ���
		float projRayToLine(
			const QVector3D& rayOri, const QVector3D& rayDir,
			const QVector3D& lineOri, const QVector3D& lineDir);

		// ��λ��ԭ���Բ����,���ڼ������Ƿ�λ��ĳһ��������
		bool intersectOriginDisk(
			const QVector3D& rayOri, const QVector3D& rayDir,
			const char axis, const float innerRadius,
			const float outerRadius, float& t);

		void transformRayToLocal(QVector3D& ori, QVector3D& dir);


		char  curSelectedAxis;		// ��¼��ǰ���ָ�����ڵ�������
		bool  isWorking;            // ��¼�Ƿ����ڱ�����
		QWeakPointer<RenderableObject> curObject;   // �����ݵ�����
		ObjectTransform oldTransform, newTransform; // ����ǰ������ı任����

		static const char otherAxis[3][2];
		static const float cosTable[13];
		static const float sinTable[13];
		static const float axisColor[3][3];
	};
}


