#pragma once
#include "manipulator2.h"
#define TRANS_M_AXIS_LENGTH        4.0f
#define TRANS_M_ARROW_BEGIN        3.0f
#define TRANS_M_ARROW_RADIUS       0.15f
#define TRANS_M_DETECT_RADIUS      0.5f
namespace videoEditting
{
	class Camera;
	class TranslateManipulator :
		public Manipulator2
	{
	public:
		TranslateManipulator(void);
		virtual ~TranslateManipulator(void);

		void draw(Camera& camera);
		char intersect(const QVector3D& rayOri, const QVector3D& rayDir);


		void beginManipulate(const QVector3D& rayOri, const QVector3D& rayDir, char axis);
		void goOnManipulate(const QVector3D& rayOri, const QVector3D& rayDir);
	private:
		void setSize(float size);
		float beginProjPos;            // ��ʼ�ƶ�ʱͶӰλ��
		float lastPos;
		QVector3D beginCenter;        // ��ʼ�ƶ�ʱ����λ��

		float axisLength;
		float arrowBegin;
		float arrowRadius;
		float detectRadius;
	};


}

