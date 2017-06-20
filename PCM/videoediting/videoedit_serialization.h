#pragma once
#include "VideoEdittingParameter.h"
#include "RenderableObject.h"
#include <QDataStream>
#include <QString>
#include <QList>
#include <QFile>
#include <QSharedPointer>


namespace videoEditting
{

	class VSerialization
	{
	public:
		VSerialization();
		friend QDataStream& operator<<(QDataStream& out, const VSerialization& v);
		friend QDataStream& operator >> (QDataStream& in, VSerialization& v);
		static  void open(const QString& fileName , VSerialization& v);
		static void save(const QString& fileName, VSerialization& v);
		void updateFrom();
	public:
		//video matting
		QString project_dir;
		QString videofilename;
		QString Trimap_dir;
		QString Alpha_dir;
		QString ForeGround_dir;
		QString BackGround_dir;
		int currentframePos;	//��ǰ֡��
		int totalFrameNumber;	//��Ƶ��֡��
		QVector<FrameInfo> frame;//��Ƶ֡����
		QList<int> keyFrameNo;		//�ؼ�֡��λ��
		QSet<int> initKeyframeNo;		//֡��õ�������Ĺؼ�֡����
		//pose estimation
		QVector <QSharedPointer<RenderableObject>> comon_scene; 
		QVector<QQuaternion> object_orientaion; //֡��Ŷ�Ӧ�ĳ����������꣩
		QVector<QVector3D>   object_translation;  //֡��Ŷ�Ӧ��ƽ�ƣ��������꣩
		QVector<int>         corr_frame_idx;//֡���

		QVector<QQuaternion> camera_orientaion; //֡��Ŷ�Ӧ������������������꣩
		QVector<QVector3D>   camera_translation;  //֡��Ŷ�Ӧ�������ƽ�ƣ��������꣩
		QVector<int>         corr_camera_frame_idx;//�����֡���
		QVector3D target;
		double  deltaAngle[2];
		double  deltaLength;
		double  length;
		//���������
		int projType;
		int     screenRes[2];
		int     glwidgetRes[2];
		int     viewoffeset[2];
		float  aspectRatio;        // width / height
		float fov_degree;
		float nearplane;
		float farplane;
	};
}
