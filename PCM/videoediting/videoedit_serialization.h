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
		int currentframePos;	//当前帧数
		int totalFrameNumber;	//视频总帧数
		QVector<FrameInfo> frame;//视频帧序列
		QList<int> keyFrameNo;		//关键帧的位置
		QSet<int> initKeyframeNo;		//帧差法得到的有序的关键帧序列
		//pose estimation
		QVector <QSharedPointer<RenderableObject>> comon_scene; 
		QVector<QQuaternion> object_orientaion; //帧序号对应的朝向（世界坐标）
		QVector<QVector3D>   object_translation;  //帧序号对应的平移（世界坐标）
		QVector<int>         corr_frame_idx;//帧序号

		QVector<QQuaternion> camera_orientaion; //帧序号对应的摄像机朝向（世界坐标）
		QVector<QVector3D>   camera_translation;  //帧序号对应的摄像机平移（世界坐标）
		QVector<int>         corr_camera_frame_idx;//摄像机帧序号
		QVector3D target;
		double  deltaAngle[2];
		double  deltaLength;
		double  length;
		//摄像机参数
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
