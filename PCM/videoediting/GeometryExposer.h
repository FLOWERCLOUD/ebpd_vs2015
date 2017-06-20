#pragma once 
#include "RenderableObject.h"
#include <QtOpenGL/QGLShaderProgram>
#define CLEAR_COLOR_FLOAT		1
#define USHORT_NORMALIZE_FACTOR	65535.0f
#define CLEAR_COLOR_SHORT		CLEAR_COLOR_FLOAT * 65535
#define NO_OBJECT_BIT			0xff	// ��ʾû���κ������һλ
#define FACE_ID_MASK			0xffffff

// ����ฺ���ó���������������Ϣ
namespace videoEditting
{
	class GeometryExposer
	{
	public:
		// ͶӰ�ķ�ʽ�������ʽ��Ӱ��depth��ֵ
		// �����͸��ͶӰ�����ֵΪ�۲�ռ����굽�����ԭ�������
		// ���������ͶӰ�����ֵΪ�۲�ռ�����zֵ
		enum ProjectionType { GE_PROJ_ORTHOGRAPHIC = 0, GE_PROJ_PERSPECTIVE = 1 };
		// �����ǲ��ü������zֵ���Ǳ�����������Ϊ���ֵ
		enum DepthType { GE_DEPTH_GEOMETRY, GE_DEPTH_THICKNESS };
		struct UVTexPixelData
		{
			unsigned short u, v;					// ��������
			union
			{
				unsigned int faceID;				// ��������ţ�ע�����8λ��pIDռ��
				struct {
					unsigned char dummy[3];			// ռλ�ֽڣ�û����
					unsigned char objectID;			// ��������ţ�ע��ռ������������Ÿ�8λ
				};
			};
		};
		struct NormalTexPixelData
		{
			unsigned short nx;
			unsigned short ny;
			unsigned short nz;
			unsigned short depth;
		};

		GeometryExposer();
		~GeometryExposer();
		static bool isFBOSupported();

		GLuint getUVTexObj() { return textureObj[0]; }
		GLuint getNormalTexObj() { return textureObj[1]; }

		// �����Ļĳ��������ֵ,x y ����Ϊ[0,1]ע����Ļ����ϵԭ�������Ͻ�
		// u v����,��Ӧ��λ���弰����Ϊ    (��)[u(16),       v(16), objectID(8), faceID(24)]���ߣ�
		// normal�����λ���弰����Ϊ      (��)[norX(16), norY(16),    norZ(16),  depth(16)]���ߣ�

		// ��÷������꼰��ȣ�ע�ⷨ�߿��ܱ�����Ļ
		bool getNormalAndDepth(const QVector2D& ratio, QVector3D& normal, float& depth)const;
		// ����������꣬��������ź����������
		bool getUVAndID(const QVector2D& ratio, QVector3D& uv, unsigned char& objectID, unsigned int& faceID)const;
		// ���������Ϣ
		bool getAll(const QVector2D& ratio, QVector3D& normal, float& depth, QVector2D& uv, unsigned char& objectID, unsigned int& faceID)const;
		// ���������ţ����û�����壬���Ϊ255
		bool getObjectID(const QVector2D& ratio, unsigned char& objectID)const;
		// �ж϶�Ӧλ���Ƿ�������
		bool isEmpty(const QVector2D& ratio)const;
		void getRegionFaceID(const QVector2D& minRatio, const QVector2D& maxRatio, int objectID, QSet<int>& faceIDSet)const;
		const UVTexPixelData* getUVAndIDArray()const { return uvTexData; }
		const NormalTexPixelData* getNormalAndDepthArray()const { return norTexData; }

		// ������Ҫ��ȡ������Ϣ������
		void setRenderObject(const QWeakPointer<RenderableObject>& object);
		void setRenderObject(const QVector<QWeakPointer<RenderableObject>>& objects);

		// ����ͶӰ��ʽ
		void setProjectionType(ProjectionType type)
		{
			projType = type; isProjDepthTypeUpdated = false;
		}
		void setDepthType(DepthType type)
		{
			depthType = type; isProjDepthTypeUpdated = true;
		}


		void init(int width, int height, int x_offeset, int y_offset);			// ��ʼ����������ֻ���
		void setResolution(int width, int height,int x_offeset,int y_offset);	// �������÷ֱ���
		void getResolution(int&width, int&height)
		{
			width = this->width; height = this->height;
		}
		void exposeGeometry();						// ��Ⱦһ�Σ�֮��Ϳ��Զ�ȡ���漸����Ϣ

	private:
		bool generateBuffers();
		void destroyBuffers();
		inline bool isPixelEmpty(int xID, int yID)const
		{
			return uvTexData[xID + yID * width].objectID == NO_OBJECT_BIT;
		}

		ProjectionType projType;
		DepthType      depthType;
		bool isProjDepthTypeUpdated;

		// �洢����������Ϣ������,ÿ����������ռ4������������64λ
		// textureData[0] ... uv ����  [u(16), v(16), objectID(8), faceID(24)]
		// textureData[1] ... �������� [norX(16), norY(16), norZ(16), depth(16)]
		UVTexPixelData* uvTexData;
		NormalTexPixelData* norTexData;

		static const GLenum renderTargets[];
		QGLShaderProgram* fboShader;
		QVector<QWeakPointer<RenderableObject>> sceneObjs;

		GLenum status;
		GLuint width, height;
		int x_offeset, y_offset;
		int    glwidgetRes[2];
		GLuint textureObj[2];
		GLuint frameBufferObj, depthBufferObj;
	};
}
