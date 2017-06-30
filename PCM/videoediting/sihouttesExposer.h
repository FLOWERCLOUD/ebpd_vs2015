#pragma once
#include "RenderableObject.h"
#include <QtOpenGL/QGLShaderProgram>

// ����ฺ���ó���������������Ϣ
namespace videoEditting
{
	class SihouttesExposer
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
			unsigned int r;
			unsigned int g;
			unsigned int b;
			unsigned int a;

		};
		struct NormalTexPixelData
		{
			unsigned int r;
			unsigned int g;
			unsigned int b;
			unsigned int a;
		};

		SihouttesExposer();
		~SihouttesExposer();
		static bool isFBOSupported();

		GLuint getUVTexObj() { return textureObj[0]; }
		GLuint getNormalTexObj() { return textureObj[1]; }

		const UVTexPixelData* getUVAndIDArray()const { return uvTexData; }
		const NormalTexPixelData* getNormalAndDepthArray()const { return norTexData; }

		// ������Ҫ��ȡ������Ϣ������
		void setRenderObject(const QWeakPointer<RenderableObject>& object);
		//void setRenderObject(const QVector<QWeakPointer<RenderableObject>>& objects);

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
		void setResolution(int width, int height, int x_offeset, int y_offset);	// �������÷ֱ���
		void getResolution(int&width, int&height)
		{
			width = this->width; height = this->height;
		}
		void renderGeometry();						// ��Ⱦһ�Σ�֮��Ϳ��Զ�ȡ���漸����Ϣ
		const QImage& getRenderedQImage()
		{
			return renderedQImage;
		}
	private:
		bool generateBuffers();
		bool reclearBuffers();
		void destroyBuffers();

		ProjectionType projType;
		DepthType      depthType;
		bool isProjDepthTypeUpdated;

		// �洢����������Ϣ������,ÿ����������ռ4������������64λ
		// textureData[0] ... uv ����  [u(16), v(16), objectID(8), faceID(24)]
		// textureData[1] ... �������� [norX(16), norY(16), norZ(16), depth(16)]
		UVTexPixelData* uvTexData;
		NormalTexPixelData* norTexData;

		static const GLenum renderTargets[];
		QSharedPointer<QGLShaderProgram> fboShader;
		QVector<QWeakPointer<RenderableObject>> sceneObjs;

		GLenum status;
		GLuint width, height;
		int x_offeset, y_offset;
		int    glwidgetRes[2];
		GLuint textureObj[2];
		GLuint frameBufferObj, depthBufferObj;
		QImage renderedQImage;
	};
}
