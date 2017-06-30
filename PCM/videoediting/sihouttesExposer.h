#pragma once
#include "RenderableObject.h"
#include <QtOpenGL/QGLShaderProgram>

// 这个类负责获得场景几何体表面的信息
namespace videoEditting
{
	class SihouttesExposer
	{
	public:
		// 投影的方式，这个方式会影响depth的值
		// 如果是透视投影，深度值为观察空间坐标到摄像机原点的连线
		// 如果是正交投影，深度值为观察空间坐标z值
		enum ProjectionType { GE_PROJ_ORTHOGRAPHIC = 0, GE_PROJ_PERSPECTIVE = 1 };
		// 决定是采用几何体的z值还是表面纹理厚度作为深度值
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

		// 设置需要获取表面信息的物体
		void setRenderObject(const QWeakPointer<RenderableObject>& object);
		//void setRenderObject(const QVector<QWeakPointer<RenderableObject>>& objects);

		// 设置投影方式
		void setProjectionType(ProjectionType type)
		{
			projType = type; isProjDepthTypeUpdated = false;
		}
		void setDepthType(DepthType type)
		{
			depthType = type; isProjDepthTypeUpdated = true;
		}


		void init(int width, int height, int x_offeset, int y_offset);			// 初始化，分配各种缓存
		void setResolution(int width, int height, int x_offeset, int y_offset);	// 重新设置分辨率
		void getResolution(int&width, int&height)
		{
			width = this->width; height = this->height;
		}
		void renderGeometry();						// 渲染一次，之后就可以读取表面几何信息
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

		// 存储场景几何信息的纹理,每个纹理像素占4个短整数，共64位
		// textureData[0] ... uv 纹理  [u(16), v(16), objectID(8), faceID(24)]
		// textureData[1] ... 法线纹理 [norX(16), norY(16), norZ(16), depth(16)]
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
