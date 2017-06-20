#pragma once
#include "basic_types.h"
#include "videoEditingCommonType.h"
#include <qimage.h>
#include <qdatastream.h>
#include <qvector4d.h>
#define UPDATE_REGION_INVALID_COORD 65535		// ���һ�����Ϸ��ľ��θ�������Ҳ���ǲ��ø��£�
#define TEX_DIM 1024
#define CANVAS_MAX_SIZE	4096					// һ��ͼ������ߴ�Ϊ CANVAS_MAX_SIZE * CANVAS_MAX_SIZE
// һ����ɫ�������شӸ�λ����λ�� ARGB ����
#define R_I2F(x)   ((x >>16) & 0xff) / 255.0f	
#define G_I2F(x)   ((x >> 8) & 0xff) / 255.0f
#define B_I2F(x)        ((x) & 0xff) / 255.0f
#define A_I2F(x)   ((x >>24) & 0xff) / 255.0f
// һ��surface�������شӸ�λ����λ�� IOR ���� �߹� ����  ����
#define RL_I2F(x)        ((x) & 0xff) / 255.0f
#define GL_I2F(x)   ((x >> 8) & 0xff) / 255.0f
#define RR_I2F(x)   ((x >>16) & 0xff) / 255.0f
#define ID_I2F(x)   ((x >>24) & 0xff) / 255.0f

#define INIT_COLOR_BITS 0x00ffffff
#define INIT_BACKBROUND_COLOR_BITS 0xffffffff
#define INIT_SURF_BITS  0x00ff0000
#define INIT_THICKNESS  0.0f
namespace videoEditting
{



	class Canvas
	{
	public:
		Canvas();
		~Canvas(void);


		void init(
			QVector<QVector3D>& vertices,
			QVector<QVector3D>& normals,
			QVector<QVector2D>& texcoords,
			QVector<ObjTriangle>& faces);

		void release();

		void getResolution(int& width, int& height)
		{
			width = this->width; height = this->height;
		}
		inline float getTotalThicknessPixel(int x, int y)
		{
			return totalThick[y * width + x];
		}
		inline QVector4D getTotalColorPixel(int x, int y)
		{
			unsigned &c = totalColor[y * width + x];
			return QVector4D(R_I2F(c), G_I2F(c), B_I2F(c), A_I2F(c));
		}

		inline QVector4D getTotalSurfacePixel(int x, int y)
		{
			unsigned &c = totalSurf[y * width + x];
			return QVector4D(RL_I2F(c), GL_I2F(c), RR_I2F(c), ID_I2F(c));
		}
		void setCurLayerFromImage(const QImage& image);

		// openGL����
		void updateGLTextures();		// ���޸ĺ��ͼ���͵�openGL
		GLint getGLColorTexObj() { return glColorTexObj; }
		GLint getGLSurfTexObj() { return glSurfTexObj; }
		GLint getGLThicknessTexObj() { return glThicknessTexObj; }
		GLint getGLBaseThicknessTexObj() { return glBaseThicknessTexObj; }

		friend QDataStream& operator<<(QDataStream& out, const Canvas& canvas);
		friend QDataStream& operator >> (QDataStream& in, Canvas& canvas);
	private:
		void updateAll();


		QImage colorImage, surfaceImage;// �洢���ظ�ʽ
		int width, height;
		// ��¼ͼ�������Ч�������ݣ�ָ����ҪԤ�ȷ���ռ�
		float * curLayerBaseThickness;	// ��ǰͼ�����µ�ͼ����ܺ��
		unsigned int* totalColor;
		unsigned int* totalSurf;
		float*        totalThick;
		// opengl �������
		GLuint glColorTexObj;
		GLuint glSurfTexObj;
		GLuint glThicknessTexObj;
		GLuint glBaseThicknessTexObj;

	};
}