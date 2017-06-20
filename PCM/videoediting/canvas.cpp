#include "canvas.h"

namespace videoEditting
{
	Canvas::Canvas()
	{
		curLayerBaseThickness = NULL;
		totalColor = totalSurf = NULL;
		totalThick = NULL;
		width = height = 0;

	}

	Canvas::~Canvas(void)
	{
		this->release();
	}
	void Canvas::init(
		QVector<QVector3D>& vertices,
		QVector<QVector3D>& normals,
		QVector<QVector2D>& texcoords,
		QVector<ObjTriangle>& faces)
	{
		int &width = this->width;
		int &height = this->height;
		if (width == 0 || height == 0)
		{	// 默认大小
			width = height = TEX_DIM;
		}
		if (!curLayerBaseThickness)
		{
			curLayerBaseThickness = new float[width * height];
		}
		if (!totalColor)
		{
			totalColor = new unsigned[width * height];
		}
		if (!totalSurf)
		{
			totalSurf = new unsigned[width * height];
		}
		if (!totalThick)
		{
			totalThick = new float[width * height];
		}


		for (int i = 0; i < width * height; ++i)
		{
			totalColor[i] = INIT_BACKBROUND_COLOR_BITS;
			totalSurf[i] = INIT_SURF_BITS;
			totalThick[i] = INIT_THICKNESS;
			curLayerBaseThickness[i] = 0;

		}


		// 分配opengl纹理
		glGenTextures(1, &glColorTexObj);
		glBindTexture(GL_TEXTURE_2D, glColorTexObj);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, totalColor);

		glGenTextures(1, &glSurfTexObj);
		glBindTexture(GL_TEXTURE_2D, glSurfTexObj);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, totalSurf);

		glGenTextures(1, &glThicknessTexObj);
		glBindTexture(GL_TEXTURE_2D, glThicknessTexObj);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA16, width, height, 0, GL_ALPHA, GL_FLOAT, totalThick);



		glGenTextures(1, &glBaseThicknessTexObj);
		glBindTexture(GL_TEXTURE_2D, glBaseThicknessTexObj);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA16, width, height, 0, GL_ALPHA, GL_FLOAT, curLayerBaseThickness);


		updateAll();
	}

	void Canvas::setCurLayerFromImage(const QImage& image)
	{
		QImage scaledImg = image.scaled(width, height, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
		colorImage = scaledImg;
	}
	void Canvas::updateGLTextures()
	{
		updateAll();
	}
	void Canvas::updateAll()
	{
		// 发送图层数组到openGL
		glBindTexture(GL_TEXTURE_2D, glColorTexObj);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, totalColor);

		glBindTexture(GL_TEXTURE_2D, glSurfTexObj);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, totalSurf);

		glBindTexture(GL_TEXTURE_2D, glThicknessTexObj);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA16, width, height, 0, GL_ALPHA, GL_FLOAT, totalThick);

		glBindTexture(GL_TEXTURE_2D, glBaseThicknessTexObj);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA16, width, height, 0, GL_ALPHA, GL_FLOAT, curLayerBaseThickness);
	}
	void Canvas::release()
	{
		if (curLayerBaseThickness)
		{
			delete[] curLayerBaseThickness;
			curLayerBaseThickness = NULL;
		}
		if (totalColor)
		{
			delete[] totalColor;
			totalColor = NULL;
		}
		if (totalSurf)
		{
			delete[] totalSurf;
			totalSurf = NULL;
		}
		if (totalThick)
		{
			delete[] totalThick;
			totalThick = NULL;
		}

		glDeleteTextures(1, &glColorTexObj);
		glDeleteTextures(1, &glSurfTexObj);
		glDeleteTextures(1, &glThicknessTexObj);
		glDeleteTextures(1, &glBaseThicknessTexObj);
	}
	QDataStream& operator<<(QDataStream& out, const Canvas& canvas)
	{
		out << canvas.width << canvas.height;
		return out;
	}

	QDataStream& operator >> (QDataStream& in, Canvas& canvas)
	{
		int selectedLayer;
		in >> canvas.width >> canvas.height;

		return in;
	}


}