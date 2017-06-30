#include "sihouttesExposer.h"
#include "VideoEDMesh.h"
#include "FreeImage.h"
#include <QtOpenGL/QGLFramebufferObject>	
#include <QtWidgets\qmessagebox.h>

#include <math.h>
using namespace std;

#define CLEAR_COLOR_FLOAT		1
namespace videoEditting
{
	const GLenum SihouttesExposer::renderTargets[] =
	{ GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT };

	SihouttesExposer::SihouttesExposer()
	{
//		fboShader = NULL;
		uvTexData = NULL;
		norTexData = NULL;
		width = 256;
		height = 256;
		x_offeset = y_offset = 0;
		projType = GE_PROJ_PERSPECTIVE;
		depthType = GE_DEPTH_GEOMETRY;
		isProjDepthTypeUpdated = true;
		depthBufferObj = frameBufferObj = -1;
		textureObj[0] = textureObj[1] = -1;
	}

	SihouttesExposer::~SihouttesExposer()
	{
		//if (fboShader)
		//	delete fboShader;
		destroyBuffers();
	}


	bool SihouttesExposer::isFBOSupported()
	{
		return QGLFramebufferObject::hasOpenGLFramebufferObjects();
	}

	void SihouttesExposer::init(int w, int h, int _x_offeset, int _y_offset)
	{
		width = w;
		height = h;
		x_offeset = _x_offeset;
		y_offset = _y_offset;
		generateBuffers();

		//fboShader = QSharedPointer<QGLShaderProgram>(new QGLShaderProgram);
		//fboShader->addShaderFromSourceFile(QGLShader::Vertex, "./rendering/myshaders/fboShaderVS.glsl");
		//fboShader->addShaderFromSourceFile(QGLShader::Fragment, "./rendering/myshaders/fboShaderFS.glsl");
		//fboShader->link();
		//fboShader->bind();
	}

	bool SihouttesExposer::generateBuffers()
	{
		if (depthBufferObj == -1)
		{
			glGenRenderbuffersEXT(1, &depthBufferObj);
		}
		glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthBufferObj);
		glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, width, height);

		if (frameBufferObj == -1)
		{
			glGenFramebuffersEXT(1, &frameBufferObj);
		}
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, frameBufferObj);

		if (uvTexData)
		{
			char* uvBuf = (char*)uvTexData;
			delete[]uvBuf;
		}
		unsigned char* buf = new unsigned char[width * height * sizeof(UVTexPixelData) * 2];
		for (int i = 0; i < width * height * sizeof(UVTexPixelData) * 2; ++i)
		{
			buf[i] = 255;
		}
		uvTexData = (UVTexPixelData*)buf;
		norTexData = (NormalTexPixelData*)(buf + width * height * sizeof(UVTexPixelData));

		if (textureObj[0] == -1)
		{
			glGenTextures(2, textureObj);
		}
		glBindTexture(GL_TEXTURE_2D, textureObj[0]);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, uvTexData);
		glBindTexture(GL_TEXTURE_2D, textureObj[1]);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, norTexData);

		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, textureObj[0], 0);
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_2D, textureObj[1], 0);
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depthBufferObj);

		status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
		if (status != GL_FRAMEBUFFER_COMPLETE_EXT)
		{
			qDebug() << "error when binding texture to fbo" << endl;
			QMessageBox::critical(NULL,
				QObject::tr("Error"),
				QObject::tr("Fail to initialize framebuffer object."));
		}
		return status == GL_FRAMEBUFFER_COMPLETE_EXT;
	}
	bool SihouttesExposer::reclearBuffers()
	{
		//此函数得保证 init 有先调用了
		glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthBufferObj);
		glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, width, height);


		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, frameBufferObj);

		unsigned char* buf = (unsigned char*)uvTexData;
		for (int i = 0; i < width * height * sizeof(UVTexPixelData) * 2; ++i)
		{
			buf[i] = 255;
		}
		//uvTexData = (UVTexPixelData*)buf;
		//norTexData = (NormalTexPixelData*)(buf + width * height * sizeof(UVTexPixelData));

		glBindTexture(GL_TEXTURE_2D, textureObj[0]);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, uvTexData);
		glBindTexture(GL_TEXTURE_2D, textureObj[1]);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, norTexData);

		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, textureObj[0], 0);
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_2D, textureObj[1], 0);
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depthBufferObj);

		status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
		if (status != GL_FRAMEBUFFER_COMPLETE_EXT)
		{
			qDebug() << "error when binding texture to fbo" << endl;
			QMessageBox::critical(NULL,
				QObject::tr("Error"),
				QObject::tr("Fail to initialize framebuffer object."));
		}
		return status == GL_FRAMEBUFFER_COMPLETE_EXT;

	}
	void SihouttesExposer::renderGeometry()
	{
		reclearBuffers();
		// 切换framebuffer
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, frameBufferObj);

		GLint view[4];
		glPushAttrib(GL_VIEWPORT_BIT);
		//frambuffer 是整个窗口，并不会受viewport影响，viewport只会影响到渲染到的窗口
		//设置为0，0，这样使得frambuffer图片对应于渲染小窗口的图片
		glViewport(0, 0, width, height);
		glClearColor(CLEAR_COLOR_FLOAT, CLEAR_COLOR_FLOAT, CLEAR_COLOR_FLOAT, CLEAR_COLOR_FLOAT);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glDrawBuffers(2, renderTargets);
		int i = 0;

		QWeakPointer<QGLShaderProgram>meshShader = Mesh::getAppearanceShader();
		if (meshShader)
		{
			if (!meshShader.data()->bind())
			{
				std::cout << "fboShader() bind  error" << std::endl;
			}

			for (QVector<QWeakPointer<RenderableObject>>::iterator pO = sceneObjs.begin();
				pO != sceneObjs.end(); ++pO)
			{
				if (*pO)
				{
					(*pO).data()->drawAppearance(); ++i;
				}
			}
			meshShader.data()->release();
		}

		glPopAttrib();

		// 切换回原来的framebuffer
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

		// 把帧缓存数据读回内存
		glBindTexture(GL_TEXTURE_2D, textureObj[0]);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)uvTexData);   //
		glBindTexture(GL_TEXTURE_2D, textureObj[1]);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)norTexData);

		GLenum errorMsg = glGetError();
		if (errorMsg != GL_NO_ERROR)
			qDebug() << "error occurs when exposing" << endl;

		//char* bits = new char[width*height * 32];
		////opengl 图像数据从 左到 右 ，从顶到底 来进行布局
		//for (int i = 0;  i < width*height * 32; ++i)
		//{
		//}
		QImage uvTexDataimg((uchar*)uvTexData, width, height, QImage::Format_RGBA8888);
		renderedQImage = uvTexDataimg.copy().mirrored();



	}

	void SihouttesExposer::setRenderObject(const QWeakPointer<RenderableObject>& object)
	{
		sceneObjs.clear();
		sceneObjs.append(object);
	}

	//void SihouttesExposer::setRenderObject(const QVector<QWeakPointer<RenderableObject>>& objects)
	//{
	//	sceneObjs = objects;
	//}

	void SihouttesExposer::destroyBuffers()
	{
		if (uvTexData)
		{
			glDeleteTextures(2, textureObj);
			glDeleteFramebuffersEXT(1, &frameBufferObj);
			glDeleteRenderbuffersEXT(1, &depthBufferObj);
			textureObj[0] = textureObj[1] = -1;
			frameBufferObj = depthBufferObj = -1;
			delete[] uvTexData;
			uvTexData = NULL;
			norTexData = NULL;
		}
	}

	void SihouttesExposer::setResolution(int w, int h, int _x_offeset, int _y_offset)
	{
		destroyBuffers();
		width = max(w, 1);
		height = max(h, 1);
		x_offeset = _x_offeset;
		y_offset = _y_offset;
		generateBuffers();
	}

}


