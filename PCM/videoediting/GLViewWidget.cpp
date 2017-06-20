#include "basic_types.h"
#include "HistoryCommand.h"
#include "Mesh.h"
#include "GLViewWidget.h"
#include "GlobalObject.h"
#include "VideoEditingWindow.h"
#include "toolbox\gl_utils\glassert.h"
#include "bulletInterface.h"
#include <QtGui>
#include <QtOpenGL>
#include <math.h>
using namespace std;

static void drawArrow(qreal length = 1.0, qreal radius = -1.0, int nbSubdivisions = 12)
{
	static GLUquadric* quadric = gluNewQuadric();

	if (radius < 0.0)
		radius = 0.05 * length;

	const qreal head = 2.5*(radius / length) + 0.1;
	const qreal coneRadiusCoef = 4.0 - 5.0 * head;

	gluCylinder(quadric, radius, radius, length * (1.0 - head / coneRadiusCoef), nbSubdivisions, 1);
	glTranslated(0.0, 0.0, length * (1.0 - head));
	gluCylinder(quadric, coneRadiusCoef * radius, 0.0, head * length, nbSubdivisions, 1);
	glTranslated(0.0, 0.0, -length * (1.0 - head));
}

static void drawAxis(qreal length)
{

	const qreal charWidth = length / 40.0;
	const qreal charHeight = length / 30.0;
	const qreal charShift = 1.04 * length;

	float color[4];
	color[0] = 0.0f;  color[1] = 0.0f;  color[2] = 1.0f;  color[3] = 1.0f;
	glColor4fv(color);
//	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
// The Z
	glBegin(GL_LINES);
	glVertex3d(-charWidth, charHeight, charShift);
	glVertex3d(charWidth, charHeight, charShift);
	glVertex3d(charWidth, charHeight, charShift);
	glVertex3d(-charWidth, -charHeight, charShift);
	glVertex3d(-charWidth, -charHeight, charShift);
	glVertex3d(charWidth, -charHeight, charShift);
	glEnd();
	drawArrow(length, 0.01*length);

	color[0] = 1.0f;  color[1] = 0.0f;  color[2] = 0.0f;  color[3] = 1.0f;
//	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
	glColor4fv(color);
	glBegin(GL_LINES);
	// The X
	glVertex3d(charShift, charWidth, -charHeight);
	glVertex3d(charShift, -charWidth, charHeight);
	glVertex3d(charShift, -charWidth, -charHeight);
	glVertex3d(charShift, charWidth, charHeight);
	glEnd();
	glPushMatrix();
	glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
	drawArrow(length, 0.01*length);
	glPopMatrix();

	color[0] = 0.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
//	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
	glColor4fv(color);
	// The Y
	glBegin(GL_LINES);
	glVertex3d(charWidth, charShift, charHeight);
	glVertex3d(0.0, charShift, 0.0);
	glVertex3d(-charWidth, charShift, charHeight);
	glVertex3d(0.0, charShift, 0.0);
	glVertex3d(0.0, charShift, 0.0);
	glVertex3d(0.0, charShift, -charHeight);
	glEnd();

	glPushMatrix();
	glRotatef(-90.0f, 1.0f, 0.0f, 0.0f);
	drawArrow(length, 0.01*length);
	glPopMatrix();

}


namespace videoEditting
{

	void drawSimulateObjects(videoEditting::Camera& camera)
	{

		glEnable(GL_LIGHTING);
		glEnable(GL_COLOR_MATERIAL);
		float Diffuse[4] = { 1,0,0,1 };
		glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
		glMaterialfv(GL_FRONT, GL_DIFFUSE, Diffuse);
//		glDisable(GL_LIGHTING);

		camera.applyGLMatrices();
//		glMatrixMode(GL_MODELVIEW);
//		glPushMatrix();


		int tmp_cur_fram = g_current_frame;
		std::vector<std::vector<QVector3D>> tmp_simulated_vertices = g_simulated_vertices;
		std::vector<std::vector<QVector3D>> tmp_simulated_normal = g_simulated_normals;
		std::vector<QVector3D> tmp_translation = g_translations;
		std::vector<QQuaternion> tmp_rotation = g_rotations;
		if (g_current_frame < g_simulated_vertices.size() && g_current_frame>=0)
		{
			std::vector<QVector3D>& simulateObj= g_simulated_vertices[g_current_frame];
			std::vector<QVector3D> simulateObjNormal;
			if(g_current_frame < g_simulated_normals.size())
				simulateObjNormal = g_simulated_normals[g_current_frame];
			
			int vtx_size = simulateObj.size();
			glColor3f(1.0f, 0.5f, 0.5f);

//			glMultMatrixf(camera.getTransform().getTransformMatrix().constData());
			QMatrix4x4 pose;
			pose.translate(g_translations[g_current_frame]);
			pose.rotate(g_rotations[g_current_frame]);
			glMultMatrixf(pose.constData());
			std::vector<int> tmp_faces = g_faces_;
			for (int i = 0; i < g_faces_.size()/3; ++i)
			{
				glBegin(GL_TRIANGLES);
				if(g_faces_[3 * i + 0] <  simulateObjNormal.size())
				{
					glNormal3f(simulateObjNormal[g_faces_[3 * i + 0]].x(), simulateObjNormal[g_faces_[3 * i + 0]].y(), simulateObjNormal[g_faces_[3 * i + 0]].z());
					
				}
				if (g_faces_[3 * i + 0] < simulateObj.size())
				{
					glVertex3f(simulateObj[g_faces_[3 * i + 0]].x(), simulateObj[g_faces_[3 * i + 0]].y(), simulateObj[g_faces_[3 * i + 0]].z());
				}
				
				if (g_faces_[3 * i + 1] < simulateObjNormal.size())
				{
					glNormal3f(simulateObjNormal[g_faces_[3 * i + 1]].x(), simulateObjNormal[g_faces_[3 * i + 1]].y(), simulateObjNormal[g_faces_[3 * i + 1]].z());

				}
				if (g_faces_[3 * i + 1] < simulateObj.size())
				{

					glVertex3f(simulateObj[g_faces_[3 * i + 1]].x(), simulateObj[g_faces_[3 * i + 1]].y(), simulateObj[g_faces_[3 * i + 1]].z());
				}
				if (g_faces_[3 * i + 2] < simulateObjNormal.size())
				{
					glNormal3f(simulateObjNormal[g_faces_[3 * i + 2]].x(), simulateObjNormal[g_faces_[3 * i + 2]].y(), simulateObjNormal[g_faces_[3 * i + 2]].z());
				}
				if (g_faces_[3 * i + 2] < simulateObj.size())
				{
					glVertex3f(simulateObj[g_faces_[3 * i + 2]].x(), simulateObj[g_faces_[3 * i + 2]].y(), simulateObj[g_faces_[3 * i + 2]].z());
				}
				glEnd();
			}

			GLenum errorMsg = glGetError();
			if (errorMsg != GL_NO_ERROR)
				qDebug() << "error occurs before draw points" << endl;

			glDisable(GL_COLOR_MATERIAL);
			glDisable(GL_LIGHTING);

			//std::vector<int> constrainted_nodes = g_constrainted_nodes;
			//glDisable(GL_DEPTH_TEST);
			//glColor3f(1.0f, 0.0f, 0.0f);
			//glPointSize(20);
			//glBegin(GL_POINTS);
			//for (int i = 0; i < constrainted_nodes.size(); ++i)
			//{
			//	glAssert(glVertex3f(g_simulated_vertices[g_current_frame][constrainted_nodes[i]].x(),
			//		g_simulated_vertices[g_current_frame][constrainted_nodes[i]].y(),
			//		g_simulated_vertices[g_current_frame][constrainted_nodes[i]].z()));
			//}
			//glEnd();
			//glEnable(GL_DEPTH_TEST);
			
		}
//		glPopMatrix();

		GLenum errorMsg = glGetError();
		if (errorMsg != GL_NO_ERROR)
			qDebug() << "error occurs when drawSimulateObjects" << endl;
	}



	void OGL_widget_skin_hidden::initializeGL() 
	{
		//OGL_widget_skin::init_glew();
		//init_opengl();
		makeCurrent();
		unsigned err = glewInit();
		if (GLEW_OK != err)
		{
			const GLubyte *e = glewGetErrorString(err);
			QMessageBox::critical(NULL, QObject::tr("Error"), QObject::tr("GLEW initialization error."));
			qDebug() << e;
		}

		if (glewIsSupported("GL_EXT_framebuffer_object"))
			qDebug() << "Old EXT FBO available" << endl;
		else
		{
			qDebug() << "Old EXT FBO NOT available" << endl;
			QMessageBox::critical(NULL, QObject::tr("Error"), QObject::tr("Old EXT FBO NOT available"));
		}

		if (glewIsSupported("GL_ARB_framebuffer_object"))
			qDebug() << "Newer ARB FBO available" << endl;
		else
		{
			qDebug() << "Newer ARB FBO NOT available" << endl;
			//QMessageBox::critical(NULL, QObject::tr("Error"), QObject::tr("New ARB FBO NOT available"));
		}


	}



	GLViewWidget::GLViewWidget(QWidget* parent, QGLWidget* sh) :QGLWidget(QGLFormat(QGL::SampleBuffers), parent,sh),
		scene(sh->context())
	{
		backgroundClr.setRgb(150, 150, 150);
		setFocusPolicy(Qt::ClickFocus);
		setMouseTracking(true);
		isAltDown = false;
		isCtrlDown = false;
		isShiftDown = false;
		curTool = NULL;
		curToolType = TOOL_SELECT;
		isPickMode = false;
		pickCallBackFun = NULL;
		backgroundtexture = -1;
		isbackgroundChanged = true;
		background_image = QImage(400, 200, QImage::Format_RGB888);
		background_image.fill(QColor(128.0, 128.0, 128.0, 255.0));
		background_image = QImage("C:/Users/hehua2015/Pictures/back1.jpg");
		cur_rendermode = solid | image_resolution;
	}

	GLViewWidget::~GLViewWidget(void)
	{
	}

	void GLViewWidget::changeRenderMode(unsigned int rendermode)
	{
		updateGL();
	}
	void GLViewWidget::rd_mode_toolb_wire_transc()
	{
		cur_rendermode &= L_MASK;
		cur_rendermode |= wireframe_transparent;
		changeRenderMode(cur_rendermode);
	}
	void GLViewWidget::rd_mode_toolb_wire()
	{
		cur_rendermode &= L_MASK;
		cur_rendermode |= wireframe;
		changeRenderMode(cur_rendermode);
	}
	void GLViewWidget::rd_mode_toolb_solid()
	{
		cur_rendermode &= L_MASK;
		cur_rendermode |= solid;
		changeRenderMode(cur_rendermode);
	}
	void GLViewWidget::rd_mode_toolb_tex()
	{
		cur_rendermode &= L_MASK;
		cur_rendermode |= texture;
		changeRenderMode(cur_rendermode);
	}
	void GLViewWidget::rd_mode_toolb_video_background_tex()
	{
		cur_rendermode &= L_MASK;
		cur_rendermode |= video_background_texture;
		changeRenderMode(cur_rendermode);
	}
	void GLViewWidget::rd_mode_image_resolution()
	{
		cur_rendermode &= R_MASK;
		cur_rendermode |= image_resolution;
		changeRenderMode(cur_rendermode);
	}
	void GLViewWidget::rd_mode_tex_glwidget_resolution()
	{
		cur_rendermode &= R_MASK;
		cur_rendermode |= glwidget_resolution;
		changeRenderMode(cur_rendermode);
	}

	void GLViewWidget::initializeGL()
	{
		qglClearColor(backgroundClr);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glShadeModel(GL_SMOOTH);

		GLfloat	light_position[] = { 1.0f, 1.0f, 1.0f, 1.0f };
		glLightfv(GL_LIGHT0, GL_POSITION, light_position);

		// Setup light parameters
		glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE); /*GL_FALSE*/
		glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);

		glEnable(GL_LIGHT0);		// Enable Light 0
		glEnable(GL_LIGHTING);

		//////////////////////////////////////////////////////////////////////////

		/* to use facet color, the GL_COLOR_MATERIAL should be enabled */
		glEnable(GL_COLOR_MATERIAL);
		glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
		glMateriali(GL_FRONT_AND_BACK, GL_SHININESS, 128);

		scene.init();
		// QMessageBox::information(NULL, QObject::tr("Info"), QObject::tr("All initialized."));
	}

	void GLViewWidget::paintGL()
	{
		makeCurrent();

		int viewport[4];
		glGetIntegerv(GL_VIEWPORT, viewport);
		glViewport(0, 0, glwidget_width, glwidget_height);
		glClearColor(0.8f, 0.8f , 0.8f , 1.0f);	
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		if (-1 == backgroundtexture)
		{
			glGenTextures(1, &backgroundtexture);
		}
		

		if (isbackgroundChanged)
		{
			glEnable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D, backgroundtexture);
			isbackgroundChanged = false;
			QImage GL_formatted_image;
			GL_formatted_image = QGLWidget::convertToGLFormat(background_image);
			GL_formatted_image.save("C:/Users/hehua2015/Pictures/back33.jpg");
			if (GL_formatted_image.isNull())
			{
				std::cerr << "error GL_formatted_image" << std::endl;
				exit(1);
			}
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glAssert(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
				GL_formatted_image.width(), GL_formatted_image.height(),
				0, GL_RGBA, GL_UNSIGNED_BYTE, GL_formatted_image.bits()));

			glBindTexture(GL_TEXTURE_2D, 0);
			glDisable(GL_TEXTURE_2D);
		}
		

		//下面的函数用于设置与背景相同比例的viewport
		glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);//resizeGL中已设好viewport，故这里可以直接使用
		glEnable(GL_SCISSOR_TEST);
		glScissor(viewport[0], viewport[1], viewport[2], viewport[3]);
		if (curToolType == TOOL_PAINT)
			glClearColor(94.0f / 255.0f, 111.0f / 255.0f, 138.0f / 255.0f, 1.0f);
		else if (curToolType == TOOL_FACE_SELECT)
			glClearColor(102.0f / 255.0f, 91.0f / 255.0f, 164.0f / 255.0f, 1.0f);
		else
			glClearColor(0.5, 0.5, 0.5, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);//如果只用viewport而不要scissor,glclear仍会清掉所有的东西
		glDisable(GL_SCISSOR_TEST);




		//画出viewport框
		glDisable(GL_LIGHTING);//关掉这个画线才有颜色，否则是黑色
		glMatrixMode(GL_PROJECTION);
		glAssert(glPushMatrix());
		glLoadIdentity();
		//glOrthox(-1.0f, 1.0f, -1.0f, 1.0f, 0.02f, 1000.0f);//这个函数会出错
		glOrtho(-1, 1, -1, 1, -1, 1);
		glMatrixMode(GL_MODELVIEW);
		glAssert(glPushMatrix());
		glLoadIdentity();
		
		//画出背景图片
		if (1)
		{
			glPushAttrib(GL_ALL_ATTRIB_BITS);
			glDisable(GL_DEPTH_TEST);

			//glShadeModel( GL_FLAT );
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			//先要activte ,再enable ，这个解决了出现绘制结果黑色的问题
			glActiveTextureARB(GL_TEXTURE0);  // 选择TEXTURE0为设置目标 
			glEnable(GL_TEXTURE_2D);  // 激活TEXTURE0单元
			glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);//不写上这个的话画出来的是黑色，不知为什么
			glAssert(glBindTexture(GL_TEXTURE_2D, backgroundtexture)); // 为TEXTURE0单元绑定texture纹理图像

			QImage GL_formatted_image;
			GL_formatted_image = QGLWidget::convertToGLFormat(background_image);
			if (GL_formatted_image.isNull())
			{
				std::cerr << "error GL_formatted_image" << std::endl;
				exit(1);
			}
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glAssert(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
				GL_formatted_image.width(), GL_formatted_image.height(),
				0, GL_RGBA, GL_UNSIGNED_BYTE, GL_formatted_image.bits()));




//			glColor3f(1.0f, 0.0f, 0.0f);
			/*下面开始绘制四边形*/
			glBegin(GL_QUADS);
			glTexCoord2f(0.0f, 1.0f);
			glVertex3f(-1.0f, 1.0f, 0.0f); // 左上顶点
			glTexCoord2f(0.0f, 0.0f);
			glVertex3f(-1.0f, -1.0f, 0.0f); // 左下顶点
			glTexCoord2f(1.0f, 0.0f);
			glVertex3f(1.0f, -1.0f, 0.0f); // 右下顶点
			glTexCoord2f(1.0f, 1.0f);
			glVertex3f(1.0f, 1.0f, 0.0f); // 右上顶点
			glEnd(); // 四边形绘制结束

			//glBegin(GL_TRIANGLES);
			//glTexCoord2f(1.0f, 0.0f);
			//glVertex2f(1.0f, 0.0f);
			//glTexCoord2f(0.0f, 1.0f);
			//glVertex2f(0.0f, 1.0f);
			//glTexCoord2f(0.0f, 0.0f);
			//glVertex2f(0.0f, 0.0f);
			//glEnd(); 
			glBindTexture(GL_TEXTURE_2D, 0);
			glDisable(GL_TEXTURE_2D);
			glEnable(GL_DEPTH_TEST);
			glPopAttrib();
		}




		glColor3f(0.2f, 0.2f, 0.8f);
		glLineWidth(1.0f);
		glBegin(GL_LINE_LOOP);
		//glVertex3f(-(viewport[2] - viewport[0]) / 2.0f, (viewport[3] - viewport[1]) / 2.0f, 0.0f);
		//glVertex3f((viewport[2] - viewport[0]) / 2.0f, (viewport[3] - viewport[1]) / 2.0f, 0.0f);
		//glVertex3f((viewport[2] - viewport[0]) / 2.0f, -(viewport[3] - viewport[1]) / 2.0f, 0.0f);
		//glVertex3f(-(viewport[2] - viewport[0]) / 2.0f, -(viewport[3] - viewport[1]) / 2.0f, 0.0f);
			glVertex3f(-1.0f, 1.0f,0.0f);
			glVertex3f(1.0f, 1.0f, 0.0f);
			glVertex3f(1.0f, -1.0f, 0.0f);
			glVertex3f(-1.0f, -1.0f, 0.0f);
			glVertex3f(-1.0f, 1.0f, 0.0f);
		glEnd();
		glMatrixMode(GL_PROJECTION);
		glAssert(glPopMatrix());
		glMatrixMode(GL_MODELVIEW);
		glAssert(glPopMatrix());


		glEnable(GL_LIGHTING);
		// 画出场景
		if (cur_rendermode & texture)
			scene.draw(false);
		//画出仿真的结果
		if(cur_rendermode & video_background_texture)
			drawSimulateObjects(scene.getCamera());

		// 画出操纵器
		if (curTool)
		{
			curTool->draw(scene.getCamera());
		}

		

		// 画出选框
		if (curToolType == TOOL_FACE_SELECT && isLMBDown)
		{
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();

			glDisable(GL_LIGHTING);
			glDisable(GL_DEPTH_TEST);
			glLineWidth(2.0f);
			glEnable(GL_LINE_STIPPLE);
			glLineStipple(1, 0x0F0F);

			float w = (float)this->width();
			float h = (float)this->height();
			float curPoint[2] = { curPos.x() / w * 2 - 1, 1 - curPos.y() / h * 2 };
			float pressPoint[2] = { pressPos.x() / w * 2 - 1, 1 - pressPos.y() / h * 2 };
			glColor3f(65.0f / 255.0f, 65.0f / 255.0f, 65.0f / 255.0f);
			glBegin(GL_LINE_LOOP);
			glVertex3f(curPoint[0], curPoint[1], 0.5);
			glVertex3f(pressPoint[0], curPoint[1], 0.5);
			glVertex3f(pressPoint[0], pressPoint[1], 0.5);
			glVertex3f(curPoint[0], pressPoint[1], 0.5);
			glEnd();

			glDisable(GL_LINE_STIPPLE);
			glEnable(GL_DEPTH_TEST);
		}
		drawCornerAxis();
		glFlush();
	}

	void GLViewWidget::resizeGL(int width, int height)
	{
		cout << "GLViewWidget " << width << " " << height << endl;
		if (!background_image.isNull())
		{
			viewport_ratio = (float)background_image.width() / (float)background_image.height();
		}
		else
		{
			viewport_ratio = 2.0f;
		}
		   
		float ratio = viewport_ratio;
		float old_ratio = width / height;
		float new_width, new_height;
		if (old_ratio < ratio)
		{
			new_width = width;
			new_height = new_width / ratio;
		}
		else {
			new_height = height;
			new_width = new_height*ratio;
		}
		glwidget_width = width;
		glwidget_height = height;
		if (!background_image.isNull() &&new_width > background_image.width() && new_height > background_image.height()) //尽可能保持背景图片的大小
		{
			camera_screen_width = background_image.width();
			camera_screen_height = background_image.height();
		}
		else
		{
			camera_screen_width = new_width;
			camera_screen_height = new_height;
		}

		scene.resizeCamera(new_width, new_height, glwidget_width, glwidget_height);
		updateGL();
	}

	void GLViewWidget::mousePressEvent(QMouseEvent *event)
	{
		int x = event->x();
		int y = event->y();
		// 优先处理拾取模式
		if (isPickMode && pickCallBackFun)
		{
			pickCallBackFun(scene.intersectObject(x, y));
			isPickMode = false;
			pickCallBackFun = NULL;
		}

		isLMBDown = event->buttons() & Qt::LeftButton;
		isMMBDown = event->buttons() & Qt::MidButton;
		isRMBDown = event->buttons() & Qt::RightButton;
		lastPos = event->pos();
		pressPos = event->pos();

//		QSharedPointer<Scene> scene = Global_WideoEditing_Window->scene;
		if (!isAltDown)
		{
			if (isLMBDown)
			{
				curPos = event->pos();

				QVector3D ori, dir;
				scene.getCamera().getRay(x, y, ori, dir);
				if (curToolType == GLViewWidget::TOOL_SELECT)    // 如果当前为选择工具，直接选择物体
				{
					curSelectObj = scene.selectObject(x, y);
				}
				else if (curToolType == GLViewWidget::TOOL_PAINT)
				{
//					scene->getPainter().beginPaint(QVector2D(x, y));
				}
				else if (curTool)
				{
					// 如果为移动/旋转/缩放工具，
					// 首先检查是否点击了操纵轴，
					// 如果是，开始操纵，如果不是，说明是想选择物体
					QWeakPointer<RenderableObject> oldObject = curSelectObj;
					QWeakPointer<RenderableObject> newObject = scene.intersectObject(x, y);
					bool isSameObj = oldObject == newObject;
					char axis = curTool->intersect(ori, dir);
					bool isHitAxis = axis != -1;
					if (!oldObject && !newObject.isNull())
					{ // 原来没有选中物体，现在选中一个新物体
						curSelectObj = newObject;
						curSelectObj.data()->select();
						curTool->captureObject(QSharedPointer<RenderableObject>(curSelectObj).toWeakRef());
					}
					else if (newObject.isNull())
					{    // 没有选中新物体
						if (isHitAxis)
						{    // 但是选中了操纵器的一个轴，于是操纵当前物体

							if (curSelectObj)
							{
								curSelectObj.data()->select();
								curTool->selectAxis(axis);
								curTool->beginManipulate(ori, dir, axis);
							}
						}
						else
						{  // 什么都没选中
							curTool->releaseObject();
							if (!curSelectObj.isNull())
							{
								curSelectObj.data()->deselect();
								curSelectObj.clear();
							}
						}
					}
					else if (isHitAxis && oldObject)
					{    // 选中原有物体的一个轴向
						 //m_curSelectObj = m_curTool->getCurObject();
						curTool->selectAxis(axis);
						curTool->beginManipulate(ori, dir, axis);
					}
					else if (!isSameObj && !isHitAxis)
					{    // 选中一个新物体
						if (!curSelectObj.isNull())
							curSelectObj.data()->deselect();
						curSelectObj = newObject;
						curSelectObj.data()->select();
						curTool->captureObject(curSelectObj);
					}
				}
			}
		}
//		Global_WideoEditing_Window->layerEditor->updateList();
//		Global_WideoEditing_Window->actionPaint->setEnabled(!curSelectObj.isNull());
//		Global_WideoEditing_Window->actionSelectFace->setEnabled(!curSelectObj.isNull());
		Global_WideoEditing_Window->transformEditor->updateWidgets();
//		updateGL();
		Global_WideoEditing_Window->updateGLView();
	}

	void GLViewWidget::mouseMoveEvent(QMouseEvent *event)
	{
		int x = event->x();
		int y = event->y();
		int dx = event->x() - lastPos.x();
		int dy = event->y() - lastPos.y();
//		QSharedPointer<Scene>scene = Global_WideoEditing_Window->scene;
		Camera& camera = scene.getCamera();

		curPos = event->pos();
		if (isAltDown)
		{
			if (event->buttons() & Qt::LeftButton)
			{
				scene.rotateCamera(dx, dy);
			}
			else if (event->buttons() & Qt::RightButton)
			{
				scene.zoomCamera(2 * (dx + dy));
			}
			else if (event->buttons() & Qt::MiddleButton)
			{
				scene.moveCamera(dx, dy);
			}
		}
		else
		{
			if (isLMBDown)
			{
				QVector3D ori, dir;
				scene.getCamera().getRay(x, y, ori, dir);
				if (curSelectObj && curTool)
				{
					if (curTool->isManipulating())
					{
						curTool->goOnManipulate(ori, dir);
					}
				}
				else if (curToolType == GLViewWidget::TOOL_PAINT)
				{
//					scene->getPainter().goOnPaint(QVector2D(x, y));
				}
			}
			else
			{
				
				QVector3D ori, dir;
				camera.getRay(x, y, ori, dir);
				if (curTool)
				{
					if (!curTool->isManipulating())
					{
						char axis = curTool->intersect(ori, dir);
						curTool->selectAxis(axis);
					}
				}
				else if (curToolType == GLViewWidget::TOOL_PAINT)
				{
//					scene->getPainter().onMouseHover(QVector2D(x, y));
				}
			}

		}
		lastPos = event->pos();
//		updateGL();
		Global_WideoEditing_Window->transformEditor->updateWidgets();
		Global_WideoEditing_Window->updateGLView();
	}

	void GLViewWidget::mouseReleaseEvent(QMouseEvent *event)
	{
		int x = event->x();
		int y = event->y();
//		QSharedPointer<Scene>scene = scene;
		Camera& camera = scene.getCamera();
		if (!isAltDown && isLMBDown)
		{
			if (curSelectObj && curTool)
			{	// 操纵器命令
				if (curTool->isManipulating())
				{
					curTool->endManipulate();
					ManipulateCommand::ManipulateCommandType type;
					switch (curToolType)
					{
					case TOOL_TRANSLATE:
						type = ManipulateCommand::MANCMD_TRANSLATE; break;
					case TOOL_ROTATE:
						type = ManipulateCommand::MANCMD_ROTATE; break;
					case TOOL_SCALE:
						type = ManipulateCommand::MANCMD_SCALE; break;
					}
					QUndoCommand* cmd = new ManipulateCommand(
						curSelectObj.data()->getObjectID(),
						curTool->getOldTransform(),
						curTool->getNewTransform(),
						type);
//					scene->getUndoStack().push(cmd);
					goto END_RELEASE;
				}
			}
			else if (curToolType == GLViewWidget::TOOL_PAINT)
			{	// 画画命令
//				scene->getPainter().endPaint(QVector2D(x, y));
			}
			else if (curToolType == GLViewWidget::TOOL_FACE_SELECT)
			{	// 面选择命令
				GeometryExposer& exposer = scene.getGeometryExposer();
				float w = this->width();
				float h = this->height();
				QVector2D minRatio(min(x, pressPos.x()) / w, min(y, pressPos.y()) / h);
				QVector2D maxRatio(max(x, pressPos.x()) / w, max(y, pressPos.y()) / h);
				QSet<int>faceIDSet;
				Mesh* objPtr = (Mesh*)curSelectObj.data();
				exposer.getRegionFaceID(minRatio, maxRatio, objPtr->getObjectID(), faceIDSet);
				if (isCtrlDown)
					objPtr->addSelectedFaceID(faceIDSet);
				else if (isShiftDown)
					objPtr->removeSelectedFaceID(faceIDSet);
				else
					objPtr->setSelectedFaceID(faceIDSet);
			}
		}
	END_RELEASE:
		isLMBDown = false;
		Global_WideoEditing_Window->transformEditor->updateWidgets();
//		updateGL();
		Global_WideoEditing_Window->updateGLView();
		return;
	}
	void GLViewWidget::keyPressEvent(QKeyEvent *event)
	{
		switch (event->key())
		{
		case Qt::Key_Alt:
			isAltDown = true;
			break;
		case Qt::Key_Control:
			isCtrlDown = true;
			break;
		case Qt::Key_Shift:
			isShiftDown = true;
			break;
		case Qt::Key_Delete:
//			QSharedPointer<RenderableObject> renObj = Global_WideoEditing_Window->viewWidget->getSelectedObject();;
//			if (renObj)
			{
//				QUndoCommand* cmd = new RemoveObjectCommand(renObj);
//				Global_WideoEditing_Window->scene->getUndoStack().push(cmd);
//				Global_WideoEditing_Window->actionSelect->trigger();
				Global_WideoEditing_Window->transformEditor->updateWidgets();
			}
		}
	}

	void GLViewWidget::keyReleaseEvent(QKeyEvent *event)
	{
		switch (event->key())
		{
		case Qt::Key_Alt:
			isAltDown = false; break;
		case Qt::Key_Control:
			isCtrlDown = false; break;
		case Qt::Key_Shift:
			isShiftDown = false; break;
		}
	}


	void GLViewWidget::wheelEvent(QWheelEvent *event)
	{
		scene.zoomCamera(event->delta());
		if (curToolType == TOOL_PAINT)
		{
//			Global_WideoEditing_Window->scene->getPainter().onMouseHover(QVector2D(lastPos.x(), lastPos.y()));
		}
		updateGL();
	}

	void GLViewWidget::drawCornerAxis()
	{
		
		int viewport[4];
		int scissor[4];

		glDisable(GL_LIGHTING);
		glDisable(GL_DEPTH_TEST);
		glEnable(GL_SCISSOR_TEST);
		// The viewport and the scissor are changed to fit the lower left
		// corner. Original values are saved.
		glGetIntegerv(GL_VIEWPORT, viewport);
		glGetIntegerv(GL_SCISSOR_BOX, scissor);
		float coord_system_region_size_ = viewport[2] < viewport[3]? viewport[2]/4.0f: viewport[3]/4.0f;
		// Axis viewport size, in pixels
		glViewport(viewport[0], viewport[1], coord_system_region_size_, coord_system_region_size_);
		glScissor(viewport[0], viewport[1], coord_system_region_size_, coord_system_region_size_);

		// The Z-buffer is cleared to make the axis appear over the
		// original image.
//		(GL_DEPTH_BUFFER_BIT);

		// Tune for best line rendering
		glLineWidth(5.0);

		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glOrtho(-1, 1, -1, 1, -1, 1);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		QVector3D trans; 
		QQuaternion rot;
		scene.getCamera().getCameraTransform(trans, rot);
		QMatrix4x4 rotation;
		rotation.rotate(rot);
		glMultMatrixf(rotation.transposed().constData()); //sccamera()->orientation().inverse().matrix());

		float axis_size = 0.8f; // other 0.2 space for drawing the x, y, z labels
		drawAxis(axis_size);

		// Draw text id
		glDisable(GL_LIGHTING);
		glColor3f(0, 0, 0);
		//renderText(axis_size, 0, 0, "X");
		//renderText(0, axis_size, 0, "Y");
		//renderText(0, 0, axis_size, "Z");

		glMatrixMode(GL_PROJECTION);
		glPopMatrix();

		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();

		// The viewport and the scissor are restored.
		glScissor(scissor[0], scissor[1], scissor[2], scissor[3]);
		glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);

		glEnable(GL_LIGHTING);
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_SCISSOR_TEST);
	}

	void GLViewWidget::setTool(ToolType type)
	{
		if (curToolType == TOOL_PAINT)
		{
//			Global_WideoEditing_Window->paintEditor->attachToCamera(false);
		}
		curToolType = type;
//		QSharedPointer<Scene> scene = Global_WideoEditing_Window->scene;
		bool isPaintMode = type == TOOL_PAINT;
		Mesh::enableWireFrame(!isPaintMode);
//		scene->getBrush().activate(isPaintMode);
//		Global_WideoEditing_Window->actionImport->setEnabled(!isPaintMode);
//		Global_WideoEditing_Window->actionPlaneLocator->setEnabled(!isPaintMode);
		switch (type)
		{
		case TOOL_TRANSLATE:
			curTool = &translateTool;
			break;
		case TOOL_ROTATE:
			curTool = &rotateTool;
			break;
		case TOOL_SCALE:
			curTool = &scaleTool;
			break;
		case TOOL_PAINT:
		case TOOL_SELECT:
		case TOOL_FACE_SELECT:
			curTool = NULL;
		}
		if (curTool)
		{
			QSharedPointer<RenderableObject>cso(curSelectObj);
			curTool->captureObject(cso.toWeakRef());
		}
		if (curToolType == GLViewWidget::TOOL_PAINT && curSelectObj)
		{
			if (curSelectObj.data()->getType() & RenderableObject::OBJ_MESH)
			{
				QWeakPointer<Mesh> wm = qWeakPointerCast<Mesh>(QSharedPointer<RenderableObject>(curSelectObj));
//				scene->getPainter().setObject(wm);
//				scene->getBrush().setObject(wm);
			}
		}
		bool hasObject = !curSelectObj.isNull();
		isPaintMode = curToolType == GLViewWidget::TOOL_PAINT;
//		Global_WideoEditing_Window->paintEditor->setEnabled(hasObject && isPaintMode);
//		Global_WideoEditing_Window->paintEditor->changeColorPicker(true);
		if (!isPaintMode)
			scene.setPickerObjectUnfrozen();
//		updateGL();
		Global_WideoEditing_Window->updateGLView();
	
	}

	bool GLViewWidget::focusCurSelected()
	{
		if (curSelectObj)
		{
			RenderableObject *obj = curSelectObj.data();
			scene.getCamera().setCenter(obj->getCenter(), obj->getApproSize());
			scene.updateGeometryImage();

			return true;
		}
		return false;
	}

	QSharedPointer<RenderableObject> GLViewWidget::getSelectedObject()
	{
		return curSelectObj;
	}

	void GLViewWidget::enterPickerMode(void(*callBackFun)(QSharedPointer<RenderableObject>))
	{
		pickCallBackFun = callBackFun;
		isPickMode = true;
		setCursor(Qt::PointingHandCursor);
	}

}

