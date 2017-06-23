#pragma once

#include "TranslateManipulator.h"
#include "RotateManipulator.h"
#include "ScaleManipulator.h"
#include "RenderableObject.h"
#include "VEScene.h"
#include "ui_glwidget_render_mode.h"
#include <QtOpenGL/QGLWidget>
#include <QImage>

class VideoEditingWindow;

namespace videoEditting
{
	const unsigned int R_MASK = 0x1F;
	const unsigned int L_MASK = ~0x1F;
	const unsigned int	wireframe_transparent = 0x1;
	const unsigned int	wireframe = 0x2;
	const unsigned int	solid = 0x4;
	const unsigned int	texture = 0x8;
	const unsigned int	video_background_texture = 0x10;
	const unsigned int	image_resolution = 0x20;
	const unsigned int	glwidget_resolution = 0x40;

	class Scene;
	
	class OGL_widget_skin_hidden : public QGLWidget {
	public:
		OGL_widget_skin_hidden(QWidget* w) : QGLWidget(w) {}

		void initializeGL();
	};

	class GLWidget_render_mode : public QWidget, public Ui::glwidgetRender_mode_toolbuttons {
		Q_OBJECT
	public:
		GLWidget_render_mode(QWidget* parent);

	};



	class GLViewWidget :public QGLWidget
	{
		Q_OBJECT
	public:
		GLViewWidget(QWidget* parent = 0, QGLWidget* sh = 0);
		~GLViewWidget(void);

		enum ToolType { TOOL_SELECT = 0, TOOL_FACE_SELECT, TOOL_TRANSLATE, TOOL_ROTATE, TOOL_SCALE, TOOL_PAINT };
		void setTool(ToolType type);
		Manipulator2*& getCurTool() { return curTool; }
		ToolType getCurToolType() { return curToolType; }
		bool focusCurSelected();
		QSharedPointer<RenderableObject> getSelectedObject();
		void clearSelectedObject() { curSelectObj = QWeakPointer<RenderableObject>(); }

		void enterPickerMode(void(*callBackFun)(QSharedPointer<RenderableObject>));
		videoEditting::Scene& getScene()
		{
			return scene;
		}
		void setBackGroundImage(QImage& image)
		{
			isbackgroundChanged = true;
			background_image = image;
			resizeGL(glwidget_width, glwidget_height);//trick
		}
		bool isGLwidgetInitialized()
		{
			return isGLwidgetInitialize;
		}
		void directDraw()
		{
			paintGL();
		}
	public slots:
		void changeRenderMode(unsigned int rendermode);
		void rd_mode_toolb_wire_transc();
		void rd_mode_toolb_wire();
		void rd_mode_toolb_solid();
		void rd_mode_toolb_tex();
		void rd_mode_toolb_video_background_tex();
		void rd_mode_image_resolution();
		void rd_mode_tex_glwidget_resolution();
		void draw_grid_toggle()
		{
			static bool l_isDrawGrid = false;
			l_isDrawGrid = !l_isDrawGrid;
			isDrawGrid = l_isDrawGrid;
		}
		void setDraw_grid(bool v)
		{
			isDrawGrid = v;
		}
		void draw_background_toggle()
		{
			static bool l_isDrawback = false;
			l_isDrawback = !l_isDrawback;
			isDrawBackGoundImage = l_isDrawback;
		}
		void setDraw_background(bool v)
		{
			isDrawBackGoundImage = v;
		}
	protected:
		void initializeGL();
		void paintGL();
		void resizeGL(int width, int height);
		void mousePressEvent(QMouseEvent *event);
		void mouseMoveEvent(QMouseEvent *event);
		void mouseReleaseEvent(QMouseEvent *event);
		void wheelEvent(QWheelEvent *event);
		void keyPressEvent(QKeyEvent *event);
		void keyReleaseEvent(QKeyEvent *event);
		void drawCornerAxis();
		void drawSimulateObjectsWithBackgroundTexture(videoEditting::Camera& camera);
	private:
		QWeakPointer<RenderableObject> curSelectObj;
		QColor backgroundClr;
		TranslateManipulator translateTool;
		RotateManipulator    rotateTool;
		ScaleManipulator     scaleTool;
		ToolType curToolType;
		Manipulator2* curTool;

		bool isPickMode;		// 拾取模式，接受在场景中拾取物体的请求
		void(*pickCallBackFun)(QSharedPointer<RenderableObject>);

		bool isAltDown;
		bool isCtrlDown;
		bool isShiftDown;
		bool isLMBDown;
		bool isMMBDown;
		bool isRMBDown;

		QPoint pressPos;
		QPoint lastPos;
		QPoint curPos;
		videoEditting::Scene scene;
		int glwidget_width;
		int glwidget_height;
		int camera_screen_width;
		int camera_screen_height;
		QImage     background_image;
		bool        isbackgroundChanged;
		float viewport_ratio;//accoring to the backgroundimage
		GLuint backgroundtexture;
		GLuint greenBacktexture;
		unsigned int cur_rendermode;
		bool isDrawGrid;
		bool isDrawBackGoundImage;
		bool isGLwidgetInitialize;
	};

	class Viewport_frame_skin2;
	/** @class OGL_viewports
	@brief Multiple viewports handling with different layouts

	*/
	class OGL_viewports_skin2 : public QFrame {
		Q_OBJECT
	public:
		OGL_viewports_skin2(QWidget* w, VideoEditingWindow* m);
		~OGL_viewports_skin2();
		/// Updates all viewports
		void updateGL();
		void enterEvent(QEvent* e);
		GLViewWidget* getCamera_viewer()
		{
			return camera_viewer;
		}
		GLViewWidget* getWorld_viewer()
		{
			return world_viewer;
		}
		private slots:
		void active_viewport_slot(int id);
	signals:
		void active_viewport_changed(int id);
	private:
		GLViewWidget* camera_viewer;
		GLViewWidget* world_viewer;
		VideoEditingWindow* _main_window;
		/// Layout containing all viewports
		QLayout* _main_layout;
		static	QSharedPointer<OGL_widget_skin_hidden> hidden;

	};
	class Viewport_frame_skin2 : public QFrame {
		Q_OBJECT
	public:

		Viewport_frame_skin2(QWidget* w, int id, OGL_widget_skin_hidden* share) :
			QFrame(w), _id(id), share_(share)
		{
			setFrameShape(QFrame::Box);
			setFrameShadow(QFrame::Plain);
			setLineWidth(1);
			setStyleSheet(QString::fromUtf8("color: rgb(0, 0, 0);"));
			setup();
		}
		void setup();

		int id() const { return _id; }
		GLViewWidget* getGLViewWidget()
		{
			return glwidget;
		}
		GLWidget_render_mode* getWidget_render()
		{
			return renderbar;
		}
	signals:
		void active(int);

		private slots:
		void activate() {
			emit active(_id);
		}

	private:
		int _id;
		GLViewWidget* glwidget;
		GLWidget_render_mode* renderbar;
		OGL_widget_skin_hidden* share_;
	};
}
