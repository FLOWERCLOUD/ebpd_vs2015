#include "GLViewWidget.h"
#include "QSplitter"

extern std::string g_icons_theme_dir;

namespace videoEditting
{
	QSharedPointer<OGL_widget_skin_hidden> OGL_viewports_skin2::hidden;

	
	GLWidget_render_mode::GLWidget_render_mode(QWidget* parent) : QWidget(parent)
	{
		setupUi(this);
		
		QIcon icon((g_icons_theme_dir + "/wireframe_transparent.svg").c_str());
		toolB_wire_transc->setIcon(icon);

		QIcon icon1((g_icons_theme_dir + "/wireframe.svg").c_str());
		toolB_wire->setIcon(icon1);

		QIcon icon2((g_icons_theme_dir + "/solid.svg").c_str());
		toolB_solid->setIcon(icon2);

		QIcon icon3((g_icons_theme_dir + "/texture.svg").c_str());
		toolB_tex->setIcon(icon3);

		QIcon icon4(QString((g_icons_theme_dir + "/wireframe_transparent.svg").c_str()));
		toolB_video_background_tex->setIcon(icon4);

		QIcon icon5((g_icons_theme_dir + "/wireframe.svg").c_str());
		toolB_image_resolution->setIcon(icon5);

		QIcon icon6((g_icons_theme_dir + "/solid.svg").c_str());
		toolB_tex_glwidget_resolution->setIcon(icon6);

		QIcon icon7((g_icons_theme_dir + "/texture.svg").c_str());
		toolB_showgrid->setIcon(icon7);
		QIcon icon8((g_icons_theme_dir + "/texture.svg").c_str());
		toolB_showbackground->setIcon(icon7);




	}



	OGL_viewports_skin2::OGL_viewports_skin2(QWidget* w, VideoEditingWindow* m):
		QFrame(w),_main_layout(0), _main_window(m)
	{

		if (!hidden)
		{
			hidden = QSharedPointer<OGL_widget_skin_hidden>(new OGL_widget_skin_hidden(this));
			hidden->hide();
			hidden->updateGL();
			hidden->makeCurrent();
		}
		QVBoxLayout* vlayout = new QVBoxLayout(this);
		vlayout->setSpacing(0);
		vlayout->setContentsMargins(0, 0, 0, 0);

		QSplitter* splitter = new QSplitter(this);
		splitter->setOrientation(Qt::Horizontal);
		splitter->setContentsMargins(0, 0, 0, 0);
		splitter->setHandleWidth(3);
		splitter->setStretchFactor(0, 1);
		splitter->setStretchFactor(1, 1);
		vlayout->addWidget(splitter);


		Viewport_frame_skin2* frame1 = new Viewport_frame_skin2(splitter, 0,hidden.data());
		QObject::connect(frame1, SIGNAL(active(int)),
			this, SLOT(active_viewport_slot(int)));
		camera_viewer = frame1->getGLViewWidget();
		
		//QVBoxLayout* layout = new QVBoxLayout(frame);
		//layout->setSpacing(0);
		//layout->setContentsMargins(0, 0, 0, 0);
		//frame->setLayout(layout);

		splitter->addWidget(frame1);

		Viewport_frame_skin2* frame2 = new Viewport_frame_skin2(splitter, 0, hidden.data());
		QObject::connect(frame2, SIGNAL(active(int)),
			this, SLOT(active_viewport_slot(int)));
		world_viewer = frame2->getGLViewWidget();
		//QVBoxLayout* layout = new QVBoxLayout(frame);
		//layout->setSpacing(0);
		//layout->setContentsMargins(0, 0, 0, 0);
		//frame->setLayout(layout);

		splitter->addWidget(frame2);

	}
	OGL_viewports_skin2::~OGL_viewports_skin2()
	{

	}


	void OGL_viewports_skin2::updateGL()
	{

	}


	void OGL_viewports_skin2::enterEvent(QEvent* e)
	{

	}


	void OGL_viewports_skin2::active_viewport_slot(int id)
	{

	}


	void Viewport_frame_skin2::setup()
	{
		QVBoxLayout* vlayout = new QVBoxLayout(this);
		vlayout->setSpacing(0);
		vlayout->setContentsMargins(0, 0, 0, 0);


		renderbar = new GLWidget_render_mode(this);

		glwidget = new GLViewWidget(this, share_);
		glwidget->setMinimumSize(QSize(100, 100));

		vlayout->addWidget(renderbar);
		vlayout->addWidget(glwidget);
		vlayout->setStretch(0, 0);//设置为0，让其保持自身大小
		vlayout->setStretch(1, 1);

		this->setLayout(vlayout);
		//setup signal slot
		connect(renderbar->toolB_wire_transc, SIGNAL(pressed()), glwidget, SLOT(rd_mode_toolb_wire_transc()));
		connect(renderbar->toolB_wire, SIGNAL(pressed()), glwidget, SLOT(rd_mode_toolb_wire()));
		connect(renderbar->toolB_solid, SIGNAL(pressed()), glwidget, SLOT(rd_mode_toolb_solid()));
		connect(renderbar->toolB_tex, SIGNAL(pressed()), glwidget, SLOT(rd_mode_toolb_tex()));
		connect(renderbar->toolB_video_background_tex, SIGNAL(pressed()), glwidget, SLOT(rd_mode_toolb_video_background_tex()));
		connect(renderbar->toolB_image_resolution, SIGNAL(pressed()), glwidget, SLOT(rd_mode_image_resolution()));
		connect(renderbar->toolB_tex_glwidget_resolution, SIGNAL(pressed()), glwidget, SLOT(rd_mode_tex_glwidget_resolution()));
		connect(renderbar->toolB_showgrid, SIGNAL(pressed()), glwidget, SLOT(draw_grid_toggle()));
		connect(renderbar->toolB_showbackground, SIGNAL(pressed()), glwidget, SLOT(draw_background_toggle()));

			
		//		QObject::connect(render->toolB_wire_transc, SIGNAL(pressed()), this, SLOT(rd_mode_toolb_wire_transc()));
	}
}
