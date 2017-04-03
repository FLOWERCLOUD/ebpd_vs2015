#include "CustomGL/glew.h"
#include "paint_canvas.h"
#include "sample_set.h"
#include "vertex.h"
#include "triangle.h"
#include "main_window.h"
#include "basic_types.h"
#include "tracer.h"
#include "globals.h"
#include "rendering/render_types.h"
#include "savePLYDialog.h"
#include "file_io.h"
#include "standardCamera.h"
#include <fstream>
#include <QGLViewer/manipulatedCameraFrame.h>
#include "bullet/BasicDemo.h"
#include "bullet/Rigidbodydemo.h"
#include "bullet/GLInstancingRenderer.h"
#include "snapshotsetting.h"
#include "saveplysetting.h"
#include "GlobalObject.h"
#include "toolbars/gizmo/gizmo.hpp"
#include "toolbars/gizmo/gizmo_rot.hpp"
#include "toolbars/gizmo/gizmo_trans.hpp"
#include "toolbars/gizmo/gizmo_trackball.hpp"
#include "toolbox/gl_utils/glsave.hpp"
#include "global_datas/g_vbo_primitives.hpp"
#include "rendering/opengl_stuff.hpp"
#include "manipulator.h"
#include "toolbox/gl_utils/glsave.hpp"
#include "rendering/depth_peeling.hpp"
#include "rendering/rendering.hpp"
#include "../control/cuda_ctrl.hpp"
#include "../global_datas/cuda_globals.hpp"
#include "../animation/bone.hpp"
#include "../animation/skeleton.hpp"
#include "toolbars/io_selection/IO_interface_skin.hpp"
#include "toolbars/io_selection/IO_skeleton.hpp"
#include "toolbars/io_selection/IO_mesh_edit.hpp"
#include "toolbars/io_selection/IO_graph.hpp"
#include "toolbars/io_selection/IO_disable_skin.hpp"
class ManipulateTool;

using namespace pcm;
using namespace qglviewer;
using namespace RenderMode;
#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE 0x809D
#endif

extern int REAL_TIME_RENDER;
extern bool isBulletRun ;
extern bool isAnimationRun;
RenderMode::WhichColorMode	which_color_mode_;
RenderMode::RenderType which_render_mode;
bool isShowNoraml = false;
static QPoint currenMousePos; 
//Manipulator g_manipulator( Manipulator::OBJECT ,struct SelectedObj() );

using namespace Depth_peeling;


static void draw_junction_sphere(bool rest_pose)
	//const std::vector<int>& selected_joints,
	//bool rest_pose)
{


	glColor4f(0.4f, 0.3f, 1.f, 0.5f);
//	const std::vector<int>& set = _skeleton.get_selection_set();
	for(unsigned i = 0; i <1 /*set.size()*/; i++)
	{
		float r  = 0.1;//g_animesh->get_junction_radius(set[i]);
		//Vec3 v = rest_pose ? g_skel->joint_rest_pos(set[i]) : g_skel->joint_pos(set[i]);
		Vec v(0.5,0.5,0.5);
		glPushMatrix();
		glTranslatef(v.x, v.y, v.z);
		glScalef(r, r, r);
		g_primitive_printer.draw(g_sphere_vbo);
		glPopMatrix();
	}
}


static void drawSkeleton(  )//const qglviewer::Camera& cam,
					//const std::vector<int>& selected_joints,
					//bool rest_pose)
{
	glColor3f(0.f, 0.f, 0.7f);
	glLineStipple(5, 0xAAAA);
	Tbx::GLEnabledSave save_line(GL_LINE_STIPPLE, true, true);
	for(int i = 0; i < 1; i++)
	{

		const Vec p0 = Vec(0,0,0);
		const Vec p1 = Vec(1,1,1);
		glBegin(GL_LINES);
		glVertex3f(p0.x, p0.y, p0.z);
		glVertex3f(p1.x, p1.y, p1.z);
		glEnd();
	}
}

static void info_gl()
{
	const GLubyte *str;
	printf("OpenGL infos with gl functions\n");
	str = glGetString(GL_RENDERER);
	printf("Renderer : %s\n", str);
	str = glGetString(GL_VENDOR);
	printf("Vendor : %s\n", str);
	str = glGetString(GL_VERSION);
	printf("OpenGL Version : %s\n", str);
	str = glGetString(GL_SHADING_LANGUAGE_VERSION);
	printf("GLSL Version : %s\n", str);
	GL_CHECK_ERRORS();
}



PaintCanvas::PaintCanvas(const QGLFormat& format, QWidget *parent):
	QGLViewer(format, parent),
	coord_system_region_size_(150),	
	single_operate_tool_(nullptr),
	show_trajectory_(false)
{
    _heuristic   = new Tbx::Selection_nearest<int>();
	setAxisIsDrawn(false);
	setGridIsDrawn(false);
	pApp = NULL;

	which_color_mode_ =VERTEX_COLOR/*SphereMode*/,
	which_render_mode =RenderMode::PointMode,

	fov = 60;
	clipRatioFar = 1;
	clipRatioNear = 1;
	nearPlane = .2f;
	farPlane = 5.f;
	takeSnapTile=false;
	centerframeNum = 0;
	is_key_l_pressed = false;

	main_window_ = (main_window*)parent;
	



	camera()->setPosition(qglviewer::Vec(1.0,  1.0, 1.0));	

	camera()->lookAt(sceneCenter());
	camera()->setType(qglviewer::Camera::PERSPECTIVE);
	camera()->showEntireScene();
}

PaintCanvas::PaintCanvas(const QGLFormat& format, int type, QWidget *parent,QWidget * mwindow,const QGLWidget* shareWidget)
	  : QGLViewer(parent, shareWidget),
	  coord_system_region_size_(150),	
	  single_operate_tool_(nullptr),
	  show_trajectory_(false),
	  m_instancingRenderer(NULL),
	  particle_(NULL),
	  _draw_gizmo(false),
	  m_render_ctx(NULL),
	  _io(NULL),
	  m_io(NULL)
{
	static int init_count = 0;
	init_count++;

	makeCurrent();// this function shouled be invoke ,otherwise glewInit will fail
	_heuristic   = new Tbx::Selection_nearest<int>();
	pApp = NULL;
	
    b3Assert(glGetError() ==GL_NO_ERROR);

	if (glewInit() != GLEW_OK)
		exit(1); // or handle the error in a nicer way
	if (!GLEW_VERSION_2_1)  // check that the machine supports the 2.1 API.
		exit(1); // or handle the error in a nicer way
	if (!GLEW_VERSION_3_1)  // check that the machine supports the 2.1 API.
		exit(1);
	if (!GLEW_VERSION_4_1)  // check that the machine supports the 2.1 API.
		exit(1);
	if( isBulletRun)
	{
		if (!m_instancingRenderer)
		{
			m_instancingRenderer = new GLInstancingRenderer(128*100,128*30*30);	
			m_instancingRenderer->init();
		}		
//		m_instancingRenderer->enableBlend(true);
	}

	info_gl();
	init_opengl();
	Cuda_ctrl::init_opengl_cuda();

	setCamera( new StandardCamera());
	setAxisIsDrawn(false);
	setGridIsDrawn(false);


	if (type < 3)
	{
		// Make camera the default manipulated frame.
//		setManipulatedFrame( camera()->frame() );
		setMouseBinding(Qt::AltModifier, Qt::LeftButton, QGLViewer::CAMERA, QGLViewer::ROTATE);
		setMouseBinding(Qt::AltModifier, Qt::RightButton, QGLViewer::CAMERA, QGLViewer::TRANSLATE);
		setMouseBinding(Qt::AltModifier, Qt::MidButton, QGLViewer::CAMERA, QGLViewer::ZOOM);
		setWheelBinding(Qt::AltModifier, QGLViewer::CAMERA, QGLViewer::ZOOM);
		setWheelBinding(Qt::NoModifier, QGLViewer::CAMERA, QGLViewer::ZOOM);
		setMouseBinding(Qt::NoModifier, Qt::LeftButton, QGLViewer::CAMERA, QGLViewer::NO_MOUSE_ACTION);
		setMouseBinding(Qt::NoModifier, Qt::RightButton, QGLViewer::CAMERA, QGLViewer::NO_MOUSE_ACTION);
		setMouseBinding(Qt::NoModifier, Qt::MidButton, QGLViewer::CAMERA, QGLViewer::NO_MOUSE_ACTION);

		 // Move camera according to viewer type (on X, Y or Z axis)
		 camera()->setPosition(qglviewer::Vec((type==0)? 1.0 : 0.0, (type==1)? 1.0 : 0.0, (type==2)? 1.0 : 0.0));
		 camera()->lookAt(sceneCenter());
		 camera()->setType(qglviewer::Camera::PERSPECTIVE);
		 camera()->showEntireScene();
		 // Forbid rotation
//		 WorldConstraint* constraint = new WorldConstraint();
//		 constraint->setRotationConstraintType(AxisPlaneConstraint::FORBIDDEN);
//		 camera()->frame()->setConstraint(constraint);
	}

	which_color_mode_ =VERTEX_COLOR/*SphereMode*/,
		which_render_mode =RenderMode::PointMode,

		fov = 60;
	clipRatioFar = 1;
	clipRatioNear = 1;
	nearPlane = .2f;
	farPlane = 5.f;
	takeSnapTile=false;
	centerframeNum = 0;
	is_key_l_pressed = false;
	main_window_ = (main_window*)mwindow;

	ss = new SnapshotSetting();
	splys = new SavePlySetting();
	
	isAnimationRun = isBulletRun;
	if(isBulletRun)
	{
		//pApp = new BasicDemo(this);
		//pApp = new RigidbodyDemo(this,"bullet/teddy.obj");
		pApp = new RigidbodyDemo(this,"resource/meshes/keg3/keg_skinning.obj");
		//pApp = new RigidbodyDemo(this,"resource/bulletoutput/bunny.obj");
		//pApp = new RigidbodyDemo(this,"resource/bulletoutput/cube.obj");
	}
	//m_peeler = new Peeler();
	//m_peeler->set_render_func( new RenderFuncWireframe( camera()) );
	//int scw = camera()->screenWidth();
	//int sch = camera()->screenHeight();
	////1228 720
	////g_pbo_depth =  new Tbx::GlBuffer_obj<GLuint>( camera()->screenWidth()*camera()->screenHeight(), GL_PIXEL_UNPACK_BUFFER);
	////g_pbo_color =  new Tbx::GlBuffer_obj<GLint>( camera()->screenWidth()*camera()->screenHeight(), GL_PIXEL_UNPACK_BUFFER);
	//m_pbo_depth =  new Tbx::GlBuffer_obj<GLuint>( 1228* 720, GL_PIXEL_UNPACK_BUFFER);
	//m_pbo_color =  new Tbx::GlBuffer_obj<GLint>( 1228* 720, GL_PIXEL_UNPACK_BUFFER);
	////g_peeler->reinit_depth_peeling(camera()->screenWidth(),camera()->screenHeight());
	//m_peeler->reinit_depth_peeling(1228 ,720);
	_pivot_user = Tbx::Vec3(0.f, 0.f, 0.f);
	_pivot = Tbx::Vec3(0.f, 0.f, 0.f);
	_pivot_mode = EOGL_widget::JOINT;

	_draw_gizmo  = true;
	m_io          = new IO_disable_skin(this);
	_heuristic   = new Tbx::Selection_nearest<int>();
	_gizmo       = new Gizmo_trans();


}
PaintCanvas::~PaintCanvas()
{

	if(pApp)delete pApp;
	if(particle_)delete particle_;
	if(m_instancingRenderer)delete m_instancingRenderer;
	if(ss) delete ss;
	if(splys) delete splys;
	if(_heuristic)delete _heuristic;
	g_primitive_printer.clear(); //clear it otherwise when g_primitive_printer destruct will get error
	if(m_render_ctx)delete m_render_ctx;

}



void PaintCanvas::draw()
{
	qglviewer::Camera* c_camera = camera();
	//makeCurrent();
	//GL_CHECK_ERRORS();

//	setVisualHintsMask( 1);
	glPushAttrib(GL_ALL_ATTRIB_BITS);
 	display_loop( camera() ,m_render_ctx);
	glPopAttrib();
	if(_heuristic->_type == Tbx::Selection::CIRCLE   && _is_mouse_in )
	{
		Tbx::Selection_circle<int>* h = (Tbx::Selection_circle<int>*)_heuristic;
		glColor3f(0.f, 0.f, 0.f);
		draw_circle( camera()->screenWidth(), camera()->screenHeight(), currenMousePos.x(), currenMousePos.y(), h->_rad);
	}
	
	if(_draw_gizmo){
		m_io->update_frame_gizmo();
		_gizmo->draw( Tbx::Camera( camera()));
	}

	if(_draw_gizmo)
	{
//		g_manipulator.draw();
		//_io->update_frame_gizmo();
		//_gizmo->draw(_cam);
	}
if(0)
{	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glAssert( glColor4f(1.f, 0.f, 0.f, 1.f) );
	draw_junction_sphere(true);
	drawSkeleton();
//	g_primitive_printer.draw(g_cylinder_vbo);
//	g_primitive_printer.draw(g_circle_lr_vbo);
	//g_primitive_printer.draw(g_arc_circle_vbo);
	//g_primitive_printer.draw(g_cube_vbo);
	//g_primitive_printer.draw(g_grid_vbo);
	//g_primitive_printer.draw(g_sphere_vbo);
//	g_primitive_printer.draw(g_circle_vbo);
	glPopAttrib();
}
	if(isAnimationRun)
	{
		glBegin(GL_POINTS);
		for (int i=0; i<nbPart_; i++)
			particle_[i].draw();
		glEnd();
	}
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	if(pApp)
		pApp->Idle();
	glPopAttrib();


	drawSampleSet(this);
	//if(takeSnapTile) pasteTile();
	//draw line tracer
	if (show_trajectory_)
	{
		Tracer::get_instance().setcenterframeNum(centerframeNum);
		Tracer::get_instance().draw();
	}
	
	//std::cout<<"updategl end"<<std::endl;





}
void PaintCanvas::fastDraw()
{
	draw();
}
void PaintCanvas::init()
{
	if(!m_render_ctx)m_render_ctx = new Render_context( width() ,height());

	if(isAnimationRun)
	{
		nbPart_ = 2000;
		particle_ = new Particle[nbPart_];
		glPointSize(3.0);
		setGridIsDrawn();
//		help();
		startAnimation();
	}



	setStateFileName("");
	glEnable(GL_DEPTH_TEST);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	QColor background_color = Qt::white;
	setBackgroundColor(background_color);

	//setGridIsDrawn();//参考平面2014-12-16
	//camera()->frame()->setSpinningSensitivity(100.f);
//	setMouseTracking(true);
	// light0
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

	/* to use material color, the GL_COLOR_MATERIAL should be disabled */
	//glDisable(GL_COLOR_MATERIAL);

	//added by huayun
	setGraph_WrapBoxShowOrNot(false);  
	setEdgeVertexsShowOrNot(false);
	if(pApp)
		pApp->InitializeNaive();



}


void PaintCanvas::animate()
{


	for (int i=0; i<nbPart_; i++)
		particle_[i].animate();
	
}

QString PaintCanvas::helpString() const
{
	QString text("<h2>A n i m a t i o n</h2>");
	text += "Use the <i>animate()</i> function to implement the animation part of your ";
	text += "application. Once the animation is started, <i>animate()</i> and <i>draw()</i> ";
	text += "are called in an infinite loop, at a frequency that can be fixed.<br><br>";
	text += "Press <b>Return</b> to start/stop the animation.";
	return text;
}




void PaintCanvas::drawCornerAxis()  
{
	int viewport[4];
	int scissor[4];

	glDisable(GL_LIGHTING);

	// The viewport and the scissor are changed to fit the lower left
	// corner. Original values are saved.
	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetIntegerv(GL_SCISSOR_BOX, scissor);

	// Axis viewport size, in pixels
	glViewport(0, 0, coord_system_region_size_, coord_system_region_size_);
	glScissor(0, 0, coord_system_region_size_, coord_system_region_size_);

	// The Z-buffer is cleared to make the axis appear over the
	// original image.
	glClear(GL_DEPTH_BUFFER_BIT);

	// Tune for best line rendering
	glLineWidth(5.0);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-1, 1, -1, 1, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glMultMatrixd(camera()->orientation().inverse().matrix());

	float axis_size = 1.2f; // other 0.2 space for drawing the x, y, z labels
	drawAxis(axis_size); 

	// Draw text id
	glDisable(GL_LIGHTING);
	glColor3f(0, 0, 0);
	renderText(axis_size, 0, 0, "X");
	renderText(0, axis_size, 0, "Y");
	renderText(0, 0, axis_size, "Z");

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	// The viewport and the scissor are restored.
	glScissor(scissor[0], scissor[1], scissor[2], scissor[3]);
	glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);

	glEnable(GL_LIGHTING);
}


void PaintCanvas::mousePressEvent(QMouseEvent *e)
{
	makeCurrent();
	emit clicked();
	currenMousePos.setX( e->pos().x());
	currenMousePos.setY( e->pos().y());



	/*select mode*/
	if (single_operate_tool_!=nullptr &&    
		single_operate_tool_->tool_type()==Tool::SELECT_TOOL)
	{
		if(!(QApplication::keyboardModifiers() == Qt::AltModifier))
		{

			QGLViewer::mousePressEvent(e);
		}
		else
			single_operate_tool_->press(e);
	}
	else if (single_operate_tool_ != nullptr &&
		single_operate_tool_->tool_type() == Tool::MANIPULATE_TOOL)
	{
		if (QApplication::keyboardModifiers() == Qt::AltModifier)
		{

			QGLViewer::mousePressEvent(e);
		}
		else
			single_operate_tool_->press(e ,camera());
	}
	else
	{	
//		g_manipulator.mousePressEvent(e, camera());  //handle,use qglviewer
		if (m_io)    //handle ,derived from impricit skinning
			m_io->mousePressEvent(e);

		switch (e->button()) // bullet callback
		{
		case  Qt::LeftButton:
			if (pApp)
				pApp->Mouse(0, 0, e->pos().x(), e->pos().y());
			break;
		case Qt::RightButton:
			if (pApp)
				pApp->Mouse(2, 0, e->pos().x(), e->pos().y());

		default:
			break;
		}

//		if( !e->isAccepted())
			QGLViewer::mousePressEvent(e);
	}

}

void PaintCanvas::mouseMoveEvent(QMouseEvent *e)
{
//	cout<<"mousemove "<<e->x()<<" "<<e->y();
	makeCurrent();
	currenMousePos.setX( e->pos().x());
	currenMousePos.setY( e->pos().y());



	main_window_->showCoordinateAndIndexUnderMouse( e->pos() );
	/*select mode*/
	if (single_operate_tool_!=nullptr && 
		single_operate_tool_->tool_type()==Tool::SELECT_TOOL )
	{

		if(!(QApplication::keyboardModifiers() == Qt::AltModifier))
		{

			QGLViewer::mouseMoveEvent(e);
		}else
			single_operate_tool_->move(e);
	}
	else if (single_operate_tool_ != nullptr &&
		single_operate_tool_->tool_type() == Tool::MANIPULATE_TOOL)
	{
		if (QApplication::keyboardModifiers() == Qt::AltModifier)
		{

			QGLViewer::mouseMoveEvent(e);
		}
		else
			single_operate_tool_->move(e ,camera());
		updateGL();

	}
	else
	{ 
		//handle ,derived from impricit skinning
		if (_heuristic->_type == Tbx::Selection::CIRCLE   && _is_mouse_in)
		{
			updateGL();
		}
		if (pApp)  // bullet callback
			pApp->Motion(e->pos().x(), e->pos().y());

//		g_manipulator.mouseMoveEvent(e, camera());  //handle,use qglviewer

		if (m_io)  //handle ,derived from impricit skinning
			m_io->mouseMoveEvent(e);
		updateGL();

//		if( !e->isAccepted())
			QGLViewer::mouseMoveEvent(e);
	}

}

void PaintCanvas::mouseReleaseEvent(QMouseEvent *e)
{
	makeCurrent();
	currenMousePos.setX( -10); //clear
	currenMousePos.setY( -10);


	/*select mode*/
	if (single_operate_tool_!=nullptr && 
		single_operate_tool_->tool_type()==Tool::SELECT_TOOL
		)
	{
		if(!(QApplication::keyboardModifiers() == Qt::AltModifier))
		{

			QGLViewer::mouseReleaseEvent(e);
		}else
			single_operate_tool_->release(e);
	}
	else if (single_operate_tool_ != nullptr &&
		single_operate_tool_->tool_type() == Tool::MANIPULATE_TOOL)
	{
		if (QApplication::keyboardModifiers() == Qt::AltModifier)
		{

			QGLViewer::mouseReleaseEvent(e);
		}
		else
			single_operate_tool_->release(e ,camera());
			updateGL();

	}
	else
	{
		//handle,use qglviewer
//		g_manipulator.mouseReleaseEvent(e, camera());
		if (m_io)  //handle ,derived from impricit skinning
			m_io->mouseReleaseEvent(e);

		switch (e->button())   // bullet callback
		{
		case  Qt::LeftButton:
			if (pApp)
				pApp->Mouse(0, 1, e->pos().x(), e->pos().y());
			break;
		case Qt::RightButton:
			if (pApp)
				pApp->Mouse(2, 1, e->pos().x(), e->pos().y());

		default:
			break;
		}
//		if( !e->isAccepted())
			QGLViewer::mouseReleaseEvent(e);
	}

}

void PaintCanvas::wheelEvent(QWheelEvent *e)
{
	makeCurrent();
	if(pApp)
		pApp->Wheel( 0 ,e->delta() );

	//if(m_io)
	//	m_io->wheelEvent(e);

	////added by huayun
	////if (  e->orientation() == Qt::Horizontal){
	//	float numDegrees = e->delta() / 120;//滚动的角度，*8就是鼠标滚动的距离
	//	//int numSteps = numDegrees / 10;//滚动的步数，*15就是鼠标滚动的角度
	//	Paint_Param::g_point_size += numDegrees*0.1;
	//	if (Paint_Param::g_point_size < 0.)
	//	{
	//		Paint_Param::g_point_size = 0.;
	//	}
	//	if (Paint_Param::g_point_size  > 100.)
	//	{
	//		Paint_Param::g_point_size = 100.;
	//	}
	//	


	//	updateGL();
	//}

	//std::cout<<Qt::ControlModifier<<std::endl;
	if (e->modifiers() == Qt::ControlModifier)
	{
		int numDegrees = e->delta() / 120;

		Paint_Param::g_point_size += 1.f * numDegrees;
		if (Paint_Param::g_point_size < 0.)
		{
			Paint_Param::g_point_size = 0.;
		}

		updateGL();
	}
	

	if (e->modifiers() == Qt::AltModifier )
	{
		int numDegrees = e->delta() / 120;

		Paint_Param::g_step_size(2) += 0.1f * numDegrees;
	

		updateGL();
	}
	else if( e->modifiers() == (Qt::AltModifier|Qt::ControlModifier) )
	{
		int numDegrees = e->delta() / 120;

		Paint_Param::g_step_size(1) += 0.1f * numDegrees;


		updateGL();
	}
	else if( e->modifiers() == (Qt::AltModifier|Qt::ControlModifier|Qt::ShiftModifier) )  
	{
		int numDegrees = e->delta() / 120;

		Paint_Param::g_step_size(0) += 0.1f * numDegrees;


		updateGL();
	}
	//change line size
	if (is_key_l_pressed)
	{
		
		int numDegrees = e->delta() / 120;
		
		Paint_Param::g_line_size += 1.f * numDegrees;
		std::cout<<Paint_Param::g_line_size<<std::endl;
		if (Paint_Param::g_line_size < 0.)
		{
			Paint_Param::g_line_size = 0.1;
		}

		updateGL();
		return;
	}
	QGLViewer::wheelEvent(e);
}

//void MyWidget::wheelEvent(QWheelEvent *event)
//{
//	int numDegrees = event->delta() / 8;//滚动的角度，*8就是鼠标滚动的距离
//	int numSteps = numDegrees / 15;//滚动的步数，*15就是鼠标滚动的角度
//	if (event->orientation() == Qt::Horizontal) {       
//		scrollHorizontally(numSteps);       //水平滚动
//	} else {
//		scrollVertically(numSteps);       //垂直滚动
//	}
//	event->accept();      //接收该事件
//}


void PaintCanvas::keyPressEvent(QKeyEvent * e)
{
	makeCurrent();

	if (single_operate_tool_ != nullptr &&
		single_operate_tool_->tool_type() == Tool::MANIPULATE_TOOL)
	{
		if (QApplication::keyboardModifiers() == Qt::AltModifier)
		{

			QGLViewer::keyPressEvent(e);
		}
		else
			single_operate_tool_->keyPressEvent(e);
		updateGL();

	}



	if(m_io)
		m_io->keyPressEvent(e);
	switch (e->key())
	{
	case Qt::Key_Z :
		if(pApp)
			pApp->Keyboard( 'z',currenMousePos.x(), currenMousePos.y());
		break;
	case Qt::Key_X :
		if(pApp)
			pApp->Keyboard( 'x',currenMousePos.x(), currenMousePos.y());
		break;
	case Qt::Key_W :
		if(pApp)
			pApp->Keyboard( 'w',currenMousePos.x(), currenMousePos.y());
		break;
	case Qt::Key_B :
		if(pApp)
			pApp->Keyboard( 'b',currenMousePos.x(), currenMousePos.y());
		break;
	case Qt::Key_D :
		if(pApp)
			pApp->Keyboard( 'd',currenMousePos.x(), currenMousePos.y());
		break;
	default:
		break;
	}
	switch (e->key())
	{
	case Qt::Key_Left :
		if(pApp)
			pApp->Special( 0x0064,currenMousePos.x(), currenMousePos.y());
		break;
	case Qt::Key_Right :
		if(pApp)
			pApp->Special( 0x0066,currenMousePos.x(), currenMousePos.y());
		break;
	case Qt::Key_Up :
		if(pApp)
			pApp->Special( 0x0065,currenMousePos.x(), currenMousePos.y());
		break;
	case Qt::Key_Down :
		if(pApp)
			pApp->Special( 0x0067,currenMousePos.x(), currenMousePos.y());
		break;
	case	Qt::Key_Alt:
		if(pApp)
			pApp->Special( 0x0004,currenMousePos.x(), currenMousePos.y());
		break;
	default:
		break;
	}
	
	if ( e->key() ==Qt::Key_Delete )
	{
		if (single_operate_tool_!=nullptr && 
			single_operate_tool_->tool_type()==Tool::SELECT_TOOL
			)
		{
			SelectTool*	select_tool = dynamic_cast<SelectTool*>(single_operate_tool_);

			const std::vector<IndexType>& selected_items =  select_tool->get_selected_vertex_idx();
			
			IndexType cur_selected_sample_idx = select_tool->cur_sample_to_operate();
			Sample& smp = (*Global_SampleSet)[cur_selected_sample_idx];

			smp.lock();
			smp.delete_vertex_group( selected_items);
			smp.unlock();
			
			//reset tree widget
			main_window_->createTreeWidgetItems();
			updateGL();
		}
	}
	if( e->key() == Qt::Key_Right || e->key() == Qt::Key_Left ||e->key() == Qt::Key_Up ||e->key() == Qt::Key_Down)
	{
		SampleSet& ss = (*Global_SampleSet);
		std::vector<IndexType> visbleindex;
		for( int i = 0 ;i < ss.size() ; ++i)
		{
			if(ss[i].is_visible())visbleindex.push_back(i);
		}
		IndexType max_index ;
		IndexType next_index;
		switch (e->key())
		{

		case Qt::Key_Right :
			{
				if(!visbleindex.size())return;
				else max_index = visbleindex[visbleindex.size() -1];
				if( max_index + 1 < ss.size())
				{
					next_index = max_index + 1;
				}else next_index = 0;
				ss[max_index].set_visble(false);
				ss[next_index].set_visble(true);
			}

			break ;
		case Qt::Key_Up :
			{
				if(!visbleindex.size())return;
				else max_index = visbleindex[0];
				if( max_index  > 0)
				{
					next_index = max_index - 1;
				}else next_index = ss.size() -1;
				ss[max_index].set_visble(false);
				ss[next_index].set_visble(true);
			}

			break ;
		case Qt::Key_Left :
			{
				if(!visbleindex.size())return;
				else max_index = visbleindex[0];
				if( max_index  > 0)
				{
					next_index = max_index - 1;
				}else next_index = ss.size() -1;
				ss[max_index].set_visble(false);
				ss[next_index].set_visble(true);
			}

			break ;
		case Qt::Key_Down :
			{
				if(!visbleindex.size())return;
				else max_index = visbleindex[visbleindex.size() -1];
				if( max_index + 1 < ss.size())
				{
					next_index = max_index + 1;
				}else next_index = 0;
				ss[max_index].set_visble(false);
				ss[next_index].set_visble(true);
			}
			break ;
		default :
			break ;
		}
		updateGL();	
		main_window_->getLayerdialog()->updateTable(0);
	}
	
	if (single_operate_tool_!=nullptr && 
		single_operate_tool_->tool_type()==Tool::SELECT_TOOL
		)
	{
		IndexType target_label = 0;
		switch(e->key() )
		{
			case Qt::Key_0 :
				{
					target_label = 0;
					Logger<<"0";
					break;
				}
			case Qt::Key_1 :
				{
					target_label = 1;
					break;
				}
			case Qt::Key_2 :
				{
					target_label = 2;
					break;

				}
			case Qt::Key_3 :
				{
					target_label = 3;
					break;
				}
			case Qt::Key_4 :
				{
					target_label = 4;
					break;
				}
			case Qt::Key_5:
				{
					target_label = 5;
					break;
				}
			case Qt::Key_6 :
				{
					target_label = 6;
					break;
				}
			case Qt::Key_7 :
				{
					target_label = 7;
					break;
				}
			case Qt::Key_8 :
				{
					target_label = 8;
					break;
				}
			case Qt::Key_9 :
				{
					target_label = 9;
					break;
				}
			case Qt::Key_A :
				{
					target_label = 10;
					Logger<<"A";
					break;
				}
			case Qt::Key_B :
				{
					target_label = 11;
					break;
				}
			case Qt::Key_C :
				{
					target_label = 12;
					break;
				}
			case Qt::Key_D :
				{
					target_label = 13;
					break;
				}
			case Qt::Key_E:
				{
					target_label = 14;
					break;
				}
			case Qt::Key_F :
				{
					target_label = 15;
					break;
				}
			case Qt::Key_G :
				{
					target_label = 16;
					break;
				}
			case Qt::Key_H :
				{
					target_label = 17;
					break;
				}
			case Qt::Key_I :
				{
					target_label = 18;
					break;
				}
			case Qt::Key_J :
				{
					target_label = 19;
					break;
				}
			case Qt::Key_K :
				{
					target_label = 20;
					break;
				}
			case Qt::Key_Alt:
				{
					return; //ignore it
				}
			default:
				{
					return; //target_label = 0;
				}
		}
		SelectTool*	select_tool = dynamic_cast<SelectTool*>(single_operate_tool_);

		const std::vector<IndexType>& selected_items =  select_tool->get_selected_vertex_idx();

		IndexType cur_selected_sample_idx = select_tool->cur_sample_to_operate();
		Sample& smp = (*Global_SampleSet)[cur_selected_sample_idx];

		smp.lock();
		//Logger<<"selected "<<selected_items.size()<<"  "<<" label"<<target_label;
		smp.set_vertex_label(selected_items ,target_label);
		smp.smoothLabel();
		smp.unlock();

		//reset tree widget
		main_window_->createTreeWidgetItems();
		updateGL();	

	}

	
	if ( e->key() == Qt::Key_L &&!e->isAutoRepeat())
	{
		
		std::cout<<"key_l_pressed"<<std::endl;
		is_key_l_pressed = true;
	}
}
void PaintCanvas::keyReleaseEvent(QKeyEvent * e)
{
	makeCurrent();
	if(m_io)
		m_io->keyReleaseEvent(e);
	//if(e->isAutoRepeat())
	//{
	//	e->ignore();
	//	return;
	//}
	if ( e->key() == Qt::Key_L&& !e->isAutoRepeat())
	{
		std::cout<<"key_l_release"<<std::endl;
		is_key_l_pressed = false;
	}
	if (pApp)
	{
		pApp->KeyboardUp( 'z',currenMousePos.x(), currenMousePos.y());
		pApp->SpecialUp( 'z',currenMousePos.x(), currenMousePos.y());
	}
				
}

void PaintCanvas::resizeEvent(QResizeEvent* e)
{	
	qglviewer::Camera* c_camera = camera();
	makeCurrent();
	if(!m_render_ctx)m_render_ctx = new Render_context( width() ,height());
	m_render_ctx->reshape( e->size().width() ,e->size().height());
	if(pApp)
		pApp->Reshape(e->size().width() ,e->size().height());
	if(m_instancingRenderer)
		m_instancingRenderer->resize(e->size().width(),e->size().height());
}
void PaintCanvas::showSelectedTraj()
{
	if (single_operate_tool_!=nullptr /*&& 
									  Register_Param::g_is_traj_compute == true*/)
	{
		SelectTool*	select_tool = dynamic_cast<SelectTool*>(single_operate_tool_);
		const std::vector<IndexType>& selected_items =  select_tool->get_selected_vertex_idx();

		Tracer& tracer = Tracer::get_instance();
		tracer.clear_records();

		IndexType m = ((*Global_SampleSet)).size(); 
		IndexType n = ((*Global_SampleSet)[0]).num_vertices();
		MatrixXXi mat( n,m );
		for (IndexType i =0; i < m; i++)
		{
			for (IndexType j=0; j<n; j++)
			{
				mat(j,i) = j;
			}
		}

		Register_Param::g_traj_matrix = mat;

		for ( IndexType i=0; i<selected_items.size(); i++ )
		{
			IndexType selected_idx = selected_items[i];
			
			auto traj = Register_Param::g_traj_matrix.row(selected_idx);
			IndexType traj_num = Register_Param::g_traj_matrix.cols() -1;

			for ( IndexType j=0; j<traj_num; j++ )
			{
				tracer.add_record(  j, traj(j) , j+1, traj(j+1) );
			}
		}
	}
	this->show_trajectory_ = true;
}
void PaintCanvas::showSelectedFrameLabel(std::vector<int>& showed_label,int curSelectedFrame)
{
	Sample& csmp= (*Global_SampleSet)[curSelectedFrame];
	for( auto iter = csmp.begin();iter !=csmp.end();++iter)
	{
		int label_id = (*iter)->label();
		int i;
		for( i = 0; i <showed_label.size() ;++i)
		{
			if( label_id == showed_label[i] )break;
		}
		if( i == showed_label.size())(*iter)->set_visble(false);
		else (*iter)->set_visble(true);
	}
	Logger<<showed_label.size()<<"  "<<curSelectedFrame;

}
void PaintCanvas::showSelectedlabelTraj(std::vector<int>& _selectedlabeltraj)
{
	/*Tracer& tracer = Tracer::get_instance();
	tracer.clear_records();

	IndexType m = ((*Global_SampleSet)).size(); 
	IndexType n = ((*Global_SampleSet)[0]).num_vertices();

	for ( IndexType i=0; i<selected_items.size(); i++ )
	{
	IndexType selected_idx = selected_items[i];

	for ( IndexType j=0; j<traj_num; j++ )
	{
	tracer.add_record(  j, traj(j) , j+1, traj(j+1) );
	}
	}*/

}



void PaintCanvas::pasteTile()
{
	QString outfile;

	glPushAttrib(GL_ENABLE_BIT);
	QImage tileBuffer=grabFrameBuffer(true).mirrored(false,true);
	if(ss->tiledSave)
	{
		outfile=QString("%1/%2_%3-%4.png")
			.arg(ss->outdir)
			.arg(ss->basename)
			.arg(tileCol,2,10,QChar('0'))
			.arg(tileRow,2,10,QChar('0'));
		tileBuffer.mirrored(false,true).save(outfile,"PNG");
	}
	else
	{
		if (snapBuffer.isNull())
			snapBuffer = QImage(tileBuffer.width() * ss->resolution, tileBuffer.height() * ss->resolution, tileBuffer.format());

		uchar *snapPtr = snapBuffer.bits() + (tileBuffer.bytesPerLine() * tileCol) + ((totalCols * tileRow) * tileBuffer.byteCount());
		uchar *tilePtr = tileBuffer.bits();

		for (int y=0; y < tileBuffer.height(); y++)
		{
			memcpy((void*) snapPtr, (void*) tilePtr, tileBuffer.bytesPerLine());
			snapPtr+=tileBuffer.bytesPerLine() * totalCols;
			tilePtr+=tileBuffer.bytesPerLine();
		}
	}
	tileCol++;
	if (tileCol >= totalCols)
	{
		tileCol=0;
		tileRow++;

		if (tileRow >= totalRows)
		{
			if(ss->snapAllLayers)
			{
				outfile=QString("%1/%2%3_L%4.png")
					.arg(ss->outdir).arg(ss->basename)
					.arg(ss->counter,2,10,QChar('0'))
					.arg(currSnapLayer,2,10,QChar('0'));
			} else {
				outfile=QString("%1/%2%3.png")
					.arg(ss->outdir).arg(ss->basename)
					.arg(ss->counter++,2,10,QChar('0'));
			}

			if(!ss->tiledSave)
			{
				bool ret = (snapBuffer.mirrored(false,true)).save(outfile,"PNG");
				if (ret)
				{
					/*this->Logf(GLLogStream::SYSTEM, "Snapshot saved to %s",outfile.toLocal8Bit().constData());*/
				}
				else
				{
					/*Logf(GLLogStream::WARNING,"Error saving %s",outfile.toLocal8Bit().constData());*/
				}
			}
			takeSnapTile=false;
			snapBuffer=QImage();
		}
	}
	updateGL();
	glPopAttrib();
	
}

void PaintCanvas::saveSnapshot()
{
	//snap all layers
	currSnapLayer = 0;
	//number of subparts
	totalCols = totalRows = ss->resolution;
	tileRow = tileCol = 0;
	if(ss->snapAllLayers)
	{
		while(currSnapLayer < (*Global_SampleSet).size())
		{
			tileRow = tileCol = 0;
			//SET CURRMESH()

		}
	}else
	{
		takeSnapTile = true;
		saveSnapshotImp(*ss);
		updateGL();
	}
}
void PaintCanvas::saveSnapshotImp(SnapshotSetting& _ss)
{


	//glPushAttrib(GL_ENABLE_BIT);
	QImage tileBuffer;
	tileBuffer =grabFrameBuffer(true).mirrored(false,true);
	glPushAttrib(GL_ENABLE_BIT);
	totalCols = totalRows = _ss.resolution;
	tileRow = tileCol = 0;
	QString outfile;


	//if (snapBuffer.isNull())
		snapBuffer = QImage( tileBuffer.width() * _ss.resolution, tileBuffer.height() * _ss.resolution, tileBuffer.format());

	
		for( int tileRow = 0 ; tileRow < totalRows ; ++tileRow)
		{
			for( int tileCol = 0 ; tileCol < totalCols ;++tileCol)
			{
				preDraw();
				glMatrixMode(GL_PROJECTION);
				glLoadIdentity();
				setTileView( totalCols ,totalRows ,tileCol ,tileRow);
				glMatrixMode(GL_MODELVIEW);
				draw();
				postDraw();
				tileBuffer =grabFrameBuffer(true).mirrored(false,true);

				uchar *snapPtr = snapBuffer.bits() + (tileBuffer.bytesPerLine() * tileCol) + ((totalCols * tileRow) * tileBuffer.byteCount());
				uchar *tilePtr = tileBuffer.bits();

				for (int y=0; y < tileBuffer.height(); y++)
				{
					memcpy((void*) snapPtr, (void*) tilePtr, tileBuffer.bytesPerLine());
					snapPtr+=tileBuffer.bytesPerLine() * totalCols;
					tilePtr+=tileBuffer.bytesPerLine();
				}
			
			}

		}

	outfile=QString("%1/%2%3.png")
		.arg(_ss.outdir).arg(_ss.basename)
		.arg(_ss.counter++,2,10,QChar('0'));
	bool ret = (snapBuffer.mirrored(false,true)).save(outfile,"PNG");
	takeSnapTile = false;
	updateGL();
	glPopAttrib();


}


void PaintCanvas::setTileView( IndexType totalCols  , IndexType totalRows ,IndexType tileCol ,IndexType tileRow )
{
	glViewport(0 ,0 , this->width() ,this->height());
	GLfloat fApspect = (GLfloat)width()/height();
	nearPlane =  camera()->zNear();
	farPlane = camera()->zFar();
	
	ScalarType deltaY = 2*nearPlane * tan(camera()->fieldOfView() / 2.0) /totalRows;
	ScalarType deltaX = deltaY* fApspect;
	//ScalarType xMin = -this->width()/2.0;
	//ScalarType yMin = -this->height()/2.0;
	ScalarType yMin = - nearPlane * tan(camera()->fieldOfView() / 2.0);
	ScalarType xMin =  yMin* fApspect;



	if (camera()->type() == qglviewer::Camera::PERSPECTIVE)
		glFrustum(xMin + tileCol*deltaX, xMin + (tileCol+1)*deltaX, yMin + (tileRow)*deltaY, yMin + (tileRow+1)*deltaY, nearPlane, farPlane);
	else glOrtho(xMin + tileCol*deltaX, xMin + (tileCol+1)*deltaX, yMin + (tileRow)*deltaY, yMin + (tileRow+1)*deltaY, nearPlane, farPlane);
	
}
void PaintCanvas::setView()
{
	glViewport(0 ,0 , this->width() ,this->height());
	GLfloat fApspect = (GLfloat)width()/height();
	glMatrixMode(GL_PROJECTION);
	/*glLoadIdentity();*/

	float viewRatio  =1.75f;
	float cameraDist = viewRatio /tanf( (float)PI * (fov* .5f) /180.0f);
	if(fov <5)cameraDist =1000;  //small hack for othographic projection where camera distance is rather meaningless->.
	nearPlane = cameraDist - 2.f* clipRatioNear;
	farPlane = cameraDist + 10.f* clipRatioFar;
	if(nearPlane <cameraDist*.1f)nearPlane = cameraDist*.1f;
	if(!takeSnapTile)
	{
		if(fov ==5) glOrtho( -viewRatio*fApspect ,viewRatio* fApspect , -viewRatio ,viewRatio,cameraDist-2.f*clipRatioNear ,cameraDist+2.f*clipRatioFar);
		/*else gluPerspective(fov , fApspect , nearPlane ,farPlane);*/
		else 
		{
			/*camera()->lookAt( qglviewer::Vec())*/
		}

	}
	else setTiledView( fov, viewRatio , fApspect ,nearPlane ,farPlane , cameraDist);
	glMatrixMode(GL_MODELVIEW);
	//glLoadIdentity;
	//gluLookAt(0 ,0 , cameraDist , 0 ,0 ,0 ,0 ,1 ,0);

}

void PaintCanvas::setTiledView(GLdouble fovY , float viewRatio , float fAspect , GLdouble zNear ,GLdouble zFar , float cameraDist)
{
	if(fovY <=5)
	{
		/*GLdouble fLeft = -viewRatio*fAspect;
		GLdouble fright = viewRatio*fAspect;
		GLdouble fboo*/

	}else
	{
		GLdouble fTop = zNear * tan((float)PI * (fov* .5f) /180.0f);
		GLdouble fRight = fTop* fAspect;
		GLdouble fBottom = -fTop;
		GLdouble fLeft = - fRight;
		//tile Dimension
		GLdouble tDimX = fabs( fRight -fLeft)/ totalCols;
		GLdouble tDimY = fabs(fTop - fBottom)/ totalRows;
	/*	glFrustum( fLeft + tDimX * tileCol ,fLeft + tDimX*(tileCol+1), 
			fBottom + tDimY * tileRow , fBottom + tDimY * (tileRow+1) , zNear ,zFar);*/

	}
}

void PaintCanvas::Logf(int level ,const char* f){}

void PaintCanvas::savePLY(SavePlySetting& ss)
{
	char fullPath[250];
	char format[30];
	FileIO::FILE_TYPE ftype;
	if( "ply" == ss.format)
	{
		ftype = FileIO::PLY;
	}else if( "obj" == ss.format)
	{
		ftype = FileIO::OBJ;
	}else if( "off" == ss.format)
	{
		ftype = FileIO::OFF;
	}else
	{

	}
	for( int i = ss.startframe ; i <= ss.endframe;++i)
	{
		sprintf( fullPath ,"%s%s%s%.3d%s",std::string(ss.outdir.toLocal8Bit().constData()).c_str() ,"/", std::string(ss.basename.toLocal8Bit().constData()).c_str(), i ,".obj");
		FileIO::saveFile(std::string(fullPath),ftype , i);
	}
	
	

	//QString outfile;
	/*outfile=QString("%1/%2%3.png")
	.arg(ss->outdir).arg(ss->basename)
	.arg(ss->counter++,2,10,QChar('0'));*/
	//char* output_file_path_ = (char*)outfile.toStdString().c_str();
	
	

}
void PaintCanvas::saveLabelFile(std::string filename ,IndexType selected_frame_idx)
{
	if(selected_frame_idx <0||selected_frame_idx +1>(*Global_SampleSet).size())return;
	//char fullPath[250];
	//sprintf( fullPath ,"%s","./squat2_edit.seg");     //必须加入.3d ，使得文件排序正常
	std::ofstream outfile( filename , std::ofstream::out);
	SampleSet& smpset =  (*Global_SampleSet);
	smpset[selected_frame_idx].set_visble(true);
	for( auto  vtxbitr = smpset[selected_frame_idx].begin() ; vtxbitr != smpset[selected_frame_idx].end() ;++vtxbitr ){
		Vertex& vtx = **vtxbitr;
		outfile<<vtx.label()<<std::endl;

	}

	outfile.close();

}
void PaintCanvas::getLabelFromFile(std::string filename ,IndexType selected_frame_idx)
{
	if(selected_frame_idx <0||selected_frame_idx +1>(*Global_SampleSet).size())return;
	//char fullPath[250];
	//sprintf( fullPath ,"%s","D:/zzb/zzb_lowrank/new_STED/original/squat2/11label/squat2_edit.seg");     //必须加入.3d ，使得文件排序正常
	std::ifstream infile( filename , std::ofstream::in);
	SampleSet& smpset =  (*Global_SampleSet);
	smpset[selected_frame_idx].set_visble(true);
	for( auto  vtxbitr = smpset[selected_frame_idx].begin() ; vtxbitr != smpset[selected_frame_idx].end() ;++vtxbitr ){
		int label = 0;
		infile>>label;
		Vertex& vtx = **vtxbitr;
		vtx.set_label(label);

	}

	infile.close();
}

void PaintCanvas::BulletOpenGLApplicationCallBack(BulletOpenGLApplication* bc)
{
	pApp = bc;
}

void PaintCanvas::set_gizmo(Gizmo::Gizmo_t type)
{
	makeCurrent();
	Gizmo* tmp = _gizmo;
	switch(type)
	{
	case Gizmo::TRANSLATION: _gizmo = new Gizmo_trans();     break;
	case Gizmo::SCALE:       _gizmo = new Gizmo_trans();     break;
	case Gizmo::ROTATION:    _gizmo = new Gizmo_rot();       break;
	case Gizmo::TRACKBALL:   _gizmo = new Gizmo_trackball(); break;
	}
	_gizmo->copy(tmp);
	delete tmp;
	updateGL();
}

void PaintCanvas::set_io(EOGL_widget::IO_t io_type)
{
	makeCurrent();
	delete _io;

	switch(io_type)
	{
	//case EOGL_widget::RBF:         _io = new IO_RBF(this);         break;
	//case EOGL_widget::DISABLE:     _io = new IO_disable_skin(this);     break;
	case EOGL_widget::GRAPH:       m_io = new IO_graph(this);       break;
	case EOGL_widget::SKELETON:    m_io = new IO_skeleton(this);    break;
	case EOGL_widget::MESH_EDIT:   m_io = new IO_mesh_edit(this);   break;
	default:          m_io = 0; break;
	}
	updateGL();
}

void PaintCanvas::set_selection(EOGL_widget::Select_t select_mode)
{
	using namespace  Tbx;
	delete _heuristic;
	switch(select_mode)
	{
	case EOGL_widget::MOUSE:     _heuristic = new Selection_nearest<int>(); break;
	case EOGL_widget::CIRCLE:    _heuristic = new Selection_circle<int>();  break;
	case EOGL_widget::BOX:       _heuristic = 0; break;
	case EOGL_widget::FREE_FORM: _heuristic = 0; break;
	default:        _heuristic = 0; break;
	}
}

void PaintCanvas::enterEvent(QEvent* e)
{
	cout<<"PaintCanvas enter";
	_is_mouse_in = true;
	
	Global_Window->update_viewports();
	e->ignore();
}

void PaintCanvas::leaveEvent( QEvent* e){
	cout<<"PaintCanvas leave";
	_is_mouse_in = false;
	Global_Window->update_viewports();
	e->ignore();
}

void PaintCanvas::set_draw_skeleton(bool s)
{
	m_render_ctx->_skeleton = s;
}

void PaintCanvas::set_phong(bool s)
{
	m_render_ctx->_plain_phong = s;
}

void PaintCanvas::set_textures(bool s)
{
	m_render_ctx->_textures = s;
}

void PaintCanvas::set_raytracing(bool s)
{
	m_render_ctx->_raytrace = s;
}

void PaintCanvas::set_rest_pose(bool s)
{
	m_render_ctx->_rest_pose = s;
}

bool PaintCanvas::rest_pose()
{
	return m_render_ctx->_rest_pose;
}

void PaintCanvas::set_draw_mesh(bool s)
{
	m_render_ctx->_draw_mesh = s;
}

bool PaintCanvas::draw_mesh()
{
	return m_render_ctx->_draw_mesh;
}

bool PaintCanvas::phong_rendering() const
{
	return m_render_ctx->_plain_phong;
}

bool PaintCanvas::raytrace() const
{
	return m_render_ctx->_raytrace;
}


void PaintCanvas::drawWithNames()
{
	Logger << "PaintCanvas::drawWithNames()" << std::endl;
}

void PaintCanvas::postSelection(const QPoint& point)
{

	QGLViewer::postSelection(point);
	if (single_operate_tool_ != nullptr &&
		single_operate_tool_->tool_type() == Tool::MANIPULATE_TOOL)
	{

		single_operate_tool_->postManipulateToolSelection();
		updateGL();

	}


}

static Tbx::Vec3 cog_selection(bool skel_mode)
{
	using namespace Cuda_ctrl;


	Tbx::Vec3 p0 = _anim_mesh->cog_mesh_selection();
	Tbx::Vec3 p1 = _anim_mesh->cog_sample_selection();

	int s_samp = _anim_mesh->get_selected_samples().size();
	int s_mesh = _anim_mesh->get_nb_selected_points();

	if( s_samp > 0 && s_mesh > 0)
		return (p0 + p1) * 0.5f;
	else if(s_samp > 0)
		return p1;
	else
		return p0;


}

// -----------------------------------------------------------------------------

static Tbx::Vec3 bone_cog(int bone_id)
{
	const Bone* b = g_skel->get_bone( bone_id );
	return (b->org() + b->end()) * 0.5f;
}

void PaintCanvas::update_pivot()
{
	using namespace Cuda_ctrl;
	const std::vector<int>& set = _skeleton.get_selection_set();

	if(_pivot_mode == EOGL_widget::FREE)
		return;
	else if(_pivot_mode == EOGL_widget::SELECTION)
	{
		if( _anim_mesh != 0 )
			_pivot = cog_selection(m_render_ctx->_skeleton);
	}
	else if(_pivot_mode == EOGL_widget::USER)
		_pivot = _pivot_user;
	else if(_pivot_mode == EOGL_widget::BONE)
	{
		if(set.size() > 0) _pivot = bone_cog(set[set.size()-1]);
	}
	else if(_pivot_mode == EOGL_widget::JOINT)
	{
		Tbx::Vec3 v;
		if(set.size() > 0){
			int idx = set[set.size()-1];
			if(idx > -1){
				v = Cuda_ctrl::_skeleton.joint_pos(idx);
				_pivot = Tbx::Vec3(v.x, v.y, v.z);
			}
		}else if(_graph.get_selected_node() > -1 ) {
			v = _graph.get_vertex( _graph.get_selected_node() );
			_pivot = Tbx::Vec3(v.x, v.y, v.z);
		}
	}
}
