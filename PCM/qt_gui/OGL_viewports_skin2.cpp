#include "paint_canvas.h"
#include "OGL_viewports_skin2.hpp"
#include "main_window.h"
#include "cameraViewer.h"
#include "global_datas/macros.hpp"
#include "../control/cuda_ctrl.hpp"

#include <QSplitter>

// Export from global.hpp //////////
//#include "g_scene_tree.hpp"
//#include "toolbox/gl_utils/glshoot.hpp"
//extern GlShoot* g_oglss;
extern bool g_shooting_state;
extern std::string g_write_dir;
extern bool isCameraViewerOn;

///////////////////////////////////

// -----------------------------------------------------------------------------

void OGL_widget_skin2_hidden::initializeGL(){
    //OGL_widget_skin2::init_glew();
    //init_opengl();
}

// -----------------------------------------------------------------------------

OGL_viewports_skin2::OGL_viewports_skin2(QWidget* w, main_window* m) :
    QFrame(w),
    _skel_mode(false),
 //   _io_type(EOGL_widget::DISABLE),
    _current_viewport(0),
	_prev_viewport(0),
    _main_layout(0),
    _main_window(m),
    _frame_count(0)
{

    // Initialize opengl shared context
//    _hidden = new OGL_widget_skin2_hidden(this);
    //_hidden->setGeometry(0,0,0,0);
//    _hidden->hide();
    // Needed to initialize the opengl shared context
//    _hidden->updateGL();
//    _hidden->makeCurrent();

	_cameraViewer = new CameraViewer();
    set_viewports_layout(SINGLE);
    QObject::connect(this, SIGNAL(active_viewport_changed(int)),
                     m   , SLOT(active_viewport(int)) );

    QObject::connect(this        , SIGNAL(update_status(QString)),
                     m->getUI()->statusBar, SLOT(showMessage(QString)) );
}


OGL_viewports_skin2::~OGL_viewports_skin2()
{
	erase_viewports();
	erase_cameraViewers();
}

// -----------------------------------------------------------------------------

void OGL_viewports_skin2::update_scene()
{
    //Scene_tree::iterator it = g_scene_tree->begin();
    //for(; it < g_scene_tree->end(); ++it)
    //{
    //    if( (*it)->type_object() == EObj::MESH)
    //    {
    //        Obj_mesh::Deformer* dfm = ((Obj_mesh*)(*it))->get_deformer();
    //        if(dfm != 0) dfm->deform();
    //    }
    //}
}

// -----------------------------------------------------------------------------

void OGL_viewports_skin2::updateGL()
{
	using namespace Cuda_ctrl;
 //   using namespace Cuda_ctrl;
    _viewports[0]->makeCurrent();

    // FPS Counting _______________
    static int fpsCount = -1;
    static int fpsLimit = 10;
    static double fps_min = 100000.;
    static double fps_max = 0.;
    static double fps_avg = 0.;
    static int nb_frames_avg = 0;
    static unsigned elapsed = 0.;
    if(fpsCount == -1) fpsCount = 0;

    // _____________________________

    //update_scene();
	if(_skel_mode)
	{
		if(_anim_mesh != 0)
		{

			// Transform HRBF samples for display and selection
			_anim_mesh->transform_samples( /*_skeleton.get_selection_set() */);
			// Animate the mesh :
			//if( !_anim_mesh->_incremental_deformation )
			_anim_mesh->deform_mesh(true);
		}

	}





    // Draw every viewports
    for(unsigned i = 0; i < _viewports.size(); i++)
    {
        PaintCanvas* v = _viewports[i];
        v->updateGL();
        // Screenshot of the active viewport
        if(g_shooting_state && v == active_viewport())
        {
            //g_oglss->set_img_size(v->width(), v->height());
            //g_oglss->shoot();
        }
    }

    // FPS Counting ________________
    elapsed += _fps_timer.elapsed();
    fpsCount++;
    if (fpsCount >= fpsLimit) {
        double ifps =  (double)fpsCount / ((double)elapsed / 1000.);
        if(ifps < fps_min) fps_min = ifps;
        if(ifps > fps_max) fps_max = ifps;
        fps_avg += ifps;
        nb_frames_avg++;
        QString msg = QString::number(ifps) +" fps ";
        msg = msg+"min: "+QString::number(fps_min)+" ";
        msg = msg+"max: "+QString::number(fps_max)+" ";
        msg = msg+"avg: "+QString::number(fps_avg/nb_frames_avg)+" ";
        msg = msg+"Implicit Visualizer:"+QString::number(ifps)+"fps, ";

        if(_current_viewport != 0)
        {
            double w = _current_viewport->camera()->screenWidth();
            double h = _current_viewport->camera()->screenHeight();
            double frame_to_Mrays = w*h*MULTISAMPX*MULTISAMPY*(1e-6);
            msg = msg+QString::number(ifps*frame_to_Mrays)+" Mray/s";
            msg = msg+" res:"+QString::number(width())+"x"+QString::number(height());
        }

        emit update_status(msg);

        if(nb_frames_avg >= fpsLimit*2){
            nb_frames_avg = 0;
            fps_avg = 0.;
            fps_min = 1000.;
            fps_max = 0.;
        }
        fpsCount = 0;
        elapsed = 0;
    }
    // _____________________________
}

// -----------------------------------------------------------------------------

void OGL_viewports_skin2::enterEvent( QEvent* e){
    if(_current_viewport != 0)
        _current_viewport->setFocus();
    e->ignore();
}

// -----------------------------------------------------------------------------

void OGL_viewports_skin2::set_viewports_layout(Layout_e setting)
{
    erase_viewports();
//	erase_cameraViewers();

    switch(setting)
    {
    case SINGLE:  _main_layout = gen_single (); break;
    case VDOUBLE: _main_layout = gen_vdouble(); break;
    case HDOUBLE: _main_layout = gen_hdouble(); break;
    case FOUR:    _main_layout = gen_four   (); break;
    }

    this->setLayout(_main_layout);
    first_viewport_as_active();
//    set_io(_io_type);
}

// -----------------------------------------------------------------------------

QLayout* OGL_viewports_skin2::gen_single()
{
    //QVBoxLayout* vlayout = new QVBoxLayout(this);
    //vlayout->setSpacing(0);
    //vlayout->setContentsMargins(0, 0, 0, 0);

    //{
    //    Viewport_frame_skin2* frame = new_viewport_frame(this, 0);
    //    _viewports_frame.push_back(frame);
    //    QVBoxLayout* layout = new QVBoxLayout(frame);
    //    layout->setSpacing(0);
    //    layout->setContentsMargins(0, 0, 0, 0);
    //    frame->setLayout(layout);

    //    PaintCanvas* ogl = new_viewport(frame);
    //    _viewports.push_back(ogl);
    //    layout->addWidget(ogl);

    //    vlayout->addWidget(frame);
    //}
	QVBoxLayout* vlayout = new QVBoxLayout(this);
	vlayout->setSpacing(0);
	vlayout->setContentsMargins(0, 0, 0, 0);

	QSplitter* splitter = new QSplitter(this);
	splitter->setOrientation(Qt::Vertical);
	splitter->setContentsMargins(0, 0, 0, 0);
	splitter->setHandleWidth(3);
	vlayout->addWidget(splitter);

	for(int i = 0; i < 1; i++){
		Viewport_frame_skin2* frame = new_viewport_frame(splitter, i);
		_viewports_frame.push_back(frame);
		QVBoxLayout* layout = new QVBoxLayout(frame);
		layout->setSpacing(0);
		layout->setContentsMargins(0, 0, 0, 0);
		frame->setLayout(layout);

		PaintCanvas* ogl = new_viewport(frame);
		_viewports.push_back(ogl);
		layout->addWidget(ogl);

		splitter->addWidget(frame);
	}
	return vlayout;
}

// -----------------------------------------------------------------------------

QLayout* OGL_viewports_skin2::gen_vdouble()
{

    QVBoxLayout* vlayout = new QVBoxLayout(this);
    vlayout->setSpacing(0);
    vlayout->setContentsMargins(0, 0, 0, 0);

    QSplitter* splitter = new QSplitter(this);
    splitter->setOrientation(Qt::Horizontal);
    splitter->setContentsMargins(0, 0, 0, 0);
    splitter->setHandleWidth(3);
    vlayout->addWidget(splitter);

    for(int i = 0; i < 2; i++){
        Viewport_frame_skin2* frame = new_viewport_frame(splitter, i);
        _viewports_frame.push_back(frame);
        QVBoxLayout* layout = new QVBoxLayout(frame);
        layout->setSpacing(0);
        layout->setContentsMargins(0, 0, 0, 0);
        frame->setLayout(layout);

        PaintCanvas* ogl = new_viewport(frame);
        _viewports.push_back(ogl);
        layout->addWidget(ogl);

        splitter->addWidget(frame);
    }
	return vlayout;
}

// -----------------------------------------------------------------------------

QLayout* OGL_viewports_skin2::gen_hdouble()
{

    QVBoxLayout* vlayout = new QVBoxLayout(this);
    vlayout->setSpacing(0);
    vlayout->setContentsMargins(0, 0, 0, 0);

    QSplitter* splitter = new QSplitter(this);
    splitter->setOrientation(Qt::Vertical);
    splitter->setContentsMargins(0, 0, 0, 0);
    splitter->setHandleWidth(3);
    vlayout->addWidget(splitter);

    for(int i = 0; i < 2; i++){
        Viewport_frame_skin2* frame = new_viewport_frame(splitter, i);
        _viewports_frame.push_back(frame);
        QVBoxLayout* layout = new QVBoxLayout(frame);
        layout->setSpacing(0);
        layout->setContentsMargins(0, 0, 0, 0);
        frame->setLayout(layout);

        PaintCanvas* ogl = new_viewport(frame);
        _viewports.push_back(ogl);
        layout->addWidget(ogl);

        splitter->addWidget(frame);
    }
    return vlayout;
}

// -----------------------------------------------------------------------------

QLayout* OGL_viewports_skin2::gen_four()
{

    QVBoxLayout* vlayout = new QVBoxLayout(this);
    vlayout->setSpacing(0);
    vlayout->setContentsMargins(0, 0, 0, 0);

    QSplitter* vsplitter = new QSplitter(this);
    vsplitter->setOrientation(Qt::Vertical);
    vsplitter->setContentsMargins(0, 0, 0, 0);
    vsplitter->setHandleWidth(3);
    vlayout->addWidget(vsplitter);

    int acc = 0;
    for(int i = 0; i < 2; i++)
    {
        QSplitter* hsplitter = new QSplitter(this);
        hsplitter->setOrientation(Qt::Horizontal);
        hsplitter->setContentsMargins(0, 0, 0, 0);
        hsplitter->setHandleWidth(3);
        vsplitter->addWidget(hsplitter);

        for(int j = 0; j < 2; j++)
        {
            Viewport_frame_skin2* frame = new_viewport_frame(hsplitter, acc);
            acc++;
            _viewports_frame.push_back(frame);
            QVBoxLayout* layout = new QVBoxLayout(frame);
            layout->setSpacing(0);
            layout->setContentsMargins(0, 0, 0, 0);
            frame->setLayout(layout);

            PaintCanvas* ogl = new_viewport(frame);
            _viewports.push_back(ogl);
            layout->addWidget(ogl);

            hsplitter->addWidget(frame);
        }
    }

    return vlayout;
}

// -----------------------------------------------------------------------------

void OGL_viewports_skin2::erase_viewports()
{
    for(unsigned i = 0; i < _viewports.size(); i++)
    {
        _viewports[i]->close();
        delete _viewports[i];
    }

    // We don't need to delete the frames because qt will do it when deleting
    // the main layout
    _viewports_frame.clear();
    _viewports.clear();
    delete _main_layout;
    _main_layout = 0;
}

void OGL_viewports_skin2::erase_cameraViewers()
{
	delete _cameraViewer;
}



// -----------------------------------------------------------------------------

PaintCanvas* OGL_viewports_skin2::new_viewport(Viewport_frame_skin2* ogl_frame)
{
    //PaintCanvas* ogl = new PaintCanvas(ogl_frame, _hidden, _main_window);
	QGLFormat format = QGLFormat::defaultFormat();
	format.setSampleBuffers(true);
	format.setSamples(8);
	PaintCanvas*   ogl =  new PaintCanvas(format,ogl_frame->id(),ogl_frame,_main_window);



    //QObject::connect(ogl, SIGNAL(drawing()), this, SLOT(incr_frame_count()));
    QObject::connect(ogl, SIGNAL( clicked() ), ogl_frame, SLOT( activate() ));
    ogl->setMinimumSize(4, 4);
    // initialize openGL and paint widget :
    ogl->updateGL();
    return ogl;
}

// -----------------------------------------------------------------------------

Viewport_frame_skin2* OGL_viewports_skin2::new_viewport_frame(QWidget* parent, int id)
{
    Viewport_frame_skin2* frame = new Viewport_frame_skin2(parent, id);
    QObject::connect(frame, SIGNAL( active(int) ),
                     this , SLOT  ( active_viewport_slot(int)) );

    return frame;
}

// -----------------------------------------------------------------------------

void OGL_viewports_skin2::set_frame_border_color(Viewport_frame_skin2* f, int r, int g, int b)
{
    QString str = "color: rgb("+
            QString::number(r)+", "+
            QString::number(g)+", "+
            QString::number(b)+");";

    f->setStyleSheet( str );
}

// -----------------------------------------------------------------------------

Vec_viewports& OGL_viewports_skin2::get_viewports()
{
    return _viewports;
}

// -----------------------------------------------------------------------------

void OGL_viewports_skin2::set_io(EOGL_widget::IO_t io_type)
{
	_io_type = io_type;
	bool state;
	switch(io_type)
	{
	case EOGL_widget::RBF:       state = true;  break;
	case EOGL_widget::DISABLE:   state = false; break;
	case EOGL_widget::GRAPH:     state = false; break;
	case EOGL_widget::SKELETON:  state = true;  break;
	case EOGL_widget::MESH_EDIT: state = true;  break;
	case EOGL_widget::BLOB:      state = false;  break;
	default:                    state = true;  break;
	}

	_skel_mode = _skel_mode || state;

	for(unsigned i = 0; i < _viewports.size(); i++){
		_viewports[i]->set_io(io_type);
		_viewports[i]->set_draw_skeleton(state);
	}
}

// -----------------------------------------------------------------------------

void OGL_viewports_skin2::set_gizmo(Gizmo::Gizmo_t type)
{
    for(unsigned i = 0; i < _viewports.size(); i++)
        _viewports[i]->set_gizmo(type);
}

// -----------------------------------------------------------------------------

void OGL_viewports_skin2::show_gizmo(bool state)
{
    for(unsigned i = 0; i < _viewports.size(); i++)
        _viewports[i]->_draw_gizmo = state;
}

// -----------------------------------------------------------------------------

//void OGL_viewports_skin2::set_cam_pivot(EOGL_widget::Pivot_t m)
//{
//	for(unsigned i = 0; i < _viewports.size(); i++)
//		_viewports[i]->set_cam_pivot_type(m);
//}

// -----------------------------------------------------------------------------

//void OGL_viewports_skin2::set_gizmo_pivot(EIO_Selection::Pivot_t piv)
//{
//	for(unsigned i = 0; i < _viewports.size(); i++)
//		_viewports[i]->set_gizmo_pivot(piv);
//}

// -----------------------------------------------------------------------------


//void OGL_viewports_skin2::set_gizmo_dir(EIO_Selection::Dir_t dir)
//{
//	for(unsigned i = 0; i < _viewports.size(); i++)
//		_viewports[i]->set_gizmo_dir( dir );
//}

// -----------------------------------------------------------------------------

void OGL_viewports_skin2::set_alpha_strength(float a)
{
    //for(unsigned i = 0; i < _viewports.size(); i++)
    //    _viewports[i]->set_alpha_strength( a );
}

// -----------------------------------------------------------------------------

void OGL_viewports_skin2::incr_frame_count()
{
    _frame_count++;
    emit frame_count_changed(_frame_count);
}

// -----------------------------------------------------------------------------

void OGL_viewports_skin2::active_viewport_slot(int id)
{
    for(unsigned i = 0; i < _viewports_frame.size(); i++)
        set_frame_border_color(_viewports_frame[i], 0, 0, 0);

    set_frame_border_color(_viewports_frame[id], 255, 0, 0);
    _current_viewport = _viewports[id];

    _current_viewport->setFocus();
    _current_viewport->makeCurrent();
//    if( prev_viewport() != _current_viewport)
//	{
		qglviewer::Camera* prevCamera = prev_viewport()->camera();
		QObject::disconnect( (const QObject*)prevCamera->frame(), SIGNAL(manipulated()), _cameraViewer, SLOT(updateGL()));
		QObject::disconnect((const QObject*)prevCamera->frame(), SIGNAL(manipulated()), _cameraViewer, SLOT(updateGL()));
		QObject::disconnect((const QObject*)prevCamera, SIGNAL(cameraChanged()), _cameraViewer, SLOT(updateGL()));
		_cameraViewer->setCamera( active_viewport()->camera());
//	}
	// Make sure every v camera movement updates the camera viewer

	QObject::connect((const QObject*)active_viewport()->camera()->frame(), SIGNAL(manipulated()), _cameraViewer, SLOT(updateGL()));
	QObject::connect((const QObject*)active_viewport()->camera()->frame(), SIGNAL(spun()), _cameraViewer, SLOT(updateGL()));
	// Also update on camera change (type or mode)
	QObject::connect( active_viewport(), SIGNAL(cameraChanged()), _cameraViewer, SLOT(updateGL()));
	_cameraViewer->setWindowTitle("Camera viewer: "+ QString::number(id));
	if(isCameraViewerOn)_cameraViewer->show();

	emit active_viewport_changed(id);
}

// -----------------------------------------------------------------------------

void OGL_viewports_skin2::first_viewport_as_active()
{
    assert(_viewports.size() > 0);
    _current_viewport = _viewports[0];
	_prev_viewport = _current_viewport;
    set_frame_border_color(_viewports_frame[0], 255, 0, 0);
    _current_viewport->setFocus();
    _current_viewport->makeCurrent();
	_cameraViewer->setCamera( _current_viewport->camera());
	active_viewport_slot(0);
}

void OGL_viewports_skin2::toggleCameraViewer()
{
	if(isCameraViewerOn)
	{
		_cameraViewer->show();
	}else
	{
		_cameraViewer->close();
	}
}

// -----------------------------------------------------------------------------
