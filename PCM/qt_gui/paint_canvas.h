#ifndef _PAINT_CANVAS_H
#define _PAINT_CANVAS_H
#include "basic_types.h"
#include "toolbox/gl_utils/glbuffer_object.hpp"
#include "QGLViewer/qglviewer.h"
#include "toolbars/gizmo/gizmo.hpp"
#include "toolbars/OGL_widget_enum.hpp"
#include "toolbox/maths/selection_heuristic.hpp"
#include "toolbox/maths/vec3.hpp"

//#include "select_tool.h"


class SnapshotSetting;
class SavePlySetting;
class main_window;
class Tool;

class BulletOpenGLApplication;
class GLInstancingRenderer;
class Render_context;
class Manipulator;
class IO_interface_skin;
class Particle
{
public :
	Particle()
	{
		init();
	}

	void init()
	{
		pos_ = qglviewer::Vec(0.0, 0.0, 0.0);
		float angle = 2.0 * M_PI * rand() / RAND_MAX;
		float norm  = 0.04 * rand() / RAND_MAX;
		speed_ = qglviewer::Vec(norm*cos(angle), norm*sin(angle), rand() / static_cast<float>(RAND_MAX) );
		age_ = 0;
		ageMax_ = 50 + static_cast<int>(100.0 * rand() / RAND_MAX);
	}
	void draw()
	{
		glColor3f(age_/(float)ageMax_, age_/(float)ageMax_, 1.0);
		glVertex3fv(pos_);
	}
	void animate()
	{
		speed_.z -= 0.05f;
		pos_ += 0.1f * speed_;

		if (pos_.z < 0.0)
		{
			speed_.z = -0.8*speed_.z;
			pos_.z = 0.0;
		}

		if (++age_ == ageMax_)
			init();
	}

private :
	qglviewer::Vec speed_, pos_;
	int age_, ageMax_;
};

class Peeler;

class PaintCanvas: public QGLViewer
{
	Q_OBJECT
public:
	PaintCanvas(const QGLFormat& format, QWidget *parent);
	PaintCanvas(const QGLFormat& format, int type, QWidget *parent,QWidget * mwindow,const QGLWidget* shareWidget = NULL);
	~PaintCanvas();
public:
	std::string title() const { return "[PaintCanvas]:"; }

	void forceUpdate(){};
	void updateCanvas(){};

	void setTracerShowOrNot( bool b ){ show_trajectory_=b; }
	void setGraph_WrapBoxShowOrNot(bool b){show_Graph_WrapBox_ = b;}
	void setEdgeVertexsShowOrNot(bool b){ show_EdgeVertexs_ = b;}

	void showSelectedTraj();
	//save screeshot
	void saveSnapshot();
	void pasteTile();
	void setView();
	void setTiledView(GLdouble fovY , float viewRatio , float fAspect , GLdouble zNear ,GLdouble zFar , float cameraDist);
	inline bool istakeSnapTile()
	{
		return takeSnapTile;
	}
	void drawCornerAxis();
	SnapshotSetting* ss;
	ScalarType fov;
	ScalarType clipRatioFar;
	ScalarType clipRatioNear;
	ScalarType nearPlane;
	ScalarType farPlane;
	void Logf(int level ,const char* f );
	//save ply
	SavePlySetting* splys;
	void savePLY(SavePlySetting& ss);
	void saveLabelFile(std::string filename ,IndexType selected_frame_idx = 0);
	void getLabelFromFile(std::string filename,IndexType selected_frame_idx = 0);
	//show trajectory
	void showSelectedlabelTraj(std::vector<int>& _selectedlabeltraj);
	void showSelectedFrameLabel(std::vector<int>& showed_label,int curSelectedFrame);

	void BulletOpenGLApplicationCallBack( BulletOpenGLApplication* bc);
	
protected:
	virtual void draw();
	virtual void fastDraw();
	virtual void init();
	virtual void animate();
	virtual QString helpString() const;
	

	// Mouse events functions
	virtual void mousePressEvent(QMouseEvent *e);
	virtual void mouseMoveEvent(QMouseEvent *e);
	virtual void mouseReleaseEvent(QMouseEvent *e);
	virtual void keyPressEvent(QKeyEvent * e);
	virtual void keyReleaseEvent(QKeyEvent * e);
	virtual void resizeEvent(QResizeEvent* e);
	virtual void enterEvent( QEvent* e);
	virtual void leaveEvent( QEvent* e);
    bool _is_mouse_in;
	// wheel event
	virtual void wheelEvent(QWheelEvent *e);
	void saveSnapshotImp(SnapshotSetting& _ss);
	void setTileView( IndexType totalCols , IndexType totalRows ,IndexType tileCol ,IndexType tileRow );

signals:
	// -------------------------------------------------------------------------
	/// @name Signals
	// -------------------------------------------------------------------------

	/// Emited on mouse press events
	void clicked();
	/// Emited for each frame (only if (g_save_anim || g_shooting_state))
	void drawing();
	/// Emited when a new active object is selected
	//void selected(Obj*);
	void cameraChanged();
private:
	int				coord_system_region_size_;
	main_window*	main_window_;
	//screeshot
	QImage snapBuffer;
	bool  takeSnapTile;
	IndexType tileCol ,tileRow , totalCols ,totalRows;
	IndexType currSnapLayer;  // snapshot; total number of layers and current layer rendered
	bool is_key_l_pressed;
	std::vector<int> showed_label;

public:
	

	Tool*	single_operate_tool_;

	bool			show_trajectory_;
	//added by huayun
	bool			show_Graph_WrapBox_;
	bool			show_EdgeVertexs_;
	IndexType		centerframeNum;
	GLInstancingRenderer* m_instancingRenderer;
private:
	BulletOpenGLApplication* pApp ; //enable some callback;
	
	int nbPart_;
	Particle* particle_;
public:
	void update_pivot();
	Tbx::Vec3 pivot() const { return _pivot; }

	EOGL_widget::Pivot_t pivot_mode() const { return _pivot_mode; }

	void set_pivot_user(const Tbx::Vec3& v){ _pivot_user = v; }

	void set_pivot_mode(EOGL_widget::Pivot_t m){ _pivot_mode = m; }
private:
	/// Pivot point position. Used by the camera to rotate around it.
	Tbx::Vec3 _pivot;

	/// Pivot point defined by the user
	Tbx::Vec3 _pivot_user;

	/// mode of rotation: defines what's the standard behavior to compute
	/// automatically the pivot point
	EOGL_widget::Pivot_t _pivot_mode;

public:

	/// set the selection mode
	void set_selection(EOGL_widget::Select_t select_mode);
//	void set_selected(Obj* o);

	const Gizmo* gizmo() const { return _gizmo; }
	Gizmo* gizmo()       { return _gizmo; }

	/// Choose the type of gizmo (rotation translation scale). Origin and
	/// orientation are kept fropm the old gizmo
	void set_gizmo(Gizmo::Gizmo_t type);

	void set_io(EOGL_widget::IO_t io_type);
	/// wether we draw and use the for objects movements
	bool _draw_gizmo;
	Tbx::Select_type<int>* get_heuristic(){ return _heuristic; }
private:
	Gizmo* _gizmo;
	/// Current heuristic for mesh's points selection. Which defines the
	/// selection area (point, square, circle etc.) used to select the mesh's
	/// points.
	//Select_type<int>* _heuristic;
	Tbx::Select_type<int>* _heuristic;
	//Peeler* m_peeler;
	//Tbx::GlBuffer_obj<GLuint>* m_pbo_depth;
	//Tbx::GlBuffer_obj<GLint>* m_pbo_color;

	/// Context for the rendering in paint canvas
	Render_context*	m_render_ctx; 

	/// handle mouse and keyboards according to the desire modeling mode
	Manipulator* _io;
	IO_interface_skin* m_io;
public:
	// -------------------------------------------------------------------------
	/// @name Getter & Setters
	// -------------------------------------------------------------------------

	/// Draw the skeleton or the graph
	void set_draw_skeleton(bool s);

	/// Enable/disable phong rendering
	void set_phong(bool s);

	/// Enable/disable textures in phong rendering
	void set_textures(bool s);

	/// Enable/disable raytracing
	void set_raytracing(bool s);

	/// Draw the mesh in rest position or in animated position
	void set_rest_pose(bool s);

	bool rest_pose();

	void set_draw_mesh(bool s);

	bool draw_mesh();

	/// @return transclucent or plain phong rendering ?
	bool phong_rendering() const;

	/// @return true if raytracing enable
	bool raytrace() const;

	virtual void beginSelection(const QPoint& point)
	{
		QGLViewer::beginSelection(point);
	}
	virtual void drawWithNames();
	virtual void endSelection(const QPoint& point)
	{
		QGLViewer::endSelection(point);
	}
	virtual void postSelection(const QPoint& point);


};

#endif