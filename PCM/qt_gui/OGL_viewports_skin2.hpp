#ifndef OGL_VIEWPORTS_SKIN2_HPP__
#define OGL_VIEWPORTS_SKIN2_HPP__

#include "toolbars/gizmo/gizmo.hpp"
#include "qt_gui/toolbars/OGL_widget_enum.hpp"
#include "GlobalObject.h"
#include <QFrame>
#include <vector>
#include <QLayout>
#include <QGLWidget>
#include <QTime>
class main_window;
class PaintCanvas;
class CameraViewer;


class Viewport_frame_skin2;

typedef  std::vector<PaintCanvas*>  Vec_viewports;
typedef  std::vector<CameraViewer*> Vec_cameraViewers;

// =============================================================================
class OGL_widget_skin2_hidden : public QGLWidget {
public:
    OGL_widget_skin2_hidden(QWidget* w) : QGLWidget(w) {}

    void initializeGL();
};
// =============================================================================

/** @class OGL_viewports
  @brief Multiple viewports handling with different layouts

*/
class OGL_viewports_skin2 : public QFrame {
    Q_OBJECT
public:
    OGL_viewports_skin2(QWidget* w, main_window* m);
	~OGL_viewports_skin2();
    /// Updates all viewports
    void updateGL();

    // -------------------------------------------------------------------------
    /// @name Getter & Setters
    // -------------------------------------------------------------------------
    enum Layout_e { SINGLE, VDOUBLE, HDOUBLE, FOUR };

    /// Erase all viewports and rebuild them according to the specified layout
    /// 'setting'
    void set_viewports_layout(Layout_e setting);

    /// @warning the list is undefined if used after a call to
    /// set_viewports_layout()
    Vec_viewports& get_viewports();

    // TODO: to be deleted
    /// sets io for all viewports
    void set_io(EOGL_widget::IO_t io_type);

    /// sets gizmo type (rotation, scaling, translation) for all viewports
    void set_gizmo(Gizmo::Gizmo_t type);

    /// Show/hide gizmo for all viewports
    void show_gizmo(bool state);

    /// sets camera pivot for all viewports
//    void set_cam_pivot(EOGL_widget::Pivot_t m);

    /// sets gizmo pivot for all viewports
//    void set_gizmo_pivot(EIO_Selection::Pivot_t piv);

    /// sets gizmo orientation for all viewports
//    void set_gizmo_dir(EIO_Selection::Dir_t dir);

    void set_alpha_strength(float a);

    /// @return the active frame
    PaintCanvas* active_viewport(){

		setGlobalCanvas(_current_viewport);
		return _current_viewport; 
	}
	PaintCanvas* prev_viewport()
	{
		return _prev_viewport;
	}

    QGLWidget* shared_viewport(){ return _hidden; }

	void toggleCameraViewer();
    // -------------------------------------------------------------------------
    /// @name Events
    // -------------------------------------------------------------------------
    void enterEvent( QEvent* e);

    // -------------------------------------------------------------------------
    /// @name Qt Signals & Slots
    // -------------------------------------------------------------------------
private slots:
    /// Designed to be called each time a single viewport draws one frame.
    void incr_frame_count();
    void active_viewport_slot(int id);

signals:
    void frame_count_changed(int);
    void active_viewport_changed(int id);
    /// Update status bar
    void update_status(QString);

private:
    // -------------------------------------------------------------------------
    /// @name Tools
    // -------------------------------------------------------------------------
    QLayout* gen_single ();
    QLayout* gen_vdouble();
    QLayout* gen_hdouble();
    QLayout* gen_four   ();

    /// suppress all viewports and layouts
    void erase_viewports();
	void erase_cameraViewers();

    /// Creates a new viewport with the correct signals slots connections
    PaintCanvas* new_viewport(Viewport_frame_skin2* ogl_frame);

    /// Creates a new viewport frame with the correct signals slots connections
    Viewport_frame_skin2* new_viewport_frame(QWidget* parent, int id);

    /// Sets the frame color by replacing its styleSheet color
    void set_frame_border_color(Viewport_frame_skin2* f, int r, int g, int b);

    void first_viewport_as_active();

    /// Before drawing compute every pre process like mesh deformation
    void update_scene();

    // -------------------------------------------------------------------------
    /// @name Attributes
    // -------------------------------------------------------------------------
    bool _skel_mode;

   EOGL_widget::IO_t _io_type;

    /// Vector of OGL_widget
    Vec_viewports		_viewports;
	CameraViewer*	_cameraViewer;

    /// List of frames associated to the viewports
    std::vector<Viewport_frame_skin2*> _viewports_frame;

    /// The active viewport
    PaintCanvas* _current_viewport;
	PaintCanvas* _prev_viewport;

    /// opengl shared context between all viewports
    /// (in order to share VBO textures etc.)
    OGL_widget_skin2_hidden* _hidden;

    /// Layout containing all viewports
    QLayout* _main_layout;

    /// main widow the widget's belongs to
    main_window* _main_window;

    /// sum of frames drawn by the viewports
    int _frame_count;

    /// Fps counting timer
    QTime  _fps_timer;
};
// =============================================================================

class Viewport_frame_skin2 : public QFrame {
    Q_OBJECT
public:

    Viewport_frame_skin2(QWidget* w, int id) : QFrame(w), _id(id)
    {
        setFrameShape(QFrame::Box);
        setFrameShadow(QFrame::Plain);
        setLineWidth(1);
        setStyleSheet(QString::fromUtf8("color: rgb(0, 0, 0);"));
    }


    int id() const { return _id; }

signals:
    void active(int);

private slots:
    void activate(){
        emit active(_id);
    }

private:
    int _id;
};
// =============================================================================

#endif // OGL_VIEWPORTS_SKIN2_HPP__
