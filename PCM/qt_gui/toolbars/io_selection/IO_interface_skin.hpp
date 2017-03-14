#ifndef IO_INTERFACE_SKIN_HPP__
#define IO_INTERFACE_SKIN_HPP__

/** @brief Abstract class handling mouse and keyboards events

    This class is design to provide dynamic behavior of mouse and keys.
    You can find specialised behaviors in IO_xxx class family.
*/

#include <QMouseEvent>
#include "toolbox/portable_includes/port_glew.h"
#include "toolbox/maths/vec2.hpp"
#include "toolbox/maths/vec3.hpp"
#include "toolbox/maths/transfo.hpp"
#include "toolbox/maths/trs.hpp"

#ifndef M_PI
#define M_PI (3.14159265358979323846f)
#endif

/////////////////////////////////////////////////////////////
// TODO: delete all this and provide proper header interface via cuda_ctrl

// From cuda_globals.hpp -------



class PaintCanvas;
class Kinematic;
class main_window;
class Gizmo;
namespace Tbx
{
	class Camera;
}
/////////////////////////////////////////////////////////////

class IO_interface_skin {
public:
    IO_interface_skin(PaintCanvas* gl_widget);

    virtual ~IO_interface_skin();

    // -------------------------------------------------------------------------

    void update_gl_matrix();

    // -------------------------------------------------------------------------

    /// Draw a message on the viewport
    void push_msge(const QString& str);

    // -------------------------------------------------------------------------

    virtual void mousePressEvent(QMouseEvent* event);

    // -------------------------------------------------------------------------

    virtual void mouseReleaseEvent(QMouseEvent* event);

    // -------------------------------------------------------------------------

    /// Camera rotation
    virtual void mouseMoveEvent(QMouseEvent* event);

    // -------------------------------------------------------------------------

    /// Camera movements back and forth
    virtual void wheelEvent(QWheelEvent* event);

    // -------------------------------------------------------------------------

    void right_view();

    void left_view();

    void top_view();

    void bottom_view();

    void front_view();

    void rear_view();

    // -------------------------------------------------------------------------

    Tbx::Vec2 increase(Tbx::Vec2 val, float incr, bool t);

    // -------------------------------------------------------------------------

    virtual void keyPressEvent(QKeyEvent* event);

    // -------------------------------------------------------------------------

    virtual void keyReleaseEvent(QKeyEvent* event);

    /// Update the gizmo orientation this is to be called by OGL_widget
    /// every frame
    virtual void update_frame_gizmo();

    /// Shortcut function to get the gizmo
    Gizmo* gizmo();

    /// Shortcut to get the skeleton's kinematic
    Kinematic* kinec();


    // -------------------------------------------------------------------------
    /// @name Attributes
    // -------------------------------------------------------------------------
    float _old_x;           ///< last mouse click in pixel x
    float _old_y;           ///< last mouse click in pixel y

    float _cam_old_x;
    float _cam_old_y;

    bool _is_ctrl_pushed;   ///< is control key pushed
    bool _is_alt_pushed;    ///< is alt key pushed
    bool _is_tab_pushed;    ///< is tabulation key pushed
    bool _is_maj_pushed;    ///< is shift key pushed
    bool _is_space_pushed;  ///< is space key pushed
    bool _is_right_pushed;  ///< is mouse right button pushed
    bool _is_left_pushed;   ///< is mouse left button pushed
    bool _is_mid_pushed;    ///< is mouse middle
    bool _is_gizmo_grabed;  ///< is a gizmo constraint selected

    float _movement_speed;  ///< speed of the camera movements
    float _rot_speed;       ///< rotation speed of the camera

    /// @name Opengl matrix
    /// which state matches the last call of update_gl_matrix()
    /// @{
    GLint    _viewport  [4];
    GLdouble _modelview [16];
    GLdouble _projection[16];
    /// @}

    /// Gizmo transformation when grabed
    Tbx::TRS _gizmo_tr;

    PaintCanvas*  _gl_widget;
    Tbx::Camera*           _cam;
    main_window* _main_win;
};

#endif // IO_INTERFACE_HPP__
