#ifndef IO_MESH_EDIT_HPP__
#define IO_MESH_EDIT_HPP__

#include "IO_skeleton.hpp"
#include "toolbox/maths/vec2i.hpp"
class  PaintCanvas;

/**
 * @name IO_mesh_edit
 * @brief Handle mouse and keys for mesh editing (vertex selection/painting)
*/
class IO_mesh_edit : public IO_skeleton {
public:
    IO_mesh_edit(PaintCanvas* gl_widget);

    // -------------------------------------------------------------------------

    virtual ~IO_mesh_edit();

    // -------------------------------------------------------------------------

    virtual void mousePressEvent( QMouseEvent* event );

    // -------------------------------------------------------------------------

    virtual void mouseReleaseEvent( QMouseEvent* event );

    // -------------------------------------------------------------------------

    virtual void mouseMoveEvent( QMouseEvent* event );

    // -------------------------------------------------------------------------

    virtual void wheelEvent( QWheelEvent* event );

    // -------------------------------------------------------------------------

    virtual void keyPressEvent(QKeyEvent* event);

    // -------------------------------------------------------------------------

    virtual void keyReleaseEvent(QKeyEvent* event);

    // -------------------------------------------------------------------------

protected:

    bool is_paint_on();


    // -------------------------------------------------------------------------

    void paint(int x, int y);

    // -------------------------------------------------------------------------

    void select(float x, float y);

    // -------------------------------------------------------------------------

    void unselect(float x, float y);

    // -------------------------------------------------------------------------


    /*-----------*
    | Attributes |
    *-----------*/
    bool     _is_edit_on;
    Tbx::Vec2i _old_mouse;  ///< mouse position at last click
};

#endif // IO_MESH_EDIT_HPP__
