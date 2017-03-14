#ifndef IO_GRAPH_HPP__
#define IO_GRAPH_HPP__

#include "IO_interface_skin.hpp"
#include "toolbox/maths/vec3.hpp"
class  PaintCanvas;
/** @brief Handle mouse and keys for graph edition

  @see IO_interface
*/
class IO_graph : public IO_interface_skin {
public:
    IO_graph(PaintCanvas* gl_widget);

    // -------------------------------------------------------------------------

    virtual ~IO_graph();

    // -------------------------------------------------------------------------

    virtual void mousePressEvent( QMouseEvent* event );

    // -------------------------------------------------------------------------

    virtual void mouseReleaseEvent( QMouseEvent* event );

    // -------------------------------------------------------------------------

    virtual void mouseMoveEvent( QMouseEvent* event );

    // -------------------------------------------------------------------------

    virtual void keyPressEvent(QKeyEvent* event);

    // -------------------------------------------------------------------------

    virtual void wheelEvent( QWheelEvent* event );

    // -------------------------------------------------------------------------

private:
    int _moved_node;    ///< current graph node to be moved

    float _mouse_z;     ///<
    Tbx::Vec3 _cursor;  ///<


};

#endif // IO_GRAPH_HPP__
