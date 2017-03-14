#ifndef IO_DISABLE_SKIN_HPP__
#define IO_DISABLE_SKIN_HPP__

#include "IO_interface_skin.hpp"
class PaintCanvas;
/** @brief Disables mouse and keys
  @see IO_interface
*/

class IO_disable_skin : public IO_interface_skin {
public:
    IO_disable_skin(PaintCanvas* gl_widget) : IO_interface_skin(gl_widget){ }

    void mousePressEvent  (QMouseEvent* e){ e->ignore(); }
    void mouseReleaseEvent(QMouseEvent* e){ e->ignore(); }
    void mouseMoveEvent   (QMouseEvent* e){ e->ignore(); }
    void wheelEvent       (QWheelEvent* e){ e->ignore(); }
    void keyPressEvent    (QKeyEvent*   e){ e->ignore(); }
    void keyReleaseEvent  (QKeyEvent*   e){ e->ignore(); }
};

#endif // IO_DISABLE_HPP__
