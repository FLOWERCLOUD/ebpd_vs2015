#ifndef CLICKABLE_FRAME_HPP__
#define CLICKABLE_FRAME_HPP__

#include <QFrame>
#include <QMouseEvent>

/** @class Clickable_frame
    @brief A QFrame which throws a signal when left clicked on.
*/

class Clickable_frame : public QFrame{
    Q_OBJECT
public:

    Clickable_frame(QWidget* parent = 0) : QFrame(parent) {}

    void mousePressEvent( QMouseEvent* event ){
        if(event->button() == Qt::LeftButton)
        {
            emit leftclick_on();
        }
    }

    /// Set the background of the frame
    /// @warning The method erased the previous QFrame style sheet
    void set_background(const QColor& cl){
        setStyleSheet("background-color: rgb("+
                      QString::number(cl.red())+", "+
                      QString::number(cl.green())+", "+
                      QString::number(cl.blue())+");");
    }

signals:
    void leftclick_on();
};

#endif // CLICKABLE_FRAME_HPP__
