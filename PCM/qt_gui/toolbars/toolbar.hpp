#ifndef TOOLBAR_HPP__
#define TOOLBAR_HPP__


#include <QToolBar>
#include <QComboBox>

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
class 	Widget_fitting;
class	Widget_viewports;
class	Widget_gizmo;
class	Widget_render_mode;
class	Widget_fitting;
class Widget_selection;


// -----------------------------------------------------------------------------

class Toolbar : public QToolBar {
    Q_OBJECT
public:

    Toolbar(QWidget* parent);

    Widget_selection*   _wgt_select;
    Widget_viewports*   _wgt_viewport;
    Widget_gizmo*       _wgt_gizmo;
    Widget_render_mode* _wgt_rd_mode;
    Widget_fitting*     _wgt_fit;
    /// ComboBox to choose the pivot mode of the camera
    QComboBox*          _pivot_comboBox;
    /// Pivot of the gizmo
    QComboBox*          _pivot_gizmo_comboBox;
    /// orientation of the gizmo
    QComboBox*          _dir_gizmo_comboBox;

public slots:

signals:

private:

};

#endif // TOOLBAR_HPP__
