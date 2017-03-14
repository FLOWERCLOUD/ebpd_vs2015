#ifndef IO_SKELETON_HPP__
#define IO_SKELETON_HPP__

#include "IO_interface_skin.hpp"
#include "toolbox/maths/vec2.hpp"
#include "../gizmo/gizmo_rot.hpp"

#include <QMenu>
#include <map>

class PaintCanvas;

/** @brief Handle mouse and keys for skeleton animation

  @see IO_interface
*/

class IO_skeleton : public IO_interface_skin {
public:

    IO_skeleton(PaintCanvas* gl_widget);

    // -------------------------------------------------------------------------

    virtual ~IO_skeleton(){
    }

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

    void trigger_menu();

    // -------------------------------------------------------------------------

    virtual void keyReleaseEvent(QKeyEvent* event);

    // -------------------------------------------------------------------------

    virtual void update_frame_gizmo();
	Tbx::Transfo global_transfo(const Tbx::TRS& gizmo_tr);

    // -------------------------------------------------------------------------

    bool _janim_on;

private:

    // -------------------------------------------------------------------------
    /// @name Tools
    // -------------------------------------------------------------------------
    void update_moved_vertex();

    // -------------------------------------------------------------------------
    /// @name Attributes
    // -------------------------------------------------------------------------

    float _mouse_z;          ///< z depth when "_moved_node" has been clicked

    int   _joint;            ///< last joint id being selected
    // TODO: checke this attribute it might be redundant with _joint
    int   _last_vertex;      ///< joint being selected

    int   _nb_selected_vert; ///< current number of skeleton's joints selected
    bool  _move_joint_mode;  ///< enable user to move the skeleton joints in rest pose

    /// joint local transformation (according to the parent)
    /// when clicking on the gizmo
    Tbx::Transfo _curr_joint_lcl;

    /// joint global position when clicking on the gizmo
    Tbx::Vec3 _curr_joint_org;

    float _selection_hysteresis;

    QMenu* _menu;            ///< a contextual menu triggered with space key

    /// Map of the textual entries of menu to the enum field in Bone_type
    /// @see Bone_type
    std::map<std::string, int> _map_menutext_enum;
};

#endif // IO_SKELETON_HPP__
