#ifndef GIZMO_ROT_HPP__
#define GIZMO_ROT_HPP__

#include "gizmo.hpp"
#include "toolbox/maths/vec3.hpp"
#include "toolbox/maths/transfo.hpp"

#include "toolbox/gl_utils/glpick.hpp"
#include "toolbox/maths/color.hpp"

namespace Tbx
{
	class Camera;
}

class Gizmo_rot : public Gizmo {
public:

    enum Axis_t {X = 0,    ///< Rotation around X axis
                 Y,        ///< Rotation around Y axis
                 Z,        ///< Rotation around X axis
                 CAM,      ///< Rotation around the camera view dir axis
                 NOAXIS    ///< no axis constraint selected
               };

    Gizmo_rot();

    void draw(const Tbx::Camera& cam);

    bool select_constraint(const Tbx::Camera& cam, int px, int py);

    void reset_constraint(){ _selected_axis = NOAXIS; }

    Tbx::TRS slide(const Tbx::Camera& cam, int px, int py);

private:
    /*------*
    | Tools |
    *------*/
    Tbx::TRS axis_rotation(const Tbx::Camera& cam, int px, int py);

    /// Draw the tangent of the selected circle
    void draw_tangent(const Tbx::Vec3& cam_dir, float dist_cam);

    /// Draw the arc circle corresponding to the axis 'axis'
    void draw_arc_circle(Axis_t axis,
                         const Tbx::Vec3& a,
                         const Tbx::Vec3& cam_dir,
                         float scale,
                         const Tbx::Color& cl);

    void draw_circle(Axis_t axis,
                     const Tbx::Vec3& cam_dir,
                     float scale,
                     const Tbx::Color& cl);

    /*-----------*
    | Attributes |
    *-----------*/

    bool _enable_trackball;

    float _rad_sphere;    ///< Radius of the sphere representing the trackball
    float _rad_handles;   ///< Radius of the tore reprensenting the handles
    float _ortho_factor;

    Tbx::Vec3 _axis;         ///< current axis direction for movements
    Axis_t _selected_axis; ///< current selected axis

public:
    /// pixel clicked when during constraint selection
    Pix _clicked;
private:

    /// World coordinates of the picked point
    Tbx::Vec3 _picked_pos;
    /// Tangent of the selected circle
    Tbx::Vec3 _picked_tangent;


};

#endif // GIZMO_ROT_HPP__
