#ifndef GIZMO_TRACKBALL_HPP__
#define GIZMO_TRACKBALL_HPP__

#include "gizmo.hpp"
#include "toolbox/maths/vec3.hpp"
#include "toolbox/maths/transfo.hpp"
#include "toolbox/gl_utils/glpick.hpp"

namespace Tbx
{
	class Camera;
}
/**
  @class Gizmo_trackball
  @brief Specialization of the gizmo to handle trackballs movements

  A trackball enables the user to grab anywhere on the screen a virtual ball
  and move it according to the dragged point. usually more intuitive to achieve
  complex rotations.

  @see Gizmo
*/
class Gizmo_trackball : public Gizmo {
public:

    enum Axis_t {BALL,
                 NOAXIS
                };

    Gizmo_trackball();

    void draw(const Tbx::Camera& cam);

    /// @note always returns true (except when show == false).
    /// Call is needed for the gizmo to compute the rotation
    bool select_constraint(const Tbx::Camera& cam, int px, int py);

    void reset_constraint(){ _selected_axis = NOAXIS; }

    Tbx::TRS slide(const Tbx::Camera& cam, int px, int py);

private:
    // -------------------------------------------------------------------------
    /// @name Tools
    // -------------------------------------------------------------------------
    Tbx::TRS trackball_rotation(const Tbx::Camera& cam, int px, int py);

    // -------------------------------------------------------------------------
    /// @name Attributes
    // -------------------------------------------------------------------------
    float _rad_sphere;    ///< Radius of the sphere representing the trackball
    float _rad_handles;   ///< Radius of the tore reprensenting the handles
    float _ortho_factor;

    /// pixel clicked when during constraint selection
    Pix _clicked;

    /// current selected axis
    Axis_t _selected_axis;

    /// World coordinates of the picked point
    Tbx::Vec3 _picked_pos;
};

#endif // GIZMO_TRACKBALL_HPP__
