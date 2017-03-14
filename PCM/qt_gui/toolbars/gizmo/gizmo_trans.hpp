#ifndef GIZMO_TRANSLATION_HPP__
#define GIZMO_TRANSLATION_HPP__

#include "gizmo.hpp"
#include "toolbox/portable_includes/port_glew.h"
#include "toolbox/maths/vec3.hpp"
#include "toolbox/maths/vec2i.hpp"
#include "toolbox/gl_utils/glpick.hpp"

namespace Tbx
{
	class Camera;
}

/**
  @name Gizmo_trans
  @brief 3D frame to provide GUI for points translations.
  This class provide methods for translating objects in the GUI.
  It memories the current position and compute its new position
  given the mouse position and axis/plane constraint. It also gives means to
  draw a custom oriented frame which is represented by arrows. Each arrow can be
  selected.

  @see Gizmo
*/

class Gizmo_trans : public Gizmo {
public:

    enum Constraint_t {AXIS, PLANE};
    enum Axis_t       {X , Y , Z ,XY, XZ, YZ, NOAXIS ,NOPLANE  };
    //enum Plane_t      {XY, XZ, YZ, NOPLANE };

    Gizmo_trans(): Gizmo(),
        _size(10.f),
        _length(0.13f),
        _radius(0.30f/50.f),
        _ortho_factor(25.f),
        _constraint(AXIS),
        _selected_axis(NOAXIS),
        //_selected_plane(NOPLANE),
        _axis(Tbx::Vec3::unit_x())
    { }

    /// draw the current selected point and frame. Frame is represented with
    /// arrows (green, red and blue) made of a cylinder and a cone for the tip.
    /// Frame is drawn at the position specified by '_frame.get_translation()'.
    /// X, Y and Z arrows orientations are defined by 'set_frame()'.
    void draw(const Tbx::Camera& cam);

    /// select the x or y or z frame axis given a camera and a mouse position
    /// updates attributes (_axis or _plane) and _constraint
    /// @return true if one of the axis is selected
    bool select_constraint(const Tbx::Camera& cam, int px, int py);

    /// reset the selected constraint set by select_constraint()
    void reset_constraint(){
        _selected_axis  = NOAXIS;
        //_selected_plane = NOPLANE;
    }
	/// Sets starting position to compute the slide
	void slide_from( const Tbx::Transfo start_frame, const Tbx::Vec2i& start_pix){
		_start_frame = start_frame;
		_start_pix   = start_pix;
	}

    /// @brief slide point given the current constraint (plane or axis)
    /// If constraint has been selected this will move the selected point to
    /// its new position by following the constraint and keeping the frame
    /// as close as possible under the mouse position (px, py)
    /// @return the translation made by the frame in world coordinates
    /// (when clicked on)
    /// @see select_constraint()
    Tbx::TRS slide(const Tbx::Camera& cam, int px, int py);
	Tbx::TRS slide_plane(Tbx::Vec3 normal  , const Tbx::Camera& cam, Tbx::Vec2i pix);
    // TODO: slide along a plane parallel to the image plane

private:
    Tbx::TRS slide_axis (Tbx::Vec3 axis_dir, const Tbx::Camera& cam, int px, int py);
    Tbx::TRS slide_plane(Tbx::Vec3 normal  , const Tbx::Camera& cam, int px, int py);

	void draw_quads();

	inline bool is_plane_constraint()
	{
		return _selected_axis == XY || _selected_axis == XZ || _selected_axis == YZ;
	}


    Tbx::Vec3 _color;       ///< point color
    float    _size;       ///< point size when drawing

    /// Current frame in which the point is moved
    //@{
    float _length;         ///< length of the arrows representing the frame
    float _radius;         ///< radius of the arrows representing the frame
    float _ortho_factor;
    //@}

    Constraint_t _constraint;     ///< current constraint (axis or plane movement)
    Axis_t       _selected_axis;  ///< current selected axis
    //Tbx::Plane_t      _selected_plane; ///< current selected plane
    Pix          _pix_diff;       ///< pixels to substract when sliding
    Pix          _org;            ///< pixel clicked when a constraint is selected
    Tbx::Vec3      _axis;           ///< current axis direction for movements
    Tbx::Vec3      _plane;          ///< current plane normal for movements
	Tbx::GLPick _picker;
	Tbx::Transfo  _start_frame;
	Tbx::Vec2i _start_pix;

};

#endif // GIZMO_TRANSLATION_HPP__
