#ifndef IO_SELECTION_ENUM_HPP__
#define IO_SELECTION_ENUM_HPP__

#include "toolbox/maths/color.hpp"
using namespace Tbx;
/// @name EIO_Selection
/// @brief enum used primarily with IO_selection class
/// @see IO_Selection Gizmo
// =============================================================================
namespace EIO_Selection {
// =============================================================================

/// Gizmo pivot point type
enum Pivot_t { MEDIAN,    ///< On median point of the current selection
               ACTIVE,    ///< Origin of the active object/data
               CURSOR_3D  ///< 3d cursor
               // TODO: individual objects centers
             };


/// Gizmo orientation
enum Dir_t {
    GLOBAL,   ///< aligned with world axis
    LOCAL,    ///< Object local space
    NORMAL,   ///< Object/data normal aligned
    VIEW      ///< Aligned with camera view
};

/// Transformation type mode
enum Transfo_t {
    TRANSLATION,
    ROTATION,
    SCALE,
    NONE         ///< no manipulation activated
};

/// Axis type (which axis or pair of axis are currently activated)
/// 'G' stands for global coordinates and 'L' for local coordinates
enum Axis_t {
    GX=0, LX=1,
    GY=2, LY=3,
    GZ=4, LZ=5,
    VIEW_PLANE = 6 ///< parralel to the image plane
};

/// @return true if 'a' is a global axis
static inline
bool is_global( Axis_t a ){ return (a == GX) || (a == GY) || (a == GZ); }

/// @return true if 'a' is a local axis
static inline
bool is_local ( Axis_t a ){ return (a == LX) || (a == LY) || (a == LZ); }

/// @return associated color to axis 'a'
static inline
Color get_axis_color( EIO_Selection::Axis_t a)
{
    Color c;
    switch(a){
    case LX: case GX:  c = Color(0.82f, 0.40f, 0.40f); break;
    case LY: case GY:  c = Color(0.40f, 0.82f, 0.40f); break;
    case LZ: case GZ:  c = Color(0.40f, 0.40f, 0.82f); break;
    default: break;
    }

    return c;
}

}// END EIO_SELCTION ===========================================================

#endif // IO_SELECTION_ENUM_HPP__
