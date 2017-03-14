#ifndef MARCHING_CUBES_CPU_HPP__
#define MARCHING_CUBES_CPU_HPP__

#include "scene_tree/implicit_surfaces/node_implicit_surface.hpp"
#include "toolbox/maths/vec3.hpp"
#include "toolbox/maths/vec3i.hpp"

// =============================================================================
namespace Marching_cubes {
// =============================================================================

class Cell{
public:
    float   val[8]; ///< value at each cube's corner
    Vec3 pos[8]; ///< cube positions
};

/// Software marching cubes polygonization
/// Slow as hell because we use gl direct mode in addition of
/// the CPU calculation...
void direct_mode_render_marching_cubes(const Node_implicit_surface* node,
                                       const Vec3 world_start,
                                       Vec3i res,
                                       Vec3 steps,
                                       float iso_lvl);

}// END MARCHING_CUBE ==========================================================

#endif // MARCHING_CUBES_CPU_HPP__
