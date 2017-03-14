#ifndef MARCHING_CUBES_HPP__
#define MARCHING_CUBES_HPP__

#include "scene_tree/implicit_surfaces/node_implicit_surface.hpp"
#include "marching_cubes_cpu.hpp"

/**
 * @namespace Marching_cubes
 * @brief Render scalar fields using marching cubes with a geometry shader
 *
 */
// =============================================================================
namespace Marching_cubes {
// =============================================================================

/// Init CPU and opengl buffers for rendering with marching cube
void init();
/// Clean various CPU OpenGl buffers
void clean();

/// Set the resolution of the 3d grid used to polygonise the scalar field.
/// Will reallocate the various buffers.
/// @warning 'res' needs to be divisible by four
void set_resolution( const Vec3i& res );

/// Given an implicit surface fill the 3d wich will be polygonised
/// by the marching cubes. Operation is done on CPU with multiple threads,
/// result is then uploaded on GPU.
void fill_3D_grid_with_scalar_field(const Node_implicit_surface* obj);

/// Polygonize the 3D grid with a geometry shader
void render_scalar_field();

/// Polygonize the 3D grid on CPU and draw with direct mode
/// @param res : alternate resolution if a component is negative we will use
/// the one defined by set_resolution();
void render_scalar_field_cpu(const Node_implicit_surface* obj, const Vec3i& res = Vec3i(-1) );

}// END MARCHING_CUBE ==========================================================

#endif // MARCHING_CUBES_HPP__
