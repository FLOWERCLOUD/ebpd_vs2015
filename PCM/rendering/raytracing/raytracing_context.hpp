#ifndef RAYTRACING_CONTEXT_HPP__
#define RAYTRACING_CONTEXT_HPP__

#include "packed_data_struct.hpp"
#include "toolbox/containers/material_cu.hpp"
#include "toolbox/maths/color.hpp"
#include "render_context.hpp"

// =============================================================================
namespace Raytracing {
// =============================================================================

struct Potential_colors{
    Color negative_color;   /// color for negative potential : f < 0
    Color extern_color;     /// color for external potential : 0 < f < 0.5
    Color intern_color;     /// color for internal potential : 0.5 < f < 1
    Color huge_color;       /// color for huge potential     : 1 < f
    Color zero_potential;   /// color for zero potential     : f ~= 0
    Color one_potential;    /// color for 1 potential        : f ~= 1
};

struct Context {
    /// Grid size to launch the raytracing kernel
    dim3 grid;
    /// Block size to launch the raytracing kernel
    dim3 block;

    /// Device buffer to store depth and color resulting from raytracing
    PBO_data pbo;
    /// Uniform material to use for the raytraced scene
    Material_cu mat;
    /// Camera settings for the raytracing
    Camera_data cam;
    /// Draw the 2D slice of the iso-surface
    bool potential_2d;
    /// Draw the scene
    bool draw_tree;

    /// Position of the 2D slice of the iso-surface
    /// @{
    Vec3 plane_n;
    Point3 plane_org;
    /// @}

    /// Enable disable lighting
    bool enable_lighting;
    /// Use environment map
    bool enable_env_map;
    /// Background color if no environment map
    Color background;
    /// defines which pixels are to be drawn in progressive mode
    int3 steps;


    /// Length of a step for the ray marching
    float step_len;
    /// Maximum number of ray steps for ray marching
    int nb_ray_steps;
    /// Maximum level of reflexion rays (zero for no reflexions)
    int nb_reflexion;

    /// When progressive is true this means raytracing will be done with
    /// multiple passes. Only a fixed amount of pixels will be traced.
    /// the result is then blured to give the user a preview of the scene.
    /// The number of pixels traced equal to max_res_nth_pass^2
    bool progressive;

    /// maximum number of pixels traced in x & y direction when 'progressive'
    /// is true
    int max_res_nth_pass;

    Potential_colors potential_colors;

    Render_context* render_ctx;

    Context() : render_ctx(0) { }

};

}// END RAYTRACING =============================================================

#endif// RAYTRACING_CONTEXT_HPP__
