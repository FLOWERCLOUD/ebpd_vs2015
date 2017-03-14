#ifndef G_VBO_PRIMITIVES_HPP__
#define G_VBO_PRIMITIVES_HPP__

/** @file g_vbo_primitives.hpp
    @brief export global variables related to vbos primitives
    @see globals.cpp
*/

#include "toolbox/gl_utils/vbo_primitives.hpp"

/// Some basic primitive are available, they are upload into the GPU
/// at the initialization of opengl
/// @see init_opengl()
/// @note 'lr' stands for low res
extern Tbx::VBO_primitives g_primitive_printer;
extern Tbx::Prim_id g_sphere_lr_vbo;
extern Tbx::Prim_id g_sphere_vbo;
extern Tbx::Prim_id g_circle_vbo;
extern Tbx::Prim_id g_arc_circle_vbo;
extern Tbx::Prim_id g_circle_lr_vbo;
extern Tbx::Prim_id g_arc_circle_lr_vbo;
extern Tbx::Prim_id g_grid_vbo;
extern Tbx::Prim_id g_cylinder_vbo;
extern Tbx::Prim_id g_cylinder_cage_vbo;
extern Tbx::Prim_id g_cube_vbo;

#endif // G_VBO_PRIMITIVES_HPP__
