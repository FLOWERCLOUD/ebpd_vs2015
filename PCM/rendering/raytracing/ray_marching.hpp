#ifndef RAY_MARCHING_HPP__
#define RAY_MARCHING_HPP__

#include "cuda_ctrl.hpp"
#include "object.hpp"

#include "filters.hpp"
#include  "raytracing_context.hpp"

class Obj_implicit_tree;

// =============================================================================
namespace Raytracing {
// =============================================================================

/// Get/Set the current raytracing context. (you can define background color
/// enable/disable lighting etc.)
/// @see Context
Context& context();

/// next call to 'trace_tree()' will fill the image buffers from the beginning.
void reset_buffers();

/// Raytrace some pixels of the implicit scene. and draw it with openGL onto
/// a quad. One
///
/// @return false if the raytracing is complete.
/// @note This functions needs to be called several times in order to fill
/// the buffers and then do nothing. to raytrace from the beginning again call
/// 'reset_buffers()'
bool trace_tree(const Obj_implicit_tree* tree);

/// Clear raytracing render buffer.
/// @note buffer are situated in Context.render_ctx.pbo_color() and
/// Context.render_ctx->pbo_depth()
void clear_buffers();

}// END RAYTRACING =============================================================


#endif // RAY_MARCHING_HPP__
