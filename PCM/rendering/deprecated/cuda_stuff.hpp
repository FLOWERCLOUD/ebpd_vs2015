#ifndef CUDA_STUFF_HPP_
#define CUDA_STUFF_HPP_

#include "camera.hpp"

// =============================================================================
namespace Raytracing {
// =============================================================================

/// raytrace the current implicit model
/// @param d_rendu Intermediate buffer used to raytrace
/// @param progressive Activate progressive raytracing : only some pixels are
/// computed at each call. The result is blurred
/// @return if the raytracing is complete (happens when progressive mode is
/// activated)
bool raytrace_implicit(const Camera& cam,
                       float4* d_rendu,
                       float* d_rendu_depth,
                       int* d_img_buf,
                       unsigned* d_depth_buf,
                       int width,
                       int height,
                       bool progressive
                       );

} // END RAYTRACING ============================================================

/// Draw the controller given the last selected bone.
/// Controllers values are directly fetched from texture via a cuda kernel
/// (x, y) the offset to draw the ctrl. (w, h) the width and height
/// @param inst_id instance of the controller
void draw_controller(int inst_id, int x, int y, int w, int h);


#endif // CUDA_STUFF_HPP_
