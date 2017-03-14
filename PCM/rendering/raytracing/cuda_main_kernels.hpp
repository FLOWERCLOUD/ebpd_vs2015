#ifndef CUDA_MAIN_KERNELS_HPP__
#define CUDA_MAIN_KERNELS_HPP__

// -----------------------------------------------------------------------------

#include <vector>

// -----------------------------------------------------------------------------

#include "toolbox/maths/bbox3.hpp"
#include "raytracing_context.hpp"
#include "toolbox/cuda_utils/cuda_utils.hpp"
#include "blending_env_type.hpp"

// -----------------------------------------------------------------------------

/**
 * @file cuda_main_kernels.cu
 * @brief holds the main cuda kernels of the project with access to the main
 * cuda textures in the project
 *
 * In order to get maximum performances we use cuda textures which have the
 * advantage of using the GPU cache. However there is a catch.
 * A texture reference must be a global and its visibility in the project is
 * limited to its translation unit (i.e: textures ref can't be used outside
 * the'.cu' file it is declared within).
 *
 * you can declare texture references with the same name in different '.cu'
 * it won't change the fact you need to bind <b>each</b> texture
 * to a cuda array (even if it's the same array).
 *
 * We use a lot of textures and want to avoid binding/unbinding them each time we
 * call a kernel. To this end we define a single file 'cuda_main_kernels.cu'
 * where textures will be binded to their cuda arrays once and for all
 * at start up. Only critical kernels called very often needs to be in this file.
 * Other kernels can be defined in others '.cu' but must bind/unbind the texture
 * at each kernel call.
*/

// -----------------------------------------------------------------------------

/// Initialize device memory and textures
/// @warning must be called first before any other cuda calls
void init_cuda(const std::vector<Blending_env::Op_t>& op);

/**
 * @namespace Raytracing
 * @brief functions related to raytracing and cuda kernels associated
*/
// =============================================================================
namespace Raytracing {
// =============================================================================

/// Do we partially evaluate the tree ?
/// @see set_bone_to_trace()
void set_partial_tree_eval(bool s);

/// Sets the eight first bone to be raytraced from the vector
void set_bone_to_trace(const std::vector<int>& bone_ids, int id);

void set_curr_skel_id(int id);

void trace( Context& ctx );

void trace_skinning_skeleton(const Context& ctx, int skel_id);

}// End Raytracing namespace ===================================================


/// Fetch a controller value from texture given its instance id.
__global__
void get_controller_values(Cuda_utils::Device::Array<float2> out_vals, int inst_id);

// =============================================================================



#endif // CUDA_MAIN_KERNELS_HPP__
