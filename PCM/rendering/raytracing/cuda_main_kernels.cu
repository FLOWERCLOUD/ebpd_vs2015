#include "cuda_main_kernels.hpp"


/// @name Cuda textures importation
/// @{
#include "blending_env_tex.hpp"
#include "blending_env_tex_binding.hpp"

#include "hrbf_env_tex.hpp"
#include "hrbf_env_tex_binding.hpp"

#include "precomputed_prim_tex.hpp"
#include "precomputed_prim_env_binding.hpp"

#include "skeleton_env_tex.hpp"
#include "skeleton_env_tex_binding.hpp"

#include "textures_env_tex.hpp"

#include "constants_tex.hpp"
/// @}

/// @name Class implementation using the previous textures
/// @{
#include "hermiteRBF.inl"
#include "precomputed_prim.inl"
/// @}

/// @name Main cuda kernels
/// @{
#include "animesh_kers_projection.hpp"
#include "raytracing_kernel.hpp"
/// @}

#include "toolbox/cuda_utils/cuda_current_device.hpp"
#include "toolbox/portable_includes/port_cuda_gl_interop.h"

#include "globals.hpp"
#include "cuda_globals.hpp"

#include "generic_prim_cu.hpp"

#include "graph.hpp"// DEBUG ======================================
#include "skeleton.hpp"// DEBUG ======================================
#include "g_vbo_primitives.hpp"// DEBUG ======================================

//#include "sys/time.h"

// -----------------------------------------------------------------------------
#include "toolbox/timer.hpp"
void init_cuda(const std::vector<Blending_env::Op_t>& op)
{
    std::cout << "\n--- CUDA CONTEXT SETUP ---\n";
    // We choose the most efficient GPU and use it :
    int device_id = Cuda_utils::get_max_gflops_device_id();

    //CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    //CUDA_SAFE_CALL( cudaDeviceReset() );

    // these two functions are said to be mutually exclusive
    //{
    /// MUST be called after OpenGL/Glew context are init and before any cuda calls that create context like malloc
    CUDA_SAFE_CALL(cudaGLSetGLDevice(device_id) );
    //CUDA_SAFE_CALL(cudaSetDevice(device_id) );
    //}

    //CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    //Cuda_utils::print_device_attribs(get_cu_device() );


    // Compute on host implicit blending operators
    // and allocate them on device memory
    std::cout << "\nInitialize blending operators" << std::endl;
    double free, total;
    Cuda_utils::get_device_memory_usage(free, total);
    std::cout << "free :" << free << "\ntotal" << total << std::endl;

    for(unsigned int i = 0; i < op.size(); ++i)
        Blending_env::enable_predefined_operator( op[i], true );

    Timer t; t.start();
    if (!Blending_env::init_env_from_cache("ENV_CACHE")){
        t.stop();
        Blending_env::init_env();
        Blending_env::make_cache_env("ENV_CACHE");
    } else
        std::cout << "END INIT FROM CACHE : " << t.stop() << "s" << std::endl;

    Blending_env::bind();

    HRBF_env::bind();

    std::cout << "allocate float constants in device memory\n";
    Constants::allocate();
    std::cout << "Done\n";

    std::cout << "allocate textures for raytracing\n";
    Textures_env::load_envmap("resource/textures/env_maps/skymap.ppm");
    Textures_env::load_blob_tex("resource/textures/tex.ppm");
    //Textures_env::load_extrusion_tex("resource/textures/test.ppm", 0);
    //Textures_env::load_extrusion_tex("resource/textures/vortex_text.ppm", 1);
    Textures_env::init();
    std::cout << "Done\n";

    Skeleton_env::init_env();

    g_cuda_context_is_init = true;

    std::cout << "\n--- END CUDA CONTEXT SETUP ---" << std::endl;
}

// =============================================================================
namespace Raytracing {
// =============================================================================

static bool full_eval = false;

// -----------------------------------------------------------------------------

void set_partial_tree_eval(bool s){ full_eval = s; }

// -----------------------------------------------------------------------------

void set_bone_to_trace(const std::vector<int>& bone_ids, int id)
{
    Skeleton_partial_eval::set_bones_to_raytrace(bone_ids, id);
}

// -----------------------------------------------------------------------------

void set_curr_skel_id(int id)
{
    Skeleton_partial_eval::set_current_skel_id(id);
}

// -----------------------------------------------------------------------------

#define RAYTRACE(TEMPLATE) raytrace_kernel<TEMPLATE><<<ctx.grid, ctx.block >>>( d_ctx.ptr() )

void trace( Context& ctx )
{
    Cuda_utils::Device::Array<Context> d_ctx(1);
    d_ctx.set(0, ctx);

    if(full_eval)
    {
        RAYTRACE(Animesh_kers::Skeleton_potential);
    }
    else
    {
        RAYTRACE(Skeleton_partial_eval);
    }

    CUDA_CHECK_ERRORS();
}

// -----------------------------------------------------------------------------

void trace_skinning_skeleton(const Context& ctx, int skel_id)
{

    Cuda_utils::Device::Array<Context> d_ctx(1);
    d_ctx.set(0, ctx);

    Raytracing::set_curr_skel_id( skel_id );
    RAYTRACE(Animesh_kers::Skeleton_potential);
}


}// End Raytracing namespace ===================================================


__global__
void get_controller_values(Cuda_utils::DA_float2 out_vals, int inst_id)
{
    int n = out_vals.size();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        float dot = cosf(idx * (1.f / (float)n) * M_PI);
#if 1
        out_vals[idx] = Blending_env::controller_fetch(inst_id, dot);
#else
        out_vals[idx] = Blending_env::global_controller_fetch(dot);
#endif
    }
}

// -----------------------------------------------------------------------------

