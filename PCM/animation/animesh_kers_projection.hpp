#ifndef ANIMESH_KERS_PROJECTION_HPP__
#define ANIMESH_KERS_PROJECTION_HPP__

/// @file animesh_potential.hpp
/// @warning should be only included once in the project
/// @brief holds the skinning algorithm to project mesh vertices onto
/// the implicit surface

#include "skeleton_env_evaluator.hpp"
#include "animesh_enum.hpp"
#include "toolbox/cuda_utils/cuda_utils.hpp"
#include "toolbox/maths/ray_cu.hpp"
#include "bone.hpp"

#include <math_constants.h>

// Max number of steps for the vertices fit with the dichotomie
#define DICHOTOMIE (20)

#define EPSILON 0.0001f

#define ENABLE_COLOR

#define ENABLE_MARCH

/** @file animation_potential.hpp
 *  Various functions that are used to compute the potential at some point of
 *  the animated skeleton. This file MUST be included in the main cuda
    program file due to the use of cuda textures. Use the header
    animation_kernels.hpp in order to call the kernels outside the cuda main
    programm file (cuda_main_kernel.cu)
 */

// =============================================================================
namespace Animesh_kers {
// =============================================================================

typedef Skeleton_env::Std_bone_eval Std_eval;

/// Evaluate skeleton potential
__device__
float eval_potential(Skeleton_env::Skel_id sid, const Point3& p, Vec3& grad)
{
#if 0
    Skeleton_env::DBone_id curr_bone;//////////////////////////////////DEBUG
    return Skeleton_env::compute_potential_test<Std_eval>(sid, p, grad, curr_bone);//////////////////////////////////DEBUG
#else
    return Skeleton_env::compute_potential<Std_eval>(sid, p, grad);
#endif
}

// -----------------------------------------------------------------------------

/// Computes the potential at each vertex of the mesh. When the mesh is
/// animated, if implicit skinning is enabled, vertices move so as to match that
/// value of the potential.
__global__
void compute_base_potential(Skeleton_env::Skel_id sid,
                            const Point3* in_verts,
                            const int nb_verts,
                            float* base_potential,
                            Vec3* base_grad)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p < nb_verts)
    {
        Vec3 gf;
        float f = eval_potential(sid, in_verts[p], gf);
        if( base_grad != 0 ) base_grad[p] = gf;
        base_potential[p] = f;
    }
}

// -----------------------------------------------------------------------------

/// Compute the potential of a subset of vertices in 'vert_list'.
/// @param sid : skeleton identifier to use for potential evaluation
/// @param in_verts : list of every mesh vertices positions
/// @param vert_list : list of vertex index which potential will be evaluated
/// @param size_vert_list : size of 'vert_list' array.
/// @param base_potential : potential evaluated for every vertices of 'vert_list'
/// stored with the same order.
/// @param base_grad : same as 'base_potential' but for gradients. if null the
/// parameter is ignored.
__global__
void compute_potential(Skeleton_env::Skel_id sid,
                       const Point3* in_verts,
                       const int nb_verts,
                       float* base_potential,
                       Vec3* base_grad)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p < nb_verts)
    {
        Vec3 gf;
        float f = eval_potential(sid, in_verts[ p ], gf);
        if( base_grad != 0 ) base_grad[ p ] = gf;

        base_potential[ p ] = f;
    }
}

// -----------------------------------------------------------------------------

__device__
float dichotomic_search(Skeleton_env::Skel_id sid,
                        const Ray_cu&r,
                        float t0, float t1,
                        Vec3& grad,
                        float iso)
{
    float t = t0;
    float f0 = eval_potential(sid, r(t0), grad);
    float f1 = eval_potential(sid, r(t1), grad);

    if(f0 > f1){
        t0 = t1;
        t1 = t;
    }

    Point3 p;
    for(unsigned short i = 0 ; i < DICHOTOMIE; ++i)
    {
        t = (t0 + t1) * 0.5f;
        p = r(t);
        f0 = eval_potential(sid, p, grad);

        if(f0 > iso){
            t1 = t;
            if((f0-iso) < EPSILON) break;
        } else {
            t0 = t;
            if((iso-f0) < EPSILON) break;
        }
    }
    return t;
}

// -----------------------------------------------------------------------------

/// Search for the gradient divergence section
__device__
float dichotomic_search_div(Skeleton_env::Skel_id sid,
                            const Ray_cu&r,
                            float t0, float t1,
                            Vec3& grad1,
                            float threshold)
{
    //#define FROM_START
    float t;
    Vec3 grad0, grad;
    float f = eval_potential(sid, r(t0), grad0);

    Point3 p;
    for(unsigned short i = 0; i < DICHOTOMIE; ++i)
    {
        t = (t0 + t1) * 0.5f;
        p = r(t);
        f = eval_potential(sid, p, grad);

        if(grad.dot(grad0) > threshold)
        {
            t1 = t;
            grad1 = grad;
        }
        else if(grad.dot(grad1) > threshold)
        {
            t0 = t;
            grad0 = grad;
        }
        else
            break;// No more divergence maybe its a false collision ?
    }
    #ifdef FROM_START
    grad = grad0;
    return t0; // if
    #else
    return t;
    #endif
}

// -----------------------------------------------------------------------------

/// transform iso to sfactor
__device__
inline static float iso_to_sfactor(float x, int s)
{
     #if 0
    x = fabsf(x);
    // We compute : 1-(x^c0 - 1)^c1
    // with c0=2 and c1=4 (note: c0 and c1 are 'slopes' at point x=0 and x=1 )
    x *= x; // x^2
    x = (x-1.f); // (x^2 - 1)
    x *= x; // (x^2 - 1)^2
    x *= x; // (x^2 - 1)^4
    return (x > 1.f) ? 1.f : 1.f - x/* (x^2 - 1)^4 */;
    #elif 1
    x = fabsf(x);
    // We compute : 1-(x^c0 - 1)^c1
    // with c0=1 and c1=4 (note: c0 and c1 are 'slopes' at point x=0 and x=1 )
    //x *= x; // x^2
    x = (x-1.f); // (x^2 - 1)
    float res = 1.f;
    for(int i = 0; i < s; i++) res *= x;
    x = res; // (x^2 - 1)^s
    return (x > 1.f) ? 1.f : 1.f - x/* (x^2 - 1)^s */;
    #else
    return 1.f;
    #endif
}

// -----------------------------------------------------------------------------

__constant__ int skel_id___;

/// @brief Raytracing interface to evaluate the whole skeleton
struct Skeleton_potential{
    __device__
    static float f(const Point3& p){
        Vec3 grad;
        return Skeleton_env::compute_potential/*_inter*/<Std_eval>(Animesh_kers::skel_id___, p, grad); ///////////////DEBUG
    }

    __device__
    static Vec3 gf(const Point3& p){
        Vec3 grad;
        Skeleton_env::compute_potential/*_inter*/<Std_eval>(Animesh_kers::skel_id___, p, grad);////////////////////////DEBUG
        return grad;
    }

    __device__
    static float fngf(const Point3& p, Vec3& gf){
        return Skeleton_env::compute_potential/*_inter*/<Std_eval>(Animesh_kers::skel_id___, p, gf);//////////////////////////DEBUG
    }

    __device__
    static Material_cu mat(const Point3& ){
        return Material_cu();
    }

    __device__ static
    bool is_inter_possible(const Ray_cu& /*ray*/,
                           float& /*tmin*/,
                           float& /*tmax*/)
    {
        return true;
    }
};

}// END Animesh_kers NAMESPACE =================================================

// BEGIN CUDA FACTS : WHAT FOLLOWS CANNOT BE MOVED IN A NAMESPACE ==============

/// Ids of the Bones to be raytrace in device memory
#define NB_BONES_MAX 8
// Well sometimes with cuda constant mem you need to put __device__ before
// the declaration sometimes you don't... You'll just have to see if you end-up
// with corrupted memory inside your kernels...
// I know sometimes cuda feels like lol omg wtf XD
__device__ __constant__ int bone_ids___[NB_BONES_MAX];
/*__device__ */__constant__ int nb_bones___[1];

/** @class Partial_bone_eval
  @brief static class for evaluating bones in the list bone_ids___[]
  @see Skeleton_partial_eval::set_bones_to_raytrace()
*/
class Partial_bone_eval{
public:
    __device__
    static float f(Skeleton_env::DBone_id id_bone, Vec3& grad, const Point3& pt){
        for(int i = 0; i < nb_bones___[0]; i++){
            if(Skeleton_env::DBone_id( bone_ids___[i] ) == id_bone){
                return Skeleton_env::fetch_and_eval_bone(id_bone, grad, pt);
            }
        }
        grad = Vec3(0.f, 0.f, 0.f);
        return 0.f;
    }
};


/** @class Skeleton_partial_eval
  @brief static class used to raytrace a limited list of bones
  This class is used by the raytracer to draw a list of bones defined with
  the function set_bones_to_raytrace()

  @see Skeleton_partial_eval::set_bones_to_raytrace()
*/
struct Skeleton_partial_eval{

    __device__
    static float f(const Point3& p){
        Vec3 gf;
        //return Skeleton_env::compute_potential_inter<Partial_bone_eval>(Animesh_kers::skel_id___, p, gf); //////////////
        return Skeleton_env::compute_potential<Partial_bone_eval>(Animesh_kers::skel_id___, p, gf);
    }

    __device__
    static Vec3 gf(const Point3& p){
        Vec3 gf;
        //Skeleton_env::compute_potential_inter<Partial_bone_eval>(Animesh_kers::skel_id___, p, gf); //////////////
        Skeleton_env::compute_potential<Partial_bone_eval>(Animesh_kers::skel_id___, p, gf);
        return gf;
    }

    __device__
    static float fngf(const Point3& p, Vec3& gf){
        //return Skeleton_env::compute_potential_inter<Partial_bone_eval>(Animesh_kers::skel_id___, p, gf); //////////////
        return Skeleton_env::compute_potential<Partial_bone_eval>(Animesh_kers::skel_id___, p, gf);
    }


    /// set bones to raytrace by taking at most the eight last entries of
    /// std::vector 'bone_ids'.
    __host__
    static void set_bones_to_raytrace(const std::vector<int>& set, Skeleton_env::Skel_id id)
    {
        std::vector<int> ids_set = set;

        assert( ids_set.size() > 0 );
        int nb_bones = min((int)ids_set.size(), NB_BONES_MAX);
        int array[NB_BONES_MAX];
        int acc = 0;
        for(int i = (ids_set.size()-1); (i >= 0) && (acc < NB_BONES_MAX); i--)
        {
            Skeleton_env::DBone_id device_id = Skeleton_env::bone_hidx_to_didx( id, ids_set[i] );
            if( device_id.is_valid() )
            {
                array[acc] = device_id.id();
                acc++;
            }
        }

        // Copy to device constant memory
        CUDA_SAFE_CALL( cudaMemcpyToSymbol(bone_ids___, array,     sizeof(int)*NB_BONES_MAX) );
        CUDA_SAFE_CALL( cudaMemcpyToSymbol(nb_bones___, &nb_bones, sizeof(int)) );
    }

    __host__
    static void set_current_skel_id(Skeleton_env::Skel_id id){
        CUDA_SAFE_CALL( cudaMemcpyToSymbol(Animesh_kers::skel_id___, &id, sizeof(int)) );
    }

    __device__
    static Material_cu mat(const Point3& ){
        return Material_cu();
    }

    __device__ static
    bool is_inter_possible(const Ray_cu& /*ray*/, float& /*tmin*/, float& /*tmax*/){
        return true;
    }
};

// END CUDA FACTS ==============================================================

#include "animation/tests/animesh_kers_proj_expe.inl"
#include "animation/tests/animesh_kers_proj_std_incr.inl"
#include "animesh_kers_proj_standard.inl"
#include "animesh_kers_colors.inl"

#endif //ANIMESH_KERS_PROJECTION_HPP__
