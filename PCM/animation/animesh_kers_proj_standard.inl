#ifndef ANIMESH_KERS_PROJ_STANDARD_HPP__
#define ANIMESH_KERS_PROJ_STANDARD_HPP__

#include "animesh_kers_projection.hpp"

/**
 * @file animesh_kers_proj_standard.inl
 * @brief implemention of the standard kernels related to mesh deformation
 *
 */

// =============================================================================
namespace Animesh_kers {
// =============================================================================

#if 1

/* ORIGINAL ALGORITHM
    Ajustement standard avec gradient
*/

/// Move the vertices along a mix between their normals and the joint rotation
/// direction in order to match their base potential at rest position
/// @param d_output_vertices  vertices array to be moved in place.
/// @param d_ssd_interpolation_factor  interpolation weights for each vertices
/// which defines interpolation between ssd animation and implicit skinning
/// 1 is full ssd and 0 full implicit skinning
/// @param do_tune_direction if false use the normal to displace vertices
/// @param gradient_threshold when the mesh's points are fitted they march along
/// the gradient of the implicit primitive. this parameter specify when the vertex
/// stops the march i.e when gradient_threshold < to the scalar product of the
/// gradient between two steps
/// @param full_eval tells is we evaluate the skeleton entirely or if we just
/// use the potential of the two nearest clusters, in full eval we don't update
/// d_vert_to_fit has it is suppossed to be the last pass
///(Skeleton_env::Skel_id skel_id,
///

__global__
void match_base_potential_standard
                         (Skeleton_env::Skel_id skel_id,
                          const bool full_fit,
                          const bool smooth_fac_from_iso,
                          Vec3* out_verts,
                          const Point3* rest_verts,
                          const Transfo* joint_tr,
                          const float* base_potential,
                          const Vec3* custom_dir,
                          Vec3* out_gradient,
                          const Skeleton_env::DBone_id* nearest_bone,
                          const EBone::Id* nearest_bone_cpu,
                          const Skeleton_env::DBone_id* nearest_joint,
                          float* smooth_factors_iso,
                          float* smooth_factors_laplacian,
                          int* vert_to_fit,
                          const int nb_vert_to_fit,
                          const unsigned short nb_iter,
                          const float gradient_threshold,
                          const float step_length,
                          const bool potential_pit,
                          int* d_vert_state,
                          const float smooth_strength,
                          const float collision_depth,
                          const int slope,
                          const bool raphson,
                          const bool* flip)

{
#ifdef ENABLE_MARCH
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_idx < nb_vert_to_fit)
    {
        const int p = vert_to_fit[thread_idx];

        // STOP CASE : Vertex already fitted
        if(p == -1) return;

        // Reset laplacian smoothing
        smooth_factors_laplacian[p] = 0.f;

//        const Skeleton_env::DBone_id nearest = full_eval ? Skeleton_env::DBone_id(-1) : nearest_bone[p];
        const float ptl = base_potential[p];

        Point3 v0 = out_verts[p].to_point3();
        Vec3 gf0;
        float f0;
        f0 = eval_potential(skel_id, v0, gf0) - ptl;

        if(smooth_fac_from_iso)
            smooth_factors_iso[p] = iso_to_sfactor(f0, slope) * smooth_strength;

        out_gradient[p] = gf0;
        // STOP CASE : Gradient is null we can't know where to march
        if(gf0.norm() <= 0.00001f){
            if(!full_fit) vert_to_fit[thread_idx] = -1;
            #ifdef ENABLE_COLOR
            d_vert_state[p] = EAnimesh::NORM_GRAD_NULL;
            #endif
            return;
        }

        // STOP CASE : Point already near enough the isosurface
        if( fabsf(f0) < EPSILON ){
            if(!full_fit) vert_to_fit[thread_idx] = -1;
            #ifdef ENABLE_COLOR
            d_vert_state[p] = EAnimesh::NOT_DISPLACED;
            #endif
            return;
        }

        #ifdef ENABLE_COLOR
        d_vert_state[p] = EAnimesh::NB_ITER_MAX;
        #endif

        // Inside we march along the inverted gradient
        // outside along the gradient :
        const float dl = (f0 > 0.f) ? -step_length : step_length;

        Ray_cu r;
        r.set_pos(v0);
        float t = 0.f;

        Vec3  gfi    = gf0;
        float    fi     = f0;
        float    abs_f0 = fabsf(f0);
        Point3 vi     = v0;

        //const Vec3 c_dir = custom_dir[p];

        for(unsigned short i = 0; i < nb_iter; ++i)
        {

            r.set_pos(v0);
            if( raphson ){
                float nm = gf0.norm_squared();
                r.set_dir( gf0 );
                t = dl * abs_f0 / nm;
                //t = t < 0.001f ? dl : t;
            } else {
                #if 1
                    r.set_dir( gf0.normalized() );
                #else
                    if( gf0.dot( c_dir ) > 0.f ) r.set_dir(  c_dir );
                    else                         r.set_dir( -c_dir );

                #endif
                t = dl;
            }

            vi = r(t);
            fi = eval_potential(skel_id, vi, gfi) - ptl;

            // STOP CASE 1 : Initial iso-surface reached
            abs_f0 = fabsf(fi);
            if(raphson && abs_f0 < EPSILON )
            {
                if(!full_fit) vert_to_fit[thread_idx] = -1;
                #ifdef ENABLE_COLOR
                d_vert_state[p] = EAnimesh::OUT_VERT;
                #endif
                break;
            }
            else if( fi * f0 <= 0.f)
            {
                t = dichotomic_search(skel_id, r, 0.f, t, gfi, ptl);

                if(!full_fit) vert_to_fit[thread_idx] = -1;

                #ifdef ENABLE_COLOR
                d_vert_state[p] = EAnimesh::FITTED;
                #endif
                break;
            }

            // STOP CASE 2 : Gradient divergence
            if( (gf0.normalized()).dot(gfi.normalized()) < gradient_threshold)
            {
                #if 0
                t = dichotomic_search_div(skel_id, r, -step_length, t, .0,
                                          gtmp, gradient_threshold);
                #endif

                if(!full_fit) vert_to_fit[thread_idx] = -1;

                smooth_factors_laplacian[p] = smooth_strength;
                #ifdef ENABLE_COLOR
                d_vert_state[p] = EAnimesh::GRADIENT_DIVERGENCE;
                #endif
                break;
            }

            // STOP CASE 3 : Potential pit
            if( ((fi - f0)*dl < 0.f) & potential_pit )
            {
                if(!full_fit) vert_to_fit[thread_idx] = -1;
                smooth_factors_laplacian[p] = smooth_strength;
                #ifdef ENABLE_COLOR
                d_vert_state[p] = EAnimesh::POTENTIAL_PIT;
                #endif
                break;
            }

            v0  = vi;
            f0  = fi;
            gf0 = gfi;

            if(gf0.norm_squared() < (0.001f*0.001f)) break;
        }

        const Point3 res = r(t);
        out_gradient[p] = gfi;
        out_verts[p] = res;
    }
#endif
}
#endif

}// END Animesh_kers NAMESPACE =================================================

#endif // ANIMESH_KERS_PROJ_STANDARD_HPP__
