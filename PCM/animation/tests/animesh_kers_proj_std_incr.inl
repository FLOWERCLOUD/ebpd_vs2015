#ifndef ANIMESH_KERS_PROJ_STD_INCR_HPP__
#define ANIMESH_KERS_PROJ_STD_INCR_HPP__

#include "animesh_kers_projection.hpp"


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
__global__
void match_base_potential(Skeleton_env::Skel_id skel_id,
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
                          float* smooth_factors,
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
                          const bool* flip,
                          Vec3* prev_verts)
{
#ifdef ENABLE_MARCH
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_idx < nb_vert_to_fit)
    {
        const int p = vert_to_fit[thread_idx];

        // STOP CASE : Vertex already fitted
        if(p == -1) return;

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
        /*
        if(gf0.norm() <= 0.00001f)
        {
            if(!full_fit) vert_to_fit[thread_idx] = -1;
            d_vert_state[p] = EAnimesh::NORM_GRAD_NULL;
            return;
        }
        */

        // STOP CASE : Point already near enough the isosurface
        if( fabsf(f0) < EPSILON )
        {
            if(!full_fit) vert_to_fit[thread_idx] = -1;
            d_vert_state[p] = EAnimesh::NOT_DISPLACED;
            return;
        }

        d_vert_state[p] = EAnimesh::NB_ITER_MAX;

        // Inside we march along the inverted gradient
        // outside along the gradient :
        float dl = (f0 > 0.f) ? -step_length : step_length;

        Ray_cu r;
        r.set_pos(v0);
        float t = 0.f;


        Vec3   gfi = gf0;
        float  fi  = f0;
        Point3 vi  = v0;

        Vec3   gfi1 = gf0;
        float  fi1  = f0;
        Point3 vi1  = v0;

        Point3 res[2];

        //const Vec3 c_dir = custom_dir[p];
        int vert_state[2] = {EAnimesh::NB_ITER_MAX, EAnimesh::NB_ITER_MAX};
        for(unsigned short s = 0; s < 2; ++s)
        {
            for(unsigned short i = 0; i < nb_iter; ++i)
            {
                r.set_pos(vi1);
                #if 0
                r.set_dir( gfi1.normalized() );
                #else
                const Vec3 c_dir = custom_dir[p];
                r.set_dir(  c_dir );
                /*
                if( gfi1.dot( c_dir ) > 0.f ) r.set_dir(  c_dir );
                else                          r.set_dir( -c_dir );
                */
                #endif
                t = dl;
                vi = r(t);
                fi = eval_potential(skel_id, vi, gfi) - ptl;

                // STOP CASE 1 : Initial iso-surface reached
                if( fi * fi1 <= 0.f)
                {
                    t = dichotomic_search(skel_id, r, 0.f, t, gfi, ptl);
                    if(!full_fit) vert_to_fit[thread_idx] = -1;
                    vert_state[s] = EAnimesh::FITTED;
                    break;
                }

                // STOP CASE 2 : Gradient divergence
                if( gfi1.norm() > EPSILON && gfi.norm() > EPSILON)
                {
                    if( (gfi1.normalized()).dot(gfi.normalized()) < gradient_threshold)
                    {
                        #if 0
                        t = dichotomic_search_div(skel_id, r, -step_length, t, .0,
                                                  gtmp, gradient_threshold);
                        #endif
                        if(!full_fit) vert_to_fit[thread_idx] = -1;

                        vert_state[s] = EAnimesh::GRADIENT_DIVERGENCE;
                        break;
                    }
                }

                // STOP CASE 3 : Potential pit
                if( ((fi - fi1) * dl < 0.f) & potential_pit )
                {
                    if(!full_fit) vert_to_fit[thread_idx] = -1;
                    //smooth_factors[p] = smooth_strength;
                    vert_state[s] = EAnimesh::POTENTIAL_PIT;
                    break;
                }

                vi1  = vi;
                fi1  = fi;
                gfi1 = gfi;

                /*
                if(gfi1.norm_squared() < (0.001f*0.001f)){
                    vert_state[s] = EAnimesh::NORM_GRAD_NULL;
                    break;
                }
                */
            }

            res[s] = r(t);
            dl = -dl;

            gfi1 = gf0;
            fi1  = f0;
            vi1  = v0;
        }


        out_gradient[p] = gfi;

        const Vec3 prev = prev_verts[p];

        int idx = 1;
        if( (prev-res[0]).norm() < (prev-res[1]).norm())
            idx = 0;

        if( vert_state[idx] == EAnimesh::NB_ITER_MAX || vert_state[idx] == EAnimesh::NORM_GRAD_NULL)
            idx = (idx + 1) % 2;

        if( flip[p] ) idx = (idx + 1) % 2;


        out_verts   [p] = res[idx];
        d_vert_state[p] = vert_state[idx];

        if( d_vert_state[p] == EAnimesh::GRADIENT_DIVERGENCE)
            smooth_factors[p] = smooth_strength;

    }
#endif
}
#endif
















/*
    INCREMENTAL ALGORITHM
*/
__global__
void match_base_potential(Skeleton_env::Skel_id skel_id,
                          const bool full_fit,
                          const bool smooth_fac_from_iso,
                          Vec3* out_verts,
                          const Point3* rest_verts,
                          const Transfo* joint_tr,
                          const float* base_potential,
                          const Vec3* normals,
                          Vec3* out_gradient,
                          const Skeleton_env::DBone_id* nearest_bone,
                          const EBone::Id* nearest_bone_cpu,
                          const Skeleton_env::DBone_id* nearest_joint,
                          float* smooth_factors_conservative,
                          float* smooth_factors_laplacian,
                          int* vert_to_fit,
                          const int nb_vert_to_fit,
                          const unsigned short nb_iter,
                          const float gradient_threshold,
                          const float step_length,
                          const bool potential_pit, // TODO: this condition should not be necessary
                          int* d_vert_state,
                          const float smooth_strength,
                          const float collision_depth,
                          const int slope,
                          const bool raphson,
                          const bool* flip)
{
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_idx >= nb_vert_to_fit) return;

    const int p = vert_to_fit[thread_idx];

    // STOP CASE : Vertex already fitted
    if(p == -1) return;

    //const float ptl = 0.6;
    const float ptl = base_potential[p];

    smooth_factors_laplacian[p] = 0.f;

    // Compute initial potential
    Point3 v0 = out_verts[p].to_point3();
#if 1
    Skeleton_env::DBone_id nearest_bone_id_gpu = nearest_bone[p];
    Bone_cu b0 = Skeleton_env::fetch_bone_cu( nearest_bone_id_gpu );
    Point3 p_plane = b0.project( v0 );
    Vec3  n_plane = (v0 - p_plane).normalized();
    // give same orientaion as gradient
    //if( n_plane.dot( gf0.normalized() ) < 0.0f )
    {
        //n_plane *= -1.f;
    }
#endif

    Vec3 gf0;
    float f0;
    Skeleton_env::DBone_id bone_start;
    //f0 = Skeleton_env::compute_potential_test<Std_eval>(skel_id, v0, gf0, bone_start) - ptl;
    f0 = eval_potential(skel_id, v0, gf0) - ptl;////////////////////////////////////////////////////////////////
    out_gradient[p] = /*n_plane.normalized()*/gf0;////////////////////////////////////////////////////////////////DEBUG

    if( smooth_factors_conservative[p] == 0.f) return;
    /////////////////////////////////////


    // STOP CASE : Gradient is null we can't know where to march
    if(gf0.norm() <= 0.00001f) // TODO: Should be proportionnal to the scale of the mesh
    {
        //if(!full_fit) vert_to_fit[thread_idx] = -1;
        d_vert_state[p] = EAnimesh::NORM_GRAD_NULL;
        return;
    }

    // STOP CASE : Point already near enough the isosurface
    if( fabsf(f0) < EPSILON ) // TODO: Should be proportionnal to the scale of the mesh
    {
        //if(!full_fit) vert_to_fit[thread_idx] = -1;
        d_vert_state[p] = EAnimesh::NOT_DISPLACED;
        return;
    }



    bool is_out = (f0 < 0.f); // Check if outside the implicit surface

    // Inside we march along the inverted gradient
    // outside along the gradient :
    float dl = (f0 > 0.f) ? -step_length : step_length;

    Ray_cu r;
    r.set_pos(v0);
    float t = 0.f;

    Vec3  gfi = gf0;
    float fi = f0;
    Point3 vi = v0;

    Skeleton_env::DBone_id bone_id;
    //bool inverted = normals[p].dot( gf0.normalized() ) < 0.0f;
    Vec3 dir = normals[p].normalized();

    dir = dir.dot( gf0.normalized() ) < 0.f ? -dir : dir;
    // Constant dir in the same direction of the grad
    for(unsigned i = 0; i < nb_iter; ++i)
    {
        r.set_dir(  /*dir*/ gf0.normalized() /*n_plane.normalized()*/ );
        r.set_pos( v0 );
        t = dl;

        vi = r(t);

        //fi = Skeleton_env::compute_potential_test<Std_eval>(skel_id, vi, gfi, bone_id) - ptl;
        fi = eval_potential(skel_id, vi, gfi) - ptl;/////////////////////////////////////////////

#if 0
        if( !is_out )
        {
            if( bone_start != bone_id /*|| bone_start != nearest_bone_id_gpu*/)
            {
                t = dichotomic_search_div_2(skel_id, r, 0.f, t, normals[p], gfi);
                smooth_factors_laplacian[p] = 1.f;

                smooth_factors_conservative[p] = 0.f;
                //t = 0.f;
                d_vert_state[p] = EAnimesh::GRADIENT_DIVERGENCE;
                break;
            }
        }
#endif

        // STOP CASE : Gradient is null
        if(gf0.norm_squared() < (0.001f*0.001f)){
            d_vert_state[p] = EAnimesh::NORM_GRAD_NULL;
            break;
        }

        // STOP CASE : Initial iso-surface reached
        if( fi * f0 <= 0.f )
        {
            t = dichotomic_search(skel_id, r, 0.f, t, gfi, ptl);

            /* if(!full_fit)*/
            //vert_to_fit[thread_idx] = -1;
//            smooth_factors[p] = smooth_factors_conservative[p] = 0.0f;

            d_vert_state[p] = EAnimesh::FITTED;
            break;
        }

        // STOP CASE : Gradient divergence
        if( (gf0.normalized()).dot(gfi.normalized()) < gradient_threshold && !is_out) // Can't collide if came from outside.
        {
            #if 0
            t = dichotomic_search_div_2(skel_id, r, 0.f, t, normals[p], gfi);
            #endif

            //if(!full_fit) vert_to_fit[thread_idx] = -1;


            // Do not relax or smooth the mesh
            //vert_to_fit[thread_idx] = -1;
            smooth_factors_conservative[p] = 0.f;
            smooth_factors_laplacian[p] = 1.f;

            t = 0.f;
            d_vert_state[p] = EAnimesh::GRADIENT_DIVERGENCE;
            break;
        }

        // STOP CASE 3 : Potential pit
        if( ((fi - f0)*dl < 0.f) & potential_pit )
        {
            t = 0.f;
            //vert_to_fit[thread_idx] = -1;
            smooth_factors_conservative[p] = 0.f;
            smooth_factors_laplacian[p] = 1.f;
            d_vert_state[p] = EAnimesh::POTENTIAL_PIT;
            break;
        }

        v0  = vi;
        f0  = fi;
        gf0 = gfi;
    }

    out_gradient[p] = gfi /*n_plane.normalized()*/;////////////////////////////////////////////////////////////////DEBUG
    out_verts[p] = r(t);
}

}// END Animesh_kers NAMESPACE =================================================

#endif // ANIMESH_KERS_PROJ_STD_INCR_HPP__
