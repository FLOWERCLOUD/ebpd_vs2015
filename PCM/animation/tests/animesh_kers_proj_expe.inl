#ifndef ANIMESH_KERS_PROJ_EXPE_HPP__
#define ANIMESH_KERS_PROJ_EXPE_HPP__

#include "animesh_kers_projection.hpp"

/**
 * @file animesh_kers_proj_expe.inl
 * @brief implemention of the experimental kernels related to mesh deformation
 *
 */

// =============================================================================
namespace Animesh_kers {
// =============================================================================

#if 0

__device__
Vec3 bone_cross(const Bone_cu& b0, const Point3& v0)
{
    return (v0 - b0.org()).cross( b0.dir() ).normalized();
}

// -----------------------------------------------------------------------------

/*
    EXPERIMENTAL ALGORITHM
*/
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
                          const bool potential_pit, // TODO: this condition should not be necessary
                          int* d_vert_state,
                          const float smooth_strength,
                          const float collision_depth,
                          const int slope,
                          const bool raphson,
                          const bool* flip,
                          Vec3* prev_verts)
{
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_idx >= nb_vert_to_fit) return;

    const int p = vert_to_fit[thread_idx];

    // STOP CASE : Vertex already fitted
    if(p == -1) return;

    //        const Skeleton_env::DBone_id nearest = full_eval ? Skeleton_env::DBone_id(-1) : nearest_bone[p];
    const float ptl = base_potential[p];

    Point3 v0         = out_verts[p].to_point3(); // Initial position
    Point3 rigid_vert = joint_tr[ nearest_bone_cpu[p] ] * rest_verts[p];

    #if 0
    {
        Skeleton_env::DBone_id bid = nearest_joint[p];
        Bone_cu b0 = Skeleton_env::fetch_bone_cu( bid );

        Skeleton_env::DBone_id pid = Skeleton_env::fetch_bone_parent(bid);
        if( pid.is_valid() ){
            Bone_cu b1 = Skeleton_env::fetch_bone_cu(  pid );
            if( fabsf( b0.dist_to( v0 ) - b1.dist_to( v0 ) ) < 0.2f ){
                d_vert_state[p] = EAnimesh::BACK;
                return;
            }
        }

        d_vert_state[p] = EAnimesh::FITTED;
        return;
    }
    #endif


    #if 0
    // Trying to compute mesh at the back of the bone
    bool back = false;
    {
        Skeleton_env::DBone_id bid = nearest_joint[p];
        Bone_cu b0 = Skeleton_env::fetch_bone_cu( bid );

        Skeleton_env::DBone_id pid = Skeleton_env::fetch_bone_parent(bid);
        if( pid.is_valid() )
        {
            Bone_cu b1 = Skeleton_env::fetch_bone_cu(  pid );
            if( (v0 - b0.project(v0)).normalized().dot( (v0 - b1.project(v0)).normalized() ) > 0.90f && fabsf( b0.dist_to( v0 ) - b1.dist_to( v0 ) ) < 0.2f){
                back = true;
            }
        }
    }
    #endif

    Skeleton_env::DBone_id nearest_bone_id_gpu = nearest_bone[p];
    Bone_cu b0 = Skeleton_env::fetch_bone_cu( nearest_bone_id_gpu );
    Point3 p_plane = b0.project( v0 );
    Vec3  n_plane = (v0 - p_plane).normalized();

    Vec3 init_orientation = bone_cross(b0, v0);

    bool behind_bone = init_orientation.dot( bone_cross(b0, rigid_vert)) < 0.f;


    Vec3 gf0;
    bool is_inter = Skeleton_env::compute_potential_inter<Std_eval>(skel_id, v0, gf0) > 0.5f;
    gf0 = Vec3::zero();

    float f0;
    Skeleton_env::DBone_id bone_start;
    f0 = Skeleton_env::compute_potential_test<Std_eval>(skel_id, v0, gf0, bone_start) - ptl;
//    f0 = eval_potential(skel_id, v0, gf0) - ptl;

    Vec3 c_dir = custom_dir[p].normalized();
    {
        // Correct direction of projection
        float angle = c_dir.dot( gf0.normalized() );
        float sign = angle > 0.f ? 1.f : -1.f;
        angle *= sign;

        float coeff = 1.f - angle;
        c_dir = c_dir + gf0.normalized() * sign * coeff * 3.f;
        c_dir.normalize();

        c_dir = gf0.normalized();
    }

    if(smooth_fac_from_iso)
        smooth_factors_iso[p] = iso_to_sfactor(f0, slope) * smooth_strength;

    out_gradient[p] = gf0;
    // STOP CASE : Gradient is null we can't know where to march
    if(gf0.norm() <= 0.00001f){
        if(!full_fit) vert_to_fit[thread_idx] = -1;
        d_vert_state[p] = EAnimesh::NORM_GRAD_NULL;
        return;
    }

    // STOP CASE : Point already near enough the isosurface
    if( fabsf(f0) < EPSILON ){
        if(!full_fit) vert_to_fit[thread_idx] = -1;
        d_vert_state[p] = EAnimesh::NOT_DISPLACED;
        return;
    }

    // Inside we march along the inverted gradient
    // outside along the gradient :
    bool is_out = (f0 < 0.f);
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
    Skeleton_env::DBone_id bone_end[2];
    int vert_state[2] = {EAnimesh::NB_ITER_MAX, EAnimesh::NB_ITER_MAX};

    // Constant dir in the same direction of the grad
    if( gfi1.dot( c_dir ) > 0.f ) r.set_dir(  c_dir );
    else                          r.set_dir( -c_dir );

    for(unsigned short s = 0; s < 2; ++s)
    {
        for(unsigned short i = 0; i < nb_iter; ++i)
        {
            r.set_pos(vi1);
            t = dl;

            vi = r(t);
//            fi = eval_potential(skel_id, vi, gfi) - ptl;
            fi = Skeleton_env::compute_potential_test<Std_eval>(skel_id, vi, gfi, bone_end[s]) - ptl;


            // STOP CASE : Gradient is null
//            if(gfi1.norm_squared() < (0.001f*0.001f)){
//                vert_state[s] = EAnimesh::NORM_GRAD_NULL;
//                break;
//            }

            // STOP CASE : going to the other side of clipping plane
            if( n_plane.dot(vi - p_plane)  < -0.01f){ // plane slightly below to ensure collisions happens first
                vert_state[s] = EAnimesh::PLANE_CULLING;
                break;
            }

            if( init_orientation.dot( (v0 - b0.org()).cross( b0.dir() ).normalized() ) < -0.001f ){
                vert_state[s] = EAnimesh::CROSS_PROD_CULLING;
                break;
            }

            // STOP CASE : Bone id has switched
            if( /*!back &&*/ !is_out)
            {
                if( bone_start != bone_end[s] || bone_start != nearest_bone_id_gpu)
                {
                    if(!full_fit) vert_to_fit[thread_idx] = -1;

                    smooth_factors[p] = smooth_strength;
                    vert_state[s] = EAnimesh::GRADIENT_DIVERGENCE;
                    break;
                }
            }

            // STOP CASE : Initial iso-surface reached
            if( fi * fi1 <= 0.f )
            {
                t = dichotomic_search(skel_id, r, 0.f, t, gfi, ptl);

                if(!full_fit) vert_to_fit[thread_idx] = -1;

                vert_state[s] = EAnimesh::FITTED;
                break;
            }

            // STOP CASE : Gradient divergence
//            if( (gfi1.normalized()).dot(gfi.normalized()) < gradient_threshold && !is_out)
//            {
//                #if 0
//                t = dichotomic_search_div(skel_id, r, -step_length, t, .0,
//                                          gtmp, gradient_threshold);
//                #endif

//                if(!full_fit) vert_to_fit[thread_idx] = -1;

//                smooth_factors[p] = smooth_strength;
//                vert_state[s] = EAnimesh::GRADIENT_DIVERGENCE;
//                p_coll[s] = true;
//                break;
//            }

            // STOP CASE 3 : Potential pit
            /*
            if( ((fi - fi1)*dl < 0.f) & potential_pit )
            {
                if(!full_fit) vert_to_fit[thread_idx] = -1;
                smooth_factors[p] = smooth_strength;
                vert_state[s] = EAnimesh::POTENTIAL_PIT;
                break;
            }
            */

            vi1  = vi;
            fi1  = fi;
            gfi1 = gfi;
            //bone_start = bone_end[s];
        }

        //Skeleton_env::compute_potential_test<Std_eval>(skel_id, vi, gfi, bone_end[s]);
        res[s] = r(t);
        dl = -dl;

        gfi1 = gf0;
        fi1  = f0;
        vi1  = v0;
        r.set_pos(v0);
    }

    // By default we trust first position more than second position
    int idx = 0;
#if 0
    if( is_out )
    {
        // Out vertices should always come back to their primitive
        Vec3 grad_nearest;
        float p = Skeleton_env::Std_bone_eval::f(nearest_bone_id_gpu, grad_nearest, v0) - ptl;

        // in case of composition we might be inside the prim even though we're
        // outside the whole tree.
        if( p > 0.f /*if inside we invert grad*/)
            grad_nearest = -grad_nearest;

        if( r._dir.dot( grad_nearest ) > 0.f ) idx = 0;
        else                                   idx = 1;
    }
    else
    {
/*
        if( vert_state[0] == EAnimesh::FITTED && bone_end[0] == nearest_bone_id){
            idx = 0;
        }else if(vert_state[1] == EAnimesh::FITTED && bone_end[1] == nearest_bone_id){
            idx = 1;
        }
        else
  */
        if( behind_bone ) // marche pas si la collision est encore plus dérrière !
        {
            // minus dir because inside we follow the inverse dir
            if( (-r._dir).dot( n_plane ) < 0.f ) idx = 0;
            else                                 idx = 1;
        }
        else
        {
            // prefer collision to fitting or iter max (should be a distance max instead of iter max i think)
            bool state_1 = (vert_state[1] == EAnimesh::FITTED || vert_state[1] == EAnimesh::NB_ITER_MAX);
            bool state_0 = (vert_state[0] == EAnimesh::FITTED || vert_state[0] == EAnimesh::NB_ITER_MAX);

            if( state_1 && vert_state[0] == EAnimesh::GRADIENT_DIVERGENCE){
                idx = 0;
            }else if( state_0 && vert_state[1] == EAnimesh::GRADIENT_DIVERGENCE ){
                idx = 1;
            }
            else if(vert_state[0] == vert_state[1])// between equal states take the nearest
            {
                idx = (res[0] - v0).norm() < (res[1] - v0).norm() ? 0 : 1;
            }
        }
    }
#endif

    // TODO: si je passe derrière mon os alors je doit revenir vers lui car je suis dans un plie
    // -> calculer si dérrier avec la transfo rigide du sommet

    // TODO: éviter les directions trop parallele à l'os (trop perpendiculaire au gradient même)
    // En fait la direction de projection doit tenir compte de la surface (vec normales) des prims
    // implicit (gradient) et doit être lisse (lissage laplacien ?)

    // TODO: Quand on revient vers l'os calcul un step_length qui garantie qu'on l'atteint et un peu au dela
    // on pourrait aussi faire un truc pareil pour quand il s'éloigne: genre au mois la distance initiale au repos à l'os vers l'ext

    d_vert_state[p] = vert_state[raphson ? 0 : 1];

//    if( back )d_vert_state[p] = EAnimesh::NORM_GRAD_NULL;
    if( flip[p] ) idx = idx+1 % 2;

    out_gradient[p] = gfi;
    out_verts[p] = res[idx]; /*+ n_plane * 2.f*/;
}
#endif


}// END Animesh_kers NAMESPACE =================================================

#endif // ANIMESH_KERS_PROJ_EXPE_HPP__
