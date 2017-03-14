#ifndef ANIMESH_KERS_HPP_
#define ANIMESH_KERS_HPP_

#include <utility>

#include "toolbox/cuda_utils/cuda_utils.hpp"
#include "toolbox/maths/transfo.hpp"
#include "toolbox/maths/mat2.hpp"

#include "mesh.hpp"
#include "skeleton.hpp"
#include "animesh_enum.hpp"

/** @namespace Kernels
    @brief The cuda kernels used to animate the mesh

    @see Animesh
 */

// =============================================================================
namespace Animesh_kers{
// =============================================================================

using namespace Cuda_utils;

/// Transform each vertex rigidely according to its nearest bone
__global__
void transform_rigid(const Point3* in_verts,
                     const Vec3* in_normals,
                     int nb_verts,
                     Vec3* out_verts,
                     Vec3* out_verts2,
                     Vec3* out_normals,
                     const Transfo* transfos,
                     Mat3 *rots,
                     const int* nearest_bone);

/// Transform each vertex with SSD
/// @param in_verts : vertices in rest position
/// @param out_verts : animated vertices
/// @param out_verts_2 : animated vertices (same as out_verts)
__global__
void transform_SSD(const Point3* in_verts,
                   const Vec3* in_normals,
                   int nb_verts,
                   Vec3* out_verts,
                   Vec3* out_verts2,
                   Vec3* out_normals,
                   const Transfo* transfos,
                   const float* weights,
                   const int* joints,
                   const int* jpv );

/// Transform each vertex with dual quaternions
/// @param in_verts : vertices in rest position
/// @param out_verts : animated vertices
/// @param out_verts_2 : animated vertices (same as out_verts)
__global__
void transform_dual_quat(const Point3* in_verts,
                         const Vec3* in_normals,
                         int nb_verts,
                         Vec3* out_verts,
                         Vec3* out_verts2,
                         Vec3* out_normals,
                         const Dual_quat_cu* d_transform,
                         const float* d_weights,
                         const int* d_joints,
                         const int* d_jpv);

__global__
void transform_arap_dual_quat(Mat3* rots,
                              int nb_verts,
                              const Dual_quat_cu* dual_quat,
                              const float* weights,
                              const int* joints,
                              const int* jpv);

/// Linear interpolation between verts_0 and verts_1 given the lerp_factor
/// at each vertices
__global__
void lerp_kernel( const int* vert_to_fit,
                  const Vec3* verts_0,
                  const Vec3* verts_1,
                  float* lerp_factor,
                  Vec3* out_verts,
                  int nb_verts);

/// Computes the potential at each vertex of the mesh. When the mesh is
/// animated, if implicit skinning is enabled, vertices move so as to match
/// that value of the potential.
__global__ void
compute_base_potential(Skeleton_env::Skel_id sid,
                       const Point3* d_input_vertices,
                       const int nb_verts,
                       float* d_base_potential,
                       Vec3* d_base_gradient );

__global__
void compute_potential(Skeleton_env::Skel_id sid,
                       const Point3* in_verts,
                       const int nb_verts,
                       float* base_potential,
                       Vec3* base_grad);

/// Match the base potential after basic ssd deformation
/// (i.e : do the implicit skinning step)
__global__
void match_base_potential_standard(Skeleton_env::Skel_id skel_id,
                                   const bool full_fit,
                                   const bool smooth_fac_from_iso,
                                   Vec3* d_output_vertices,
                                   const Point3* rest_verts,
                                   const Transfo* joint_tr,
                                   const float* d_base_potential,
                                   const Vec3* custom_dir,
                                   Vec3* d_gradient,
                                   const Skeleton_env::DBone_id* nearest_bone,
                                   const EBone::Id* nearest_bone_cpu,
                                   const Skeleton_env::DBone_id* nearest_joint,
                                   float* d_smooth_factors_iso,
                                   float* d_smooth_factors,
                                   int* d_vert_to_fit,
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
                                   const bool* flip);

__global__
void match_base_potential(Skeleton_env::Skel_id skel_id,
                          const bool full_fit,
                          const bool smooth_fac_from_iso,
                          Vec3* d_output_vertices,
                          const Point3* rest_verts,
                          const Transfo* joint_tr,
                          const float* d_base_potential,
                          const Vec3* custom_dir,
                          Vec3* d_gradient,
                          const Skeleton_env::DBone_id* nearest_bone,
                          const EBone::Id* nearest_bone_cpu,
                          const Skeleton_env::DBone_id* nearest_joint,
                          float* d_smooth_factors_iso,
                          float* d_smooth_factors,
                          int* d_vert_to_fit,
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
                          const bool* flip);

/// Compute on GPU the normals of the mesh using the normal at each face
void compute_normals(const int* tri,
                     const int* quad,
                     Device::Array<EMesh::Prim_idx_vertices> piv,
                     int nb_tri,
                     int nb_quad,
                     const Vec3* vertices,
                     Device::Array<Vec3> unpacked_normals,
                     int unpack_factor,
                     Vec3* out_normals);

/// Compute the tangents at each face
void compute_tangents(const int* tri,
                      const int* quad,
                      const int* unpacked_tri,
                      const int* unpacked_quad,
                      Device::Array<EMesh::Prim_idx_vertices> piv,
                      int nb_tri,
                      int nb_quad,
                      const Vec3* vertices,
                      const float* tex_coords,
                      Device::Array<Vec3> unpacked_tangents,
                      int unpack_factor,
                      Vec3* out_tangents);

/// Tangential relaxation of the vertices. Each vertex is expressed with the
/// mean value coordinates (mvc) of its neighborhood. While animating we try
/// to move back the vertices to their old position with the mvc.
/// (N.B mvc are barycentric coordinates computed in the tangent plane of the
/// vertex, the plane can be defined either by the vertex's normal or
/// implicit gradient)
void conservative_smooth(Vec3* d_input_verts,
                         Vec3* d_verts,
                         Vec3* d_buff_verts,
                         Vec3* d_normals,
                         Vec3* d_base_grad,
                         Mat2 *rots,
                         float* d_2nd_ring_lengths,
                         int* d_2nd_ring_list,
                         int* d_2nd_ring_list_offsets,
                         float* d_1st_ring_cotan,
                         float* d_1st_ring_lengths,
                         float* d_1st_ring_angle,
                         const DA_int& d_1st_ring_list,
                         const DA_int& d_1st_ring_list_offsets,
                         const DA_float& d_edge_mvc,
                         const int* d_vert_to_fit,
                         int nb_vert_to_fit,
                         bool /*use_vert_to_fit*/,
                         float strength,
                         int nb_iter,
                         const float* smooth_fac,
                         bool use_smooth_fac);

/// A basic laplacian smooth which move the vertices between its position
/// and the barycenter of its neighborhoods
/// @param factor sets for each vertex a weight between [0 1] which define
/// the smoothing strenght
/// @param use_smooth_factors do we use the array "factor" for smoothing
/// @param strength smoothing force when "use_smooth_factors"==false
void laplacian_smooth(Vec3* d_vertices,
                      Vec3* d_tmp_vertices,
                      float* fst_ring_cotan_weights,
                      DA_int d_1st_ring_list,
                      DA_int d_1st_ring_list_offsets,
                      float* factors,
                      bool use_smooth_factors,
                      float strength,
                      int nb_iter,
                      int nb_min_neighbours);

/// Basic diffusion of values on the mesh. For each vertex i we compute :
/// new_val(i) = val(i) * (1. - strength) + strength / sum( val(neighborhoods(i)) )
/// @param d_values diffused values computed in place
/// @param d_values_buffer an allocated buffer of the same size as 'd_values'
/// @param strenght is the strenght of the diffusion
/// @param nb_iter number of iteration to apply the diffusion. Avoid odd numbers
/// wich will imply a recopy of the array d_values, prefer even numbers
void diffuse_values(float* d_values,
                    float* d_values_buffer,
                    DA_int d_1st_ring_list,
                    DA_int d_1st_ring_list_offsets,
                    float strength,
                    int nb_iter);

/// Copy d_vertices_in of size n in d_vertices_out
template< class T >
__global__
void copy_arrays(const T* d_in, T* d_out, int n)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p < n)  d_out[p] = d_in[p];
}

/// Fill the array with its the subscript index at each element
__global__
void fill_index(DA_int array);

/// For rendering some vertives needs to be duplicated because of multiples
/// textures coordinates. This kernels does that.
__global__
void unpack_vert_and_normals(const Vec3* packed_vert,
                             const Vec3* packed_normals,
                             const Vec3* packed_tangents,
                             const EMesh::Packed_data* packed_vert_map,
                             Vec3* unpacked_vert,
                             Vec3* unpacked_normals,
                             Vec3* unpacked_tangents,
                             int nb_vert);

}// END Animesh_kers NAMESPACE =================================================

#endif // ANIMESH_KERS_HPP_
