#include "animesh_kers.hpp"

#include "toolbox/maths/dual_quat_cu.hpp"
#include "toolbox/cuda_utils/cuda_current_device.hpp"
#include "toolbox/maths/color.hpp"
#include "toolbox/utils.hpp"
#include "toolbox/maths/intersections/intersection.hpp"
#include "toolbox/maths/svd_2x2.hpp"

#include "toolbox/cuda_utils/cuda_utils_thrust.hpp"

#ifndef PI
#define PI (3.14159265358979323846f)
#endif

// =============================================================================
namespace Animesh_kers{
// =============================================================================

__global__
void transform_rigid(const Point3* in_verts,
                     const Vec3* in_normals,
                     int nb_verts,
                     Vec3* out_verts,
                     Vec3* out_verts2,
                     Vec3* out_normals,
                     const Transfo* transfos,
                     Mat3 *rots,
                     const int* nearest_bone)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p < nb_verts )
    {

        Transfo t = transfos[ nearest_bone[p] ];
        // Compute animated position
        Vec3 vi = (t * in_verts[p]).to_vec3();
        out_verts [p] = vi;
        out_verts2[p] = vi;
        // Compute animated normal
        out_normals[p] = t.fast_invert().transpose() * in_normals[p];

        if(rots != 0) rots[p] = t.get_mat3() * rots[p];
    }
}

// -----------------------------------------------------------------------------

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
                   const int* jpv)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p < nb_verts )
    {
        // compute vertex new position
        const int st_j = jpv[2*p  ]; // First joint idx in d_transform
        const int nb_j = jpv[2*p+1]; // nb joints influencing the vertex
        Transfo t;
        t = ( nb_j > 0) ? transfos[ joints[st_j] ] * weights[st_j] :
                          Transfo::identity();

        for(int j = st_j + 1; j < (st_j + nb_j); j++)
        {
            const int   k = joints [j];
            const float w = weights[j];
            t = t + transfos[k] * w;
        }

        // Compute animated position
        Vec3 vi = (t * in_verts[p]).to_vec3();
        out_verts [p] = vi;
        out_verts2[p] = vi;
        // Compute animated normal
        out_normals[p] = t.fast_invert().transpose() * in_normals[p];
    }
}

// -----------------------------------------------------------------------------

__global__
void transform_dual_quat( const Point3* in_verts,
                          const Vec3* in_normals,
                          int nb_verts,
                          Vec3* out_verts,
                          Vec3* out_verts2,
                          Vec3* out_normals,
                          const Dual_quat_cu* dual_quat,
                          const float* weights,
                          const int* joints,
                          const int* jpv)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p < nb_verts)
    {
        // compute vertex new position
        //Vec3 axis = {0.f, 0.f, 0.f};

        const int st_j = jpv[2*p  ];
        const int nb_j = jpv[2*p+1];

        int   k0 = -1;
        float w0 = 0.f;
        Dual_quat_cu dq_blend;
        Quat_cu q0;

        if(nb_j != 0)
        {
            k0 = joints [st_j];
            w0 = weights[st_j];
        }else
            dq_blend = Dual_quat_cu::identity();

        if(k0 != -1) dq_blend = dual_quat[k0] * w0;

        int pivot = k0;

        q0 = dual_quat[pivot].rotation();

        for(int j = st_j+1; j < st_j + nb_j; j++)
        {
            const int k = joints [j];
            float w = weights[j];
            const Dual_quat_cu& dq = (k == -1) ? Dual_quat_cu::identity() : dual_quat[k];

            if( dq.rotation().dot( q0 ) < 0.f )
                w *= -1.f;

            dq_blend = dq_blend + dq * w;
        }

        // Compute animated position
        Vec3 vi = dq_blend.transform( in_verts[p] ).to_vec3();
        out_verts [p] = vi;
        out_verts2[p] = vi;
        // Compute animated normal
        out_normals[p] = dq_blend.rotate( in_normals[p] );
    }
}

// -----------------------------------------------------------------------------

__global__
void transform_arap_dual_quat(Mat3* rots,
                              int nb_verts,
                              const Dual_quat_cu* dual_quat,
                              const float* weights,
                              const int* joints,
                              const int* jpv)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p < nb_verts)
    {
        // compute vertex new position
        //Vec3 axis = {0.f, 0.f, 0.f};

        const int st_j = jpv[2*p  ];
        const int nb_j = jpv[2*p+1];

        int   k0 = -1;
        float w0 = 0.f;
        Dual_quat_cu dq_blend;
        Quat_cu q0;

        if(nb_j != 0)
        {
            k0 = joints [st_j];
            w0 = weights[st_j];
        }else
            dq_blend = Dual_quat_cu::identity();

        if(k0 != -1) dq_blend = dual_quat[k0] * w0;

        int pivot = k0;

        q0 = dual_quat[pivot].rotation();

        for(int j = st_j+1; j < st_j + nb_j; j++)
        {
            const int k = joints [j];
            float w = weights[j];
            const Dual_quat_cu& dq = (k == -1) ? Dual_quat_cu::identity() : dual_quat[k];

            if( dq.rotation().dot( q0 ) < 0.f )
                w *= -1.f;

            dq_blend = dq_blend + dq * w;
        }

        if(rots != 0)
        {
            Mat3 rot = dq_blend.to_transformation().get_mat3();
            rots[p] = rot; /** rots[p]*/;
        }
    }
}

// -----------------------------------------------------------------------------

__global__
void lerp_kernel(const int* vert_to_fit,
                 const Vec3* verts_0,
                 const Vec3* verts_1,
                 float* lerp_factor,
                 Vec3* out_verts,
                 int nb_verts)
{
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_idx < nb_verts)
    {
        const int p = vert_to_fit[thread_idx];
        if( p != -1 ){
            const float f = lerp_factor[p];
            out_verts[p] = verts_0[p] * (1.f - f) + verts_1[p] * f;
        }
    }
}

// -----------------------------------------------------------------------------

/// Clean temporary storage
__global__
void clean_unpacked_normals(Device::Array<Vec3> unpacked_normals)
{
    int n = unpacked_normals.size();
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if( p < n){
        unpacked_normals[p] = Vec3(0.f, 0.f, 0.f);
    }
}

// -----------------------------------------------------------------------------

/// Compute the normal of triangle pi
__device__ Vec3
compute_normal_tri(const EMesh::Prim_idx& pi, const Vec3* prim_vertices) {
    const Point3 va(prim_vertices[pi.a]);
    const Point3 vb(prim_vertices[pi.b]);
    const Point3 vc(prim_vertices[pi.c]);
    return ((vb - va).cross(vc - va)).normalized();
}

// -----------------------------------------------------------------------------

__device__ Vec3
compute_normal_quad(const EMesh::Prim_idx& pi, const Vec3* prim_vertices) {
    const Point3 va(prim_vertices[pi.a]);
    const Point3 vb(prim_vertices[pi.b]);
    const Point3 vc(prim_vertices[pi.c]);
    const Point3 vd(prim_vertices[pi.d]);
    Vec3 vab = (vb - va);
    Vec3 vbc = (vc - vb);
    Vec3 vcd = (vd - vc);
    Vec3 vda = (va - vb);

    return ((vda - vbc).cross(vab - vcd)).normalized();
    //return Vec3(1,1,1).normalized();
}

// -----------------------------------------------------------------------------

/** Assign the normal of each face to each of its vertices
  */
__global__ void
compute_unpacked_normals_tri(const int* faces,
                             Device::Array<EMesh::Prim_idx_vertices> piv,
                             int nb_faces,
                             const Vec3* vertices,
                             Device::Array<Vec3> unpacked_normals,
                             int unpack_factor){
    int n = nb_faces;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p < n){
        EMesh::Prim_idx pidx;
        pidx.a = faces[3*p    ];
        pidx.b = faces[3*p + 1];
        pidx.c = faces[3*p + 2];
        EMesh::Prim_idx_vertices pivp = piv[p];
        Vec3 nm = compute_normal_tri(pidx, vertices);
        int ia = pidx.a * unpack_factor + pivp.ia;
        int ib = pidx.b * unpack_factor + pivp.ib;
        int ic = pidx.c * unpack_factor + pivp.ic;
        unpacked_normals[ia] = nm;
        unpacked_normals[ib] = nm;
        unpacked_normals[ic] = nm;
    }
}

// -----------------------------------------------------------------------------

__global__ void
compute_unpacked_normals_quad(const int* faces,
                              Device::Array< EMesh::Prim_idx_vertices> piv,
                              int nb_faces,
                              int piv_offset,
                              const Vec3* vertices,
                              Device::Array<Vec3> unpacked_normals,
                              int unpack_factor){
    int n = nb_faces;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p < n)
    {
        EMesh::Prim_idx pidx;
        pidx.a = faces[4*p];
        pidx.b = faces[4*p + 1];
        pidx.c = faces[4*p + 2];
        pidx.d = faces[4*p + 3];
        EMesh::Prim_idx_vertices pivp = piv[p + piv_offset];
        Vec3 nm = compute_normal_quad(pidx, vertices);
        int ia = pidx.a * unpack_factor + pivp.ia;
        int ib = pidx.b * unpack_factor + pivp.ib;
        int ic = pidx.c * unpack_factor + pivp.ic;
        int id = pidx.d * unpack_factor + pivp.id;
        unpacked_normals[ia] = nm;
        unpacked_normals[ib] = nm;
        unpacked_normals[ic] = nm;
        unpacked_normals[id] = nm;
        //unpacked_normals[p] = Vec3(pivp.ia, pivp.ib, pivp.ic);
    }

}

// -----------------------------------------------------------------------------

/// Average the normals assigned to each vertex
__global__
void pack_normals( Device::Array<Vec3> unpacked_normals,
                  int unpack_factor,
                  Vec3* normals)
{
    int n = unpacked_normals.size() / unpack_factor;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p < n){
        Vec3 nm = Vec3::zero();
        for(int i = 0; i < unpack_factor; i++){
            nm = nm + unpacked_normals[p * unpack_factor + i];
        }
        normals[p] = nm.normalized();
    }
}

// -----------------------------------------------------------------------------

/// Compute the normals of the mesh using the normal at each face
void compute_normals(const int* tri,
                     const int* quad,
                     Device::Array<EMesh::Prim_idx_vertices> piv,
                     int nb_tri,
                     int nb_quad,
                     const Vec3* vertices,
                     Device::Array<Vec3> unpacked_normals,
                     int unpack_factor,
                     Vec3* out_normals)
{

    const int block_size = 512;
    const int nb_threads_clean = unpacked_normals.size();
    const int grid_size_clean = (nb_threads_clean + block_size - 1) / block_size;
    const int nb_threads_pack = unpacked_normals.size() / unpack_factor;
    const int grid_size_pack = (nb_threads_pack + block_size - 1) / block_size;

    const int nb_threads_compute_tri = nb_tri;
    const int grid_size_compute_tri = (nb_threads_compute_tri + block_size - 1) / block_size;

    const int nb_threads_compute_quad = nb_quad;
    const int grid_size_compute_quad = (nb_threads_compute_quad + block_size - 1) / block_size;

    CUDA_CHECK_KERNEL_SIZE(block_size, grid_size_clean);
    clean_unpacked_normals<<< grid_size_clean, block_size>>>(unpacked_normals);

    CUDA_CHECK_ERRORS();

    if(nb_tri > 0){
#if 1
        compute_unpacked_normals_tri<<< grid_size_compute_tri, block_size>>>
                                   (tri,
                                    piv,
                                    nb_tri,
                                    vertices,
                                    unpacked_normals,unpack_factor);
#else
        compute_unpacked_normals_tri_debug_cpu
                                   (tri,
                                    piv,
                                    nb_tri,
                                    vertices,
                                    unpacked_normals,
                                    unpack_factor,
                                    block_size,
                                    grid_size_compute_tri);
#endif
        CUDA_CHECK_ERRORS();
    }
    if(nb_quad > 0){
        compute_unpacked_normals_quad<<< grid_size_compute_quad, block_size>>>
                                     (quad,
                                      piv,
                                      nb_quad,
                                      nb_tri,
                                      vertices,
                                      unpacked_normals,unpack_factor);
        CUDA_CHECK_ERRORS();
    }

    pack_normals<<< grid_size_pack, block_size>>>( unpacked_normals,
                                                  unpack_factor,
                                                  out_normals);
    CUDA_CHECK_ERRORS();
}

// -----------------------------------------------------------------------------

/// Compute the tangent of triangle pi
__device__ Vec3
compute_tangent_tri(const EMesh::Prim_idx& pi,
                    const EMesh::Prim_idx& upi,
                    const Vec3* prim_vertices,
                    const float* tex_coords)
{
    const Point3 va(prim_vertices[pi.a]);
    const Point3 vb(prim_vertices[pi.b]);
    const Point3 vc(prim_vertices[pi.c]);

    float2 st1 = { tex_coords[upi.b*2    ] - tex_coords[upi.a*2    ],
                   tex_coords[upi.b*2 + 1] - tex_coords[upi.a*2 + 1]};

    float2 st2 = { tex_coords[upi.c*2    ] - tex_coords[upi.a*2    ],
                   tex_coords[upi.c*2 + 1] - tex_coords[upi.a*2 + 1]};

    const Vec3 e1 = vb - va;
    const Vec3 e2 = vc - va;

    float coef = 1.f / (st1.x * st2.y - st2.x * st1.y);
    Vec3 tangent;
    tangent.x = coef * ((e1.x * st2.y)  + (e2.x * -st1.y));
    tangent.y = coef * ((e1.y * st2.y)  + (e2.y * -st1.y));
    tangent.z = coef * ((e1.z * st2.y)  + (e2.z * -st1.y));

    return tangent;
}

// -----------------------------------------------------------------------------

__device__ Vec3
compute_tangent_quad(const EMesh::Prim_idx& pi, const Vec3* prim_vertices) {
    const Point3 va(prim_vertices[pi.a]);
    const Point3 vb(prim_vertices[pi.b]);
    const Point3 vc(prim_vertices[pi.c]);
    const Point3 vd(prim_vertices[pi.d]);
    Vec3 vab = (vb - va);
    Vec3 vbc = (vc - vb);
    Vec3 vcd = (vd - vc);
    Vec3 vda = (va - vb);

    return ((vda - vbc).cross(vab - vcd)).normalized();
    //return Vec3(1,1,1).normalized();
}

// -----------------------------------------------------------------------------

/** Assign the normal of each face to each of its vertices
  */
__global__ void
compute_unpacked_tangents_tri(const int* faces,
                              const int* unpacked_faces,
                              Device::Array<EMesh::Prim_idx_vertices> piv,
                              int nb_faces,
                              const Vec3* vertices,
                              const float* tex_coords,
                              Device::Array<Vec3> unpacked_tangents,
                              int unpack_factor)
{
    int n = nb_faces;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p < n){
        EMesh::Prim_idx pidx;
        pidx.a = faces[3*p    ];
        pidx.b = faces[3*p + 1];
        pidx.c = faces[3*p + 2];
        EMesh::Prim_idx upidx;
        upidx.a = unpacked_faces[3*p    ];
        upidx.b = unpacked_faces[3*p + 1];
        upidx.c = unpacked_faces[3*p + 2];
        EMesh::Prim_idx_vertices pivp = piv[p];
        Vec3 nm = compute_tangent_tri(pidx, upidx, vertices, tex_coords);
        int ia = pidx.a * unpack_factor + pivp.ia;
        int ib = pidx.b * unpack_factor + pivp.ib;
        int ic = pidx.c * unpack_factor + pivp.ic;
        unpacked_tangents[ia] = nm;
        unpacked_tangents[ib] = nm;
        unpacked_tangents[ic] = nm;
    }
}

// -----------------------------------------------------------------------------

__global__ void
compute_unpacked_tangents_quad(const int* faces,
                               const int* unpacked_faces,
                               Device::Array< EMesh::Prim_idx_vertices> piv,
                               int nb_faces,
                               int piv_offset,
                               const Vec3* vertices,
                               const float* tex_coords,
                               Device::Array<Vec3> unpacked_tangents,
                               int unpack_factor)
{
    int n = nb_faces;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p < n){
        EMesh::Prim_idx pidx;
        pidx.a = faces[4*p];
        pidx.b = faces[4*p + 1];
        pidx.c = faces[4*p + 2];
        pidx.d = faces[4*p + 3];
        EMesh::Prim_idx_vertices pivp = piv[p + piv_offset];
        Vec3 nm = compute_tangent_quad(pidx, vertices);
        int ia = pidx.a * unpack_factor + pivp.ia;
        int ib = pidx.b * unpack_factor + pivp.ib;
        int ic = pidx.c * unpack_factor + pivp.ic;
        int id = pidx.d * unpack_factor + pivp.id;
        unpacked_tangents[ia] = nm;
        unpacked_tangents[ib] = nm;
        unpacked_tangents[ic] = nm;
        unpacked_tangents[id] = nm;
        //unpacked_normals[p] = Vec3(pivp.ia, pivp.ib, pivp.ic);
    }

}

// -----------------------------------------------------------------------------

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
                      Vec3* out_tangents)
{

    const int block_size = 512;
    const int nb_threads_clean = unpacked_tangents.size();
    const int grid_size_clean = (nb_threads_clean + block_size - 1) / block_size;
    const int nb_threads_pack = unpacked_tangents.size() / unpack_factor;
    const int grid_size_pack = (nb_threads_pack + block_size - 1) / block_size;

    const int nb_threads_compute_tri = nb_tri;
    const int grid_size_compute_tri = (nb_threads_compute_tri + block_size - 1) / block_size;

    const int nb_threads_compute_quad = nb_quad;
    const int grid_size_compute_quad = (nb_threads_compute_quad + block_size - 1) / block_size;

    CUDA_CHECK_KERNEL_SIZE(block_size, grid_size_clean);
    clean_unpacked_normals<<< grid_size_clean, block_size>>>(unpacked_tangents);

    CUDA_CHECK_ERRORS();

    if(nb_tri > 0){
        compute_unpacked_tangents_tri<<< grid_size_compute_tri, block_size>>>
                                   (tri,
                                    unpacked_tri,
                                    piv,
                                    nb_tri,
                                    vertices,
                                    tex_coords,
                                    unpacked_tangents,unpack_factor);
        CUDA_CHECK_ERRORS();
    }
    if(nb_quad > 0){
        assert(false);
        // TODO: handle quads
        compute_unpacked_tangents_quad<<< grid_size_compute_quad, block_size>>>
                                     (quad,
                                      unpacked_quad,
                                      piv,
                                      nb_quad,
                                      nb_tri,
                                      vertices,
                                      tex_coords,
                                      unpacked_tangents,unpack_factor);
        CUDA_CHECK_ERRORS();
    }

    pack_normals<<< grid_size_pack, block_size>>>(unpacked_tangents,
                                                  unpack_factor,
                                                  out_tangents);
    CUDA_CHECK_ERRORS();
}

// -----------------------------------------------------------------------------

#if 1
/**
 * ARAP energy
 */
__global__
void conservative_smooth_kernel(Vec3* rest_verts,
                                Vec3* in_vertices,
                                Vec3* out_verts,
                                const Vec3* grads,
                                Vec3* d_base_grad,
                                const Mat2* rots,
                                float* sec_ring_lengths,
                                int* sec_ring_list,
                                int* sec_ring_list_offsets,
                                float* fst_ring_cotans,
                                float* fst_ring_lengths,
                                float* fst_ring_angle,
                                const int* fst_ring_list,
                                const int* fst_ring_list_offsets,
                                const float* edge_mvc,
                                const int* vert_to_fit,
                                float force,
                                int nb_verts,
                                const float* smooth_fac,
                                bool use_smooth_fac)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_idx < nb_verts)
    {
        const int i = vert_to_fit[thread_idx];
        if( i == -1 ) return;

        Vec3 grad = grads[i].normalized();

        const Vec3 in_vert = in_vertices[i];

        if(grad.norm() < 0.00001f){
            out_verts[i] = in_vert;
            return;
        }


        Mat3 framet_base = Mat3::coordinate_system( -d_base_grad[i] ).transpose();
        Mat3 frame = Mat3::coordinate_system( -grad );
        Mat3 framet = frame.transpose();

        Vec2 energy_grad(0.f, 0.f);
        const int dep    = fst_ring_list_offsets[2*i  ];
        const int nb_ngb = fst_ring_list_offsets[2*i+1];
        const int end    = dep + nb_ngb;
        for(int n = dep; n < end; n++)
        {
            const int j = fst_ring_list[n];
            Vec3 neigh = in_vertices[ j ];

            Vec3 e = framet * (in_vert - neigh);
            Vec3 e_rest = framet_base * (rest_verts[i] - rest_verts[j]);

            float a_j = rots[j].get_angle();

            Mat3 frame_neigh = Mat3::coordinate_system( -grads[j].normalized() );
            Vec3 e_in_neigh = frame_neigh.transpose() * (in_vert - neigh);

            a_j += Vec2(1.f, 0.f).signed_angle( Vec2(e_in_neigh.y, e_in_neigh.z) );

            a_j -= Vec2(1.f, 0.f).signed_angle( Vec2(e.y, e.z) );

            energy_grad += 4.f * fst_ring_cotans[n] * (Vec2(e.y, e.z)  - 0.5f * ( rots[i] + Mat2::rotate( a_j ) ) * Vec2(e_rest.y, e_rest.z) );
        }

        Vec3 cog_proj = in_vert - (frame.y() * energy_grad.x + frame.z() * energy_grad.y) * 0.002;//0.00001;
        const float u = use_smooth_fac ? smooth_fac[i] : force;
        out_verts[i]  = cog_proj * u + in_vert * (1.f - u);
    }
}


__global__
void updat_rotations(Vec3* rest_verts,
                     Vec3* in_vertices,
                     const Vec3* normals,
                     Vec3* d_base_grad,
                     Mat2* rots,
                     float* sec_ring_lengths,
                     int* sec_ring_list,
                     int* sec_ring_list_offsets,
                     float* fst_ring_cotans,
                     float* fst_ring_lengths,
                     float* fst_ring_angle,
                     const int* fst_ring_list,
                     const int* fst_ring_list_offsets,
                     const float* edge_mvc,
                     const int* vert_to_fit,
                     float force,
                     int nb_verts,
                     const float* smooth_fac,
                     bool use_smooth_fac)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_idx < nb_verts)
    {
        const int p = vert_to_fit[thread_idx];
        if( p == -1 ) return;

        Vec3 n       = normals[p].normalized();
        const Vec3 in_vert = in_vertices[p];

        if(n.norm() < 0.00001f){
            return;
        }

        Mat2 eye = Mat2::identity();

        Mat3 framet_base = Mat3::coordinate_system( -d_base_grad[p] ).transpose();
        Mat3 framet = Mat3::coordinate_system( -n ).transpose();

        int valence = fst_ring_list_offsets[ 2 * p + 1];
        int degree = 0;

        //Eigen::MatrixXd P(2, valence), Q(2, valence);

        // FIXME: use number max of neighbors://////////////////////////////////////////////////
        Vec2 P[10];
        Vec2 Q[10];

        const int dep    = fst_ring_list_offsets[ 2 * p    ];
        const int end    = dep + valence;
        for(int j = dep; j < end; j++)
        {
            const int curr = fst_ring_list[j];
            Vec3 neigh = in_vertices[ curr ];

            Vec3 e = framet * (in_vert - neigh);
            Vec3 e_rest = framet_base * (rest_verts[p] - rest_verts[curr]);

            P[degree] = Vec2(e_rest.y, e_rest.z) * fst_ring_cotans[j];
            Q[degree++] = Vec2(e.y, e.z);
        }

        // Compute the 3 by 3 covariance matrix:
        // actually S = (P * W * Q.t()); W is already considerred in the previous step (P=P*W)
        //Eigen::Matrix2d S = (P * Q.transpose());
        Mat2 S(0.f);
        for(int j = 0; j < degree; j++)
        {
            S(0,0) += P[j].x * Q[j].x; S(0,1) += P[j].x * Q[j].y;
            S(1,0) += P[j].y * Q[j].x; S(1,1) += P[j].y * Q[j].y;
        }

        // Compute the singular value decomposition S = UDV.t
        SVD_2x2<true, true> svd(S); // X = U * D * V.t()

        Mat2 V = svd.matrix_v();
        Mat2 Ut = svd.matrix_u().transpose();

        double det = (V * Ut).det();
        //std::cout << det << std::endl;
        eye(1, 1) = det;	// remember: Eigen starts from zero index

        // V*U.t may be reflection (determinant = -1). in this case, we need to change the sign of
        // column of U corresponding to the smallest singular value (3rd column)
        rots[p] = (V * eye * Ut); //Ri = (V * eye * U.t());

    }
}

#elif 0
__global__
void conservative_smooth_kernel(Vec3* rest_verts,
                                Vec3* in_vertices,
                                Vec3* out_verts,
                                const Vec3* normals,
                                float* sec_ring_lengths,
                                int* sec_ring_list,
                                int* sec_ring_list_offsets,
                                float* fst_ring_cotans,
                                float* fst_ring_lengths,
                                float* fst_ring_angle,
                                const int* fst_ring_list,
                                const int* fst_ring_list_offsets,
                                const float* edge_mvc,
                                const int* vert_to_fit,
                                float force,
                                int nb_verts,
                                const float* smooth_fac,
                                bool use_smooth_fac)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_idx < nb_verts)
    {
        const int p = vert_to_fit[thread_idx];
        if( p == -1 ) return;

        Vec3 n       = normals[p].normalized();
        const Vec3 in_vert = in_vertices[p];

        if(n.norm() < 0.00001f){
            out_verts[p] = in_vert;
            return;
        }

        Vec3 cog(0.f, 0.f, 0.f);
        Vec3 corr(0.f, 0.f, 0.f);
        Vec3 grad(0.f, 0.f, 0.f);

        const int dep    = fst_ring_list_offsets[2*p  ];
        const int nb_ngb = fst_ring_list_offsets[2*p+1];
        const int end    = dep + nb_ngb;
        float sum_areas = 0.f;
        float sum = 0.f;
        float avg_rest_len = 0.f;
        bool inverted = false;
        for(int i = dep; i < end; i++)
        {
            const int curr = fst_ring_list[i];
            const int next = fst_ring_list[ (i+1) >= end  ? dep : i+1 ];
            const float rest_len = fst_ring_lengths[i];

            float mvc = edge_mvc[i];

            Vec3 v1 = in_vertices[ curr ];
            Vec3 v2 = in_vertices[ next ];

            Vec3 normal = (v1 - in_vert).cross(v2 - in_vert).normalized();

            if( n.dot( -normal ) < 0.0f ) inverted = true;

            Vec3 edge0 = in_vertices[curr] - in_vert;
            {

                #if 0
                Vec3 edge1 = in_vertices[next] - in_vert;
                Vec3 tri_n = edge0.cross( edge1 );
                float area = tri_n.norm() / 2.f;
                area *= area;
                corr += tri_n.normalized() * area;
                sum_areas += area;
                #else
                // EDge relax
                corr += edge0 * /*2.f **/ (rest_len-edge0.norm()) / rest_len;
                #endif
                avg_rest_len += rest_len;

                //const float fact = (edge.norm() - rest_len) / rest_len;
                // corr += (n.proj_on_plane( edge )).normalized() * fact;
                //((n.proj_on_plane( edge )).norm() - edge.norm()) / edge.norm()
                //mvc *= 1.f - fact;
            }

            Vec3 cog_neigh = in_vertices[curr];

            Vec3 g = compute_grad_cog( curr, in_vertices, normals, fst_ring_list, fst_ring_list_offsets, edge_mvc, p);
            grad += n.proj_on_plane( g );

            //mvc = rest_len / edge0.norm() ;
            sum += mvc;


            //Vec3 dir = (cog_neigh - in_vert);
            //float len = (dir).norm();
            cog_neigh = n.proj_on_plane(Point3(in_vert), Point3(cog_neigh));
            //cog_neigh = in_vert + (cog_neigh - in_vert).normalized() * len;

            cog = cog + cog_neigh * mvc;
        }

        if( fabs(sum) < 0.00001f ){
            out_verts[p] = in_vert;
            return;
        }

        cog /= sum;
        avg_rest_len /= (float)nb_ngb;
/*
        corr = Vec3(0.f);
        for(int i = dep; i < end; i++){
            const int curr = fst_ring_list[i];
            const float rest_len = fst_ring_lengths[i];

            Vec3 edge0 = in_vertices[curr] - cog;
            corr += edge0 * (rest_len-edge0.norm());
        }
        */

        corr = n.proj_on_plane( corr );

        // EDge relax
        //if( corr.norm() > 0.0000001 )

//        if( !inverted )
//            cog = in_vert + -corr * 0.01;
//        else

        Vec3 vec(0.f);
        #if 0
        if( corr.norm() > 0.0001)
        {
            if(corr.norm() > avg_rest_len*2.f)
                vec = -corr * 0.01 / corr.norm();
            else
                vec = -corr * 0.1;
        }
        cog = in_vert + ((cog - in_vert) * 0.5 + vec)*0.1;
        #else

        //cog = in_vert + ((cog - in_vert) * 0.05 + -corr * 0.01);
        #endif

        // 0.05 + 0.1

        //cog = in_vert + -corr * 0.01;

        //cog = in_vert + ((cog - in_vert) * -2.f + grad) * -0.5 * 0.5;

        //cog = in_vert + ((cog - in_vert) * -2.f + grad) * -0.5 * 0.5;

        //cog = in_vert + (cog - in_vert) * 0.01;
        //corr = n.proj_on_plane( corr );
        //cog = in_vert + (-corr * 0.01 + (cog - in_vert));

        // this force the smoothing to be only tangential :
        Vec3 cog_proj = cog;//n.proj_on_plane(Point3(in_vert), Point3(cog));
        // this is more like a conservative laplacian smoothing
       // const Vec3 cog_proj = cog;

//         cog_proj = in_vert + Vec3::unit_y() * (cog_proj - in_vert).y; //////////// DEBUG project on x axis

        const float u = use_smooth_fac ? smooth_fac[p] : force;
        out_verts[p]  = cog_proj * u + in_vert * (1.f - u);
    }
}

#endif
// -----------------------------------------------------------------------------

template< class T >
__global__ static
void copy_vert_to_fit(const T* d_in,
                      T* d_out,
                      const int* vert_to_fit,
                      int n)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_idx < n){
        const int p = vert_to_fit[thread_idx];
        if(p != -1) d_out[p] = d_in[p];
    }
}

// -----------------------------------------------------------------------------

void conservative_smooth(Vec3* d_input_verts,
                         Vec3* d_verts,
                         Vec3* d_buff_verts,
                         Vec3* d_normals,
                         Vec3* d_base_grad,
                         Mat2* rots,
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
                         bool use_smooth_fac)
{
    assert( nb_iter > 0);
    if(nb_vert_to_fit == 0) return;

    const int block_size = 256;
    // nb_threads == nb_mesh_vertices
    const int nb_threads = nb_vert_to_fit;
    const int grid_size  = (nb_threads + block_size - 1) / block_size;
    Vec3* d_verts_a = d_verts;
    Vec3* d_verts_b = d_buff_verts;

    if(nb_iter > 1){
        // TODO: we could only copy vert_to_fit and there neighbors and avoid
        // copy over every elements.
        Cuda_utils::mem_cpy_dtd(d_buff_verts, d_verts, d_1st_ring_list_offsets.size()/2);
    }


    for(int i = 0; i < nb_iter; i++)
    {
        conservative_smooth_kernel<<<grid_size, block_size>>>(d_input_verts,
                                                              d_verts_a,
                                                              d_verts_b,
                                                              d_normals,
                                                              d_base_grad,
                                                              rots,
                                                              d_2nd_ring_lengths,
                                                              d_2nd_ring_list,
                                                              d_2nd_ring_list_offsets,
                                                              d_1st_ring_cotan,
                                                              d_1st_ring_lengths,
                                                              d_1st_ring_angle,
                                                              d_1st_ring_list.ptr(),
                                                              d_1st_ring_list_offsets.ptr(),
                                                              d_edge_mvc.ptr(),
                                                              d_vert_to_fit,
                                                              strength,
                                                              nb_vert_to_fit,
                                                              smooth_fac,
                                                              use_smooth_fac);
        CUDA_CHECK_ERRORS();
#if 1
        updat_rotations<<<grid_size, block_size>>>(d_input_verts,
                                                   d_verts_b,
                                                   d_normals,
                                                   d_base_grad,
                                                   rots,
                                                   d_2nd_ring_lengths,
                                                   d_2nd_ring_list,
                                                   d_2nd_ring_list_offsets,
                                                   d_1st_ring_cotan,
                                                   d_1st_ring_lengths,
                                                   d_1st_ring_angle,
                                                   d_1st_ring_list.ptr(),
                                                   d_1st_ring_list_offsets.ptr(),
                                                   d_edge_mvc.ptr(),
                                                   d_vert_to_fit,
                                                   strength,
                                                   nb_vert_to_fit,
                                                   smooth_fac,
                                                   use_smooth_fac);
        CUDA_CHECK_ERRORS();
#endif

        Utils::swap(d_verts_a, d_verts_b);
    }

    if(nb_iter % 2 == 1){
        // d_vertices[n] = d_tmp_vertices[n]
        copy_vert_to_fit<<<grid_size, block_size>>>
            (d_buff_verts, d_verts, d_vert_to_fit, nb_threads);
        CUDA_CHECK_ERRORS();
    }
}

// -----------------------------------------------------------------------------

__global__
void laplacian_smooth_kernel(const Vec3* in_vertices,
                             Vec3* output_vertices,
                             const float* fst_ring_cotan_weights,
                             const int* fst_ring_list,
                             const int* fst_ring_list_offsets,
                             const float* factors,
                             bool use_smooth_factors,
                             float strength,
                             int nb_min_neighbours,
                             int n)
{
        int p = blockIdx.x * blockDim.x + threadIdx.x;
        if(p < n)
        {
            Vec3 in_vertex = in_vertices[p];
            Vec3 centroid  = Vec3(0.f, 0.f, 0.f);

            int offset = fst_ring_list_offsets[2*p  ];
            int nb_ngb = fst_ring_list_offsets[2*p+1];
            //if(nb_ngb > nb_min_neighbours)
            {
                float sum = 0.f;
                for(int i = offset; i < offset + nb_ngb; i++){
                    int j = fst_ring_list[i];
                    centroid += in_vertices[j] * fst_ring_cotan_weights[i];
                    sum += fst_ring_cotan_weights[i];
                }

                centroid = centroid * (1.f/sum);

                if(factors != 0){
                    float factor = factors[p];
                    output_vertices[p] = centroid * factor + in_vertex * (1.f-factor);
                } else
                    output_vertices[p] = centroid * strength + in_vertex * (1.f-strength);
            }

            //else                output_vertices[p] = in_vertex;
        }

}

// -----------------------------------------------------------------------------

void laplacian_smooth(Vec3* d_vertices,
                      Vec3* d_tmp_vertices,
                      float* fst_ring_cotan_weights,
                      DA_int d_1st_ring_list,
                      DA_int d_1st_ring_list_offsets,
                      float* factors,
                      bool use_smooth_factors,
                      float strength,
                      int nb_iter,
                      int nb_min_neighbours)
{
    const int block_size = 256;
    // nb_threads == nb_mesh_vertices
    const int nb_threads = d_1st_ring_list_offsets.size() / 2;
    const int grid_size = (nb_threads + block_size - 1) / block_size;
    Vec3* d_vertices_a = d_vertices;
    Vec3* d_vertices_b = d_tmp_vertices;
    for(int i = 0; i < nb_iter; i++)
    {
        laplacian_smooth_kernel<<<grid_size, block_size>>>(d_vertices_a,
                                                           d_vertices_b,
                                                           fst_ring_cotan_weights,
                                                           d_1st_ring_list.ptr(),
                                                           d_1st_ring_list_offsets.ptr(),
                                                           factors,
                                                           use_smooth_factors,
                                                           strength,
                                                           nb_min_neighbours,
                                                           nb_threads);
        CUDA_CHECK_ERRORS();
        Utils::swap(d_vertices_a, d_vertices_b);
    }

    if(nb_iter % 2 == 1){
        // d_vertices[n] = d_tmp_vertices[n]
        copy_arrays<<<grid_size, block_size>>>(d_tmp_vertices, d_vertices, nb_threads);
        CUDA_CHECK_ERRORS();
    }
}

// -----------------------------------------------------------------------------

__global__
void diffusion_kernel(const float* in_values,
                      float* out_values,
                      const int* fst_ring_list,
                      const int* fst_ring_list_offsets,
                      float strength,
                      int nb_vert)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p < nb_vert)
    {
        const float in_val   = in_values[p];
        float centroid = 0.f;

        const int offset = fst_ring_list_offsets[2*p  ];
        const int nb_ngb = fst_ring_list_offsets[2*p+1];

        for(int i = offset; i < (offset + nb_ngb); i++)
        {
            const int j = fst_ring_list[i];
            centroid += in_values[j];
            //centroid = fmaxf(centroid, in_values[j]);
        }

        centroid = centroid * (1.f/nb_ngb);

        out_values[p] = centroid * strength + in_val * (1.f-strength);
    }
}

// -----------------------------------------------------------------------------

void diffuse_values(float* d_values,
                    float* d_values_buffer,
                    DA_int d_1st_ring_list,
                    DA_int d_1st_ring_list_offsets,
                    float strength,
                    int nb_iter)
{

    const int block_size = 256;
    // nb_threads == nb_mesh_vertices
    const int nb_threads = d_1st_ring_list_offsets.size() / 2;
    const int grid_size = (nb_threads + block_size - 1) / block_size;
    float* d_values_a = d_values;
    float* d_values_b = d_values_buffer;
    strength = std::max( 0.f, std::min(1.f, strength));
    for(int i = 0; i < nb_iter; i++)
    {
        diffusion_kernel<<<grid_size, block_size>>>
            (d_values_a, d_values_b, d_1st_ring_list.ptr(), d_1st_ring_list_offsets.ptr(), strength, nb_threads);
        CUDA_CHECK_ERRORS();
        Utils::swap(d_values_a, d_values_b);
    }

    if(nb_iter % 2 == 1){
        // d_vertices[n] = d_tmp_vertices[n]
        copy_arrays<<<grid_size, block_size>>>(d_values_buffer, d_values, nb_threads);
        CUDA_CHECK_ERRORS();
    }
}

// -----------------------------------------------------------------------------

__global__
void unpack_vert_and_normals(const Vec3* packed_vert,
                             const Vec3* packed_normals,
                             const Vec3* packed_tangents,
                             const EMesh::Packed_data* packed_vert_map,
                             Vec3* unpacked_vert,
                             Vec3* unpacked_normals,
                             Vec3* unpacked_tangents,
                             int nb_vert)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p < nb_vert)
    {
        Vec3 pv = packed_vert[p];
        Vec3 pn = packed_normals[p];
        Vec3 pt;
        if(unpacked_tangents != 0)
            pt = packed_tangents[p];

        EMesh::Packed_data d = packed_vert_map[p];
        int idx = d._idx_data_unpacked;
        for(int i = 0; i < d._nb_ocurrence; i++)
        {
            unpacked_vert    [idx+i] = pv;
            unpacked_normals [idx+i] = pn;
            if(unpacked_tangents != 0)
                unpacked_tangents[idx+i] = pt;
        }
    }
}

// -----------------------------------------------------------------------------

__global__
void fill_index(DA_int array)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p < array.size()) array[p] = p;
}

}// END KERNELS NAMESPACE ======================================================
