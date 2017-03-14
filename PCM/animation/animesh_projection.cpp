#include "animesh.hpp"

/**
 * @file animesh_projection.cu
 * @brief implemention of the Animesh class related to deformation
 *
 */


#include "../control/cuda_ctrl.hpp"
#include "toolbox/utils.hpp"
#include "../animation/skeleton.hpp"
#include "../animation/animesh_rig.h"
#include <iostream>
using std::cout;
using std::endl;
using namespace Tbx;
// -----------------------------------------------------------------------------

bool Animesh::is_dynamic_color( EAnimesh::Color_type mesh_col )
{
    return mesh_col == EAnimesh::ANIM_SMOOTH_CONSERVATIVE ||
           mesh_col == EAnimesh::ANIM_SMOOTH_LAPLACIAN ||
           mesh_col == EAnimesh::NORMAL ||
           mesh_col == EAnimesh::GRAD_POTENTIAL ||
           mesh_col == EAnimesh::VERTICES_STATE ||
           mesh_col == EAnimesh::EDGE_STRESS ||
           mesh_col == EAnimesh::AREA_STRESS ||
           mesh_col == EAnimesh::GAUSS_CURV;
}

// -----------------------------------------------------------------------------

void Animesh::update_base_potential()
{

	cout<<"update_base_potential"<<endl;
    //if(!do_update_potential) return;

    //Timer time;
    //time.start();
    //const int nb_verts = d_input_vertices.size();
    //const int block_size = 256;
    //const int grid_size =
    //        (nb_verts + block_size - 1) / block_size;

    //_skel->reset();

    //Animesh_kers::compute_base_potential<<<grid_size, block_size>>>
    //    (_skel->skel_id(), d_input_vertices.ptr(), nb_verts, d_base_potential.ptr(), d_base_gradient.ptr());

    //compute_mvc();///////

    //CUDA_CHECK_ERRORS();

    //_skel->unreset();

    //if(mesh_color == EAnimesh::BASE_POTENTIAL)
    //    set_colors(mesh_color);

    //std::cout << "Update base potential in " << time.stop() << " sec" << std::endl;
}

// -----------------------------------------------------------------------------

void Animesh::update_bone_samples(EBone::Id bone_id,
                                  const std::vector<Vec3>& nodes,
                                  const std::vector<Vec3>& n_nodes)
{
	cout<<"Animesh::update_bone_samples"<<endl;
    //const float rad_hrbf = _skel->get_hrbf_radius( bone_id );

    //// We update nodes in bones
    //Bone_hrbf*  hrbf_bone = new Bone_hrbf(rad_hrbf);
    //HermiteRBF& hrbf      = hrbf_bone->get_hrbf();

    //std::cout << "HRBF interpolation \n";
    //std::cout << "Solving for " << nodes.size() << " nodes" << std::endl;

    //// Solve/compute compute HRBF weights
    //Timer t;
    //t.start();
    //hrbf.init_coeffs(nodes, n_nodes);
    //std::cout << "Solved in: " << t.stop() << " sec \n" << std::endl;

    //hrbf_bone->set_radius( _skel->get_bone( bone_id)->radius() );
    //_skel->set_bone(bone_id, hrbf_bone);

    //update_base_potential();
}

// -----------------------------------------------------------------------------

void Animesh::compute_tangents(const std::vector<Tbx::Vec3>& vertices, std::vector<Tbx::Vec3>& tangents)
{
		cout<<"Animesh::compute_tangents"<<endl;
    // TODO: optimize the tangent computation by precomputed coefficients
    // only related top connexity and uv coords.
    //int* map_tri   = 0;
    //int* map_quad  = 0;
    //Vec2* map_tex = 0;
    //if(_mesh->get_nb_tris()  > 0) _mesh->_mesh_gl._index_bo_tri. cuda_map_to(map_tri);
    //if(_mesh->get_nb_quads() > 0) _mesh->_mesh_gl._index_bo_quad.cuda_map_to(map_quad);
    //_mesh->_mesh_gl._tex_bo.cuda_map_to(map_tex);

    //Animesh_kers::compute_tangents(d_input_tri.ptr(),
    //                               d_input_quad.ptr(),
    //                               map_tri,
    //                               map_quad,
    //                               d_piv,
    //                               _mesh->get_nb_tris(),
    //                               _mesh->get_nb_quads(),
    //                               vertices,
    //                               (float*)map_tex,
    //                               d_unpacked_tangents,
    //                               _mesh->_mesh_he.get_max_faces_per_vertex(),
    //                               tangents);

    //if(_mesh->get_nb_tris()  > 0) _mesh->_mesh_gl._index_bo_tri. cuda_unmap();
    //if(_mesh->get_nb_quads() > 0) _mesh->_mesh_gl._index_bo_quad.cuda_unmap();
    //_mesh->_mesh_gl._tex_bo.cuda_unmap();
    //CUDA_CHECK_ERRORS();
}

// -----------------------------------------------------------------------------

void Animesh::compute_normals(const std::vector<Tbx::Vec3>& vertices, std::vector<Tbx::Vec3>& normals)
{
	cout<<"Animesh compute_normals"<<endl;
   /* if(_mesh->get_nb_faces() > 0)
    {
        Animesh_kers::compute_normals(d_input_tri.ptr(),
                                      d_input_quad.ptr(),
                                      d_piv,
                                      _mesh->get_nb_tris(),
                                      _mesh->get_nb_quads(),
                                      vertices,
                                      d_unpacked_normals,
                                      _mesh->_mesh_he.get_max_faces_per_vertex(),
                                      normals);
    }
    CUDA_CHECK_ERRORS();*/
}

// -----------------------------------------------------------------------------


void Animesh::smooth_mesh(Vec3* output_vertices,
                          Vec3* verts_buffer,
                          float* factors,
                          int nb_iter,
                          bool local_smoothing)
{
		cout<<"Animesh::smooth_mesh"<<endl;

    //if(nb_iter == 0) return;
    //Animesh_kers::laplacian_smooth(output_vertices,
    //                               verts_buffer,
    //                               hd_1st_ring_cotan.d_ptr(),
    //                               d_1st_ring_list,
    //                               d_1st_ring_list_offsets,
    //                               factors,
    //                               local_smoothing,
    //                               smooth_force_a,
    //                               nb_iter,
    //                               3);

}

// -----------------------------------------------------------------------------

void Animesh::fit_mesh(int nb_vert_to_fit,
                       int* d_vert_to_fit,
                       bool full_eval,
                       bool smooth_fac_from_iso,
                       Vec3* d_vertices,
                       int nb_steps,
                       float smooth_strength)
{
	cout<<"Animesh::fit_mesh"<<endl;
    //if(nb_vert_to_fit == 0) return;

    //const int nb_vert    = nb_vert_to_fit;
    //const int block_size = 16;
    //const int grid_size  = (nb_vert + block_size - 1) / block_size;

    //CUDA_CHECK_ERRORS();
    //CUDA_CHECK_KERNEL_SIZE(block_size, grid_size);

    //Animesh_kers::match_base_potential
    //    <<<grid_size, block_size >>>    //hd_verts_3drots.update_device_mem();
    //    (_skel->skel_id(),
    //     full_eval,
    //     smooth_fac_from_iso,
    //     d_vertices,
    //     d_input_vertices.ptr(),
    //     _skel->d_transfos(),
    //     d_base_potential.ptr(),
    //     d_ssd_normals.ptr(),
    //     hd_gradient.d_ptr() /*d_ssd_normals.ptr()*/,
    //     d_nearest_bone_in_device_mem.ptr(),
    //     d_vertices_nearest_bones.ptr(),
    //     d_nearest_joint_in_device_mem.ptr(),
    //     d_smooth_factors_conservative.ptr(),
    //     d_smooth_factors_laplacian.ptr(),
    //     d_vert_to_fit,
    //     nb_vert_to_fit,
    //     /* (do_tune_direction && !full_eval), */
    //     (unsigned short)nb_steps,
    //     Cuda_ctrl::_debug._collision_threshold,
    //     Cuda_ctrl::_debug._step_length,
    //     Cuda_ctrl::_debug._potential_pit,
    //     (int*)d_vertices_state.ptr(),
    //     smooth_strength,
    //     Cuda_ctrl::_debug._collision_depth,
    //     Cuda_ctrl::_debug._slope_smooth_weight,
    //     Cuda_ctrl::_debug._raphson,
    //     d_flip_propagation.ptr());

    //CUDA_CHECK_ERRORS();

    // copy device mem gradient into host mem
    //hd_gradient.update_host_mem(); ///////////DEBUG will slow down animation
}

// -----------------------------------------------------------------------------

void Animesh::fit_mesh_std(int nb_vert_to_fit,
                           int* d_vert_to_fit,
                           bool full_eval,
                           bool smooth_fac_from_iso,
                           Vec3* d_vertices,
                           int nb_steps,
                           float smooth_strength)
{
		cout<<"Animesh::fit_mesh_std"<<endl;
    //if(nb_vert_to_fit == 0) return;

    //const int nb_vert    = nb_vert_to_fit;
    //const int block_size = 16;
    //const int grid_size  = (nb_vert + block_size - 1) / block_size;

    //CUDA_CHECK_ERRORS();
    //CUDA_CHECK_KERNEL_SIZE(block_size, grid_size);

    //Animesh_kers::match_base_potential_standard
    //    <<<grid_size, block_size >>>    //hd_verts_3drots.update_device_mem();
    //    (_skel->skel_id(),
    //     full_eval,
    //     smooth_fac_from_iso,
    //     d_vertices,
    //     d_input_vertices.ptr(),
    //     _skel->d_transfos(),
    //     d_base_potential.ptr(),
    //     d_ssd_normals.ptr(),
    //     hd_gradient.d_ptr() /*d_ssd_normals.ptr()*/,
    //     d_nearest_bone_in_device_mem.ptr(),
    //     d_vertices_nearest_bones.ptr(),
    //     d_nearest_joint_in_device_mem.ptr(),
    //     d_smooth_factors_conservative.ptr(),
    //     d_smooth_factors_laplacian.ptr(),
    //     d_vert_to_fit,
    //     nb_vert_to_fit,
    //     /* (do_tune_direction && !full_eval), */
    //     (unsigned short)nb_steps,
    //     Cuda_ctrl::_debug._collision_threshold,
    //     Cuda_ctrl::_debug._step_length,
    //     Cuda_ctrl::_debug._potential_pit,
    //     (int*)d_vertices_state.ptr(),
    //     smooth_strength,
    //     Cuda_ctrl::_debug._collision_depth,
    //     Cuda_ctrl::_debug._slope_smooth_weight,
    //     Cuda_ctrl::_debug._raphson,
    //     d_flip_propagation.ptr());

    //CUDA_CHECK_ERRORS();

    // copy device mem gradient into host mem
    //hd_gradient.update_host_mem(); ///////////DEBUG will slow down animation
}

// -----------------------------------------------------------------------------

void Animesh::compute_blended_dual_quat_rots()
{
	
	cout<<"Animesh::compute_blended_dual_quat_rots"<<endl;
    //const int block_size = 16;
    //const int grid_size  = (_mesh->get_nb_vertices() + block_size - 1) / block_size;

    //{
	const std::vector<Dual_quat_cu>& tr = _skel->d_dual_quat();
	Animesh_kers::transform_arap_dual_quat(
		hd_verts_3drots,
		_mesh->get_nb_vertices(),
		tr,
		h_weights);
}

// -----------------------------------------------------------------------------

void Animesh::geometric_deformation(EAnimesh::Blending_type type,
                                    const std::vector<Tbx::Vec3>& d_in,
                                    std::vector<Tbx::Vec3>& out,
                                    std::vector<Tbx::Vec3>& out2,
                                    const void* transfos)
{
	cout<<"Animesh::geometric_deformation"<<endl;

    //const int block_size = 16;
    //const int grid_size  = (d_in.size() + block_size - 1) / block_size;


    if(type == EAnimesh::DUAL_QUAT_BLENDING)
    {
        const std::vector<Dual_quat_cu>& tr = (transfos != 0) ? *(std::vector<Dual_quat_cu>*)transfos : _skel->d_dual_quat();
        Animesh_kers::transform_dual_quat
            (d_in,
             d_base_gradient,
             d_in.size(),
             out,
             out2,
             d_ssd_normals,
             tr,
			 h_weights);
    }
    else if(type == EAnimesh::MATRIX_BLENDING )
    {
        const std::vector<Transfo>& tr = (transfos != 0) ? *(std::vector<Transfo>*)transfos : _skel->d_transfos();
        Animesh_kers::transform_SSD
            (d_in,
             d_base_gradient,
             d_in.size(),
             out,
             out2,
             d_ssd_normals,
             tr,
			 h_weights	);
    }
    else if(type == EAnimesh::RIGID )
    {
        const std::vector<Transfo>& tr = (transfos != 0) ? *(std::vector<Transfo>*)transfos : _skel->d_transfos();
        Animesh_kers::transform_rigid
                 (d_in,
                  d_base_gradient,
                  d_in.size(),
                  out,
                  out2,
                  d_ssd_normals,
                  tr,
                  /*hd_verts_3drots.d_ptr()*/0,
                  &d_vertices_nearest_bones[0]);
    }

    compute_blended_dual_quat_rots();

}

// -----------------------------------------------------------------------------

void Animesh::ssd_lerp(Vec3* out_verts)
{
	cout<<"Animesh::ssd_lerp"<<endl;
    //const int nb_vert_to_fit = d_vert_to_fit_base.size();
    //if( nb_vert_to_fit == 0) return;
    //const int block_size = 16;
    //const int grid_size  = (nb_vert_to_fit + block_size - 1) / block_size;

    //Animesh_kers::lerp_kernel<<<grid_size, block_size>>>
    //        (d_vert_to_fit_base.ptr(),
    //         out_verts,
    //         (Vec3*)d_ssd_vertices.ptr(),
    //         hd_ssd_interpolation_factor.d_ptr(),
    //         out_verts,
    //         nb_vert_to_fit);

    //CUDA_CHECK_ERRORS();
}

// -----------------------------------------------------------------------------

//#include "toolbox/timer.hpp"
//#include <QTime>

void Animesh::transform_vertices(EAnimesh::Blending_type type, bool refresh)
{
	 cout<<"Animesh::transform_vertices"<<endl;

    using namespace Cuda_ctrl;
//    //Timer time;
//    //time.start();
//
    const int nb_vert    = d_input_vertices.size();

     std::vector<Tbx::Vec3>& ssd_verts    = d_ssd_vertices;
     std::vector<Tbx::Vec3>& out_verts    = hd_output_vertices;
     std::vector<Tbx::Vec3>& out_normals  = d_output_normals;
     std::vector<Tbx::Vec3>& out_tangents = d_output_tangents;
//    Vec3* ssd_normals  = (Vec3*)d_ssd_normals.ptr();
//

//
    geometric_deformation(type, d_input_vertices, out_verts, ssd_verts);

    ///////////////////////////////
    if( !refresh )
    {
        std::vector<Transfo> transfos_incr( _skel->nb_joints() );
        for(int i = 0; i < _skel->nb_joints(); ++i) {
            Transfo tr_prev = _skel->_kinec->get_prev_transfo( i );
            Transfo tr = _skel->get_transfo( i );
            transfos_incr[i]=  tr * tr_prev.fast_invert();
        }

        std::vector<Vec3> dummy(hd_prev_output_vertices);
        geometric_deformation(EAnimesh::RIGID, dummy, hd_prev_output_vertices, hd_prev_output_vertices, &transfos_incr[0]);

    }
//    ///////////////////////////////
//

//
    compute_normals(out_verts, d_ssd_normals);
//
	out_normals = d_ssd_normals;
//

//
    //compute_normals(out_verts, out_normals);
//
    if(_mesh->_has_tex_coords && _mesh->_has_bumpmap)
        compute_tangents(out_verts, out_normals);
    else
        out_tangents.clear();
//
//    // Fill the buffer objects with the deformed vertices.
//    // vertex with multiple texture coordinates are duplicated
	update_opengl_buffers(nb_vert,
		out_verts,
		out_normals,
		out_tangents,
		&(_mesh->_mesh_gl._vbo),
		&(_mesh->_mesh_gl._normals_bo),
		&(_mesh->_mesh_gl._tangents_bo));

	hd_prev_output_vertices =  hd_output_vertices;
	hd_tmp_vertices = hd_output_vertices;
//
    //std::cout << "transform :" << (float)time.stop()/1000.f << std::endl;$
}

// -----------------------------------------------------------------------------

void Animesh::transform_vertices_incr(bool refresh)
{
	cout<<"Animesh::transform_vertices_incr"<<endl;
    assert(false);
    return;
}

// -----------------------------------------------------------------------------

void Animesh::reset_vertices()
{
	 cout<<"Animesh::reset_vertices"<<endl;
//    using namespace Cuda_ctrl;
//
//    //////////////
//    // Reset verts
//    hd_output_vertices.     copy_from_hd( d_input_vertices );
//    hd_tmp_vertices.        copy_from_hd( d_input_vertices );
//    hd_prev_output_vertices.copy_from_hd( d_input_vertices );
//
//
//    //////////////
//    // Reset ARAP rotations
//    hd_verts_3drots.fill( Mat3::identity() );
//    hd_verts_3drots.update_device_mem();
//
//    hd_verts_rots.fill( Mat2::identity() );
//    hd_verts_rots.update_device_mem();
//
//    //////////////
//    // Base potential
//    update_base_potential();
//    hd_gradient.copy_from_hd( d_base_gradient  );
//
//    /////////////////
//    // Compute normals tangent colors and refill all VBOS
//    const int nb_vert  = d_input_vertices.size();
//    Vec3* out_verts    = (Vec3*)hd_output_vertices.d_ptr();
//    Vec3* out_normals  = (Vec3*)d_output_normals.ptr();
//    Vec3* out_tangents = (Vec3*)d_output_tangents.ptr();
//
//    compute_normals(out_verts, out_normals);
//
//    if(_mesh->_has_tex_coords && _mesh->_has_bumpmap)
//        compute_tangents(out_verts, out_normals);
//    else
//        out_tangents = 0;
//
//    // Fill the buffer objects with the deformed vertices.
//    // vertex with multiple texture coordinates are duplicated
//    update_opengl_buffers(nb_vert,
//                          out_verts,
//                          out_normals,
//                          out_tangents,
//                          &(_mesh->_mesh_gl._vbo),
//                          &(_mesh->_mesh_gl._normals_bo),
//                          &(_mesh->_mesh_gl._tangents_bo));
//
//#if !defined(NDEBUG) || 1
//    if( is_dynamic_color(mesh_color) )
//        set_colors(mesh_color);
//#endif
}
