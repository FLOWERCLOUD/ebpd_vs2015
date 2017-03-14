#include "animesh.hpp"

#include "toolbox/std_utils/vector.hpp"
#include "toolbox/maths/vec3.hpp"
#include "toolbox/gl_utils/glbuffer_object.hpp"
//#include "animesh_kers_colors.hpp"
//#include "animesh_kers.hpp"
#include "../global_datas/macros.hpp"
//#include "distance_field.hpp"
//#include "auto_rig.hpp"
#include "../control/color_ctrl.hpp"
#include "../animation/skeleton.hpp"
#include "../meshes/mesh_utils.hpp"
#include "../meshes/mesh.hpp"
#include "../animation/animesh_enum.hpp"
#include "animesh_colors.h"
// -----------------------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <limits>
#include <cmath>

// -----------------------------------------------------------------------------

namespace Cuda_ctrl{
    extern Color_ctrl _color;
}


using namespace Tbx;
using std::cout;
using std::endl;
// -----------------------------------------------------------------------------

float distsqToSeg(const Point3& v, const Point3& p1, const Point3& p2);

// -----------------------------------------------------------------------------

Animesh::Animesh(Mesh* m_, Skeleton* s_) :
    _mesh(m_), _skel(s_),
    _do_bone_deform(s_->nb_joints(), true),
    mesh_color(EAnimesh::BASE_POTENTIAL),
    do_implicit_skinning(false),
    do_smooth_mesh(false),
    do_local_smoothing(true),
    do_interleave_fitting(false),
    do_update_potential(true),
    smoothing_iter(7),
    diffuse_smooth_weights_iter(6),
    smooth_force_a(0.5f),
    d_input_smooth_factors(_mesh->get_nb_vertices(), 0.f),
    d_smooth_factors_conservative(_mesh->get_nb_vertices(), 0.f),
    d_smooth_factors_laplacian(_mesh->get_nb_vertices(), 0.f),
    d_input_vertices(_mesh->get_nb_vertices()),
    d_1st_ring_lengths(_mesh->get_size_1st_ring_list()),
    hd_1st_ring_mvc(_mesh->get_size_1st_ring_list() ),
    hd_1st_ring_edges( _mesh->get_size_1st_ring_list() ),
    d_vertices_state(_mesh->get_nb_vertices()),
    d_vertices_states_color(EAnimesh::NB_CASES),
    h_junction_radius(_skel->nb_joints()),
//    d_input_normals(m->get_nb_vertices()),
    hd_prev_output_vertices(_mesh->get_nb_vertices()),
    hd_output_vertices(_mesh->get_nb_vertices()),
    hd_tmp_vertices(_mesh->get_nb_vertices()),
    d_output_normals(_mesh->get_nb_vertices()),
    d_output_tangents(_mesh->get_nb_vertices()),
    d_ssd_normals(_mesh->get_nb_vertices()),
    d_ssd_vertices(_mesh->get_nb_vertices()),
    hd_gradient(_mesh->get_nb_vertices()),
    hd_velocity(_mesh->get_nb_vertices()),
    d_input_tri(_mesh->get_nb_tris()*3),
    d_input_quad(_mesh->get_nb_quads()*4),
    hd_verts_bone_proj( _mesh->get_nb_vertices() ),
    d_1st_ring_list(_mesh->get_size_1st_ring_list()),
    d_1st_ring_list_offsets(2 * _mesh->get_nb_vertices()),
    d_joints(), d_weights(),
    d_jpv(2 * _mesh->get_nb_vertices()),
    h_weights(_mesh->get_nb_vertices()),
    d_base_potential(_mesh->get_nb_vertices()),
    d_base_gradient(_mesh->get_nb_vertices()),
    d_piv(_mesh->get_nb_faces()),
    d_packed_vert_map(_mesh->get_nb_vertices()),
    d_unpacked_normals(_mesh->get_nb_vertices() * _mesh->_mesh_he.get_max_faces_per_vertex()),
    d_unpacked_tangents(_mesh->get_nb_vertices() * _mesh->_mesh_he.get_max_faces_per_vertex()),
    d_rot_axis(_mesh->get_nb_vertices()),
    hd_ssd_interpolation_factor(_mesh->get_nb_vertices(), 0.f),
    hd_edge_list( _mesh->get_nb_edges() ),
    hd_edge_bending( _mesh->get_nb_edges() ),
    hd_sum_angles( _mesh->get_nb_vertices() ),
    hd_map_verts_to_free_vertices( _mesh->get_nb_vertices() ),
    hd_edge_lengths(_mesh->get_nb_edges()),
    hd_tri_areas(_mesh->get_nb_tris()),
    h_vertices_nearest_bones(_mesh->get_nb_vertices()),
    d_vertices_nearest_bones(_mesh->get_nb_vertices()),
    nb_vertices_by_bones(_skel->get_bones().size()),
    h_vertices_nearest_joint(_mesh->get_nb_vertices()),
    d_vertices_nearest_joint(_mesh->get_nb_vertices()),
    //d_nearest_bone_in_device_mem(_mesh->get_nb_vertices()),
    //d_nearest_joint_in_device_mem(_mesh->get_nb_vertices()),
    h_nearest_bone_dist(_mesh->get_nb_vertices()),
    hd_nearest_joint_dist(_mesh->get_nb_vertices()),
    vmap_old_new(_mesh->get_nb_vertices()),
    vmap_new_old(_mesh->get_nb_vertices()),
    d_rear_verts(_mesh->get_nb_vertices()),
    h_half_angles(_skel->nb_joints()),
    d_half_angles(_skel->nb_joints()),
    h_orthos(_skel->nb_joints()),
    d_orthos(_skel->nb_joints()),
    d_flip_propagation(_mesh->get_nb_vertices()),
    d_grad_transfo(_mesh->get_nb_vertices()),
    h_vert_buffer(_mesh->get_nb_vertices()),
    d_vert_buffer(_mesh->get_nb_vertices()),
    d_vert_buffer_2(_mesh->get_nb_vertices()),
    d_vals_buffer(_mesh->get_nb_vertices()),
    hd_displacement(_mesh->get_nb_vertices(), 0.f)
{

    ////////////////////////////////////////////////////////////////////////////////// HACK: DEBUG:
    //_do_bone_deform[_skel->root()] = false;
#if 0
    // HACK ARMADILLO
    _do_bone_deform[41] = false;
    _do_bone_deform[24] = false;
    _do_bone_deform[20] = false;
    _do_bone_deform[21] = false;
    _do_bone_deform[25] = false;
#endif
    ////////////////////////////////////////////////////////////////////////////////// HACK: DEBUG:

    ///////////
    // Compute nearest bone and nearest joint from each vertices
    clusterize( EAnimesh::EUCLIDEAN );

    init_smooth_factors(d_input_smooth_factors);
    set_default_bones_radius();
    compute_smooth_parts();

    int nb_vert = _mesh->get_nb_vertices();
    std::vector<EAnimesh::Vert_state> h_vert_state(nb_vert);
    for (int i = 0; i < nb_vert; ++i)
    {
        vmap_old_new[i] = i;
        vmap_new_old[i] = i;
        h_vert_state[i] = EAnimesh::NOT_DISPLACED;
    }

    d_vertices_state = h_vert_state;

    ////////////////
    // Define color used to debug the vertices fitting on implicit surfaces
    d_vertices_states_color[EAnimesh::POTENTIAL_PIT      ]= Vec4(1.f, 0.f, 1.f, 0.99f); // purple
    d_vertices_states_color[EAnimesh::GRADIENT_DIVERGENCE]= Vec4(1.f, 0.f, 0.f, 0.99f); // red
    d_vertices_states_color[EAnimesh::NB_ITER_MAX        ]= Vec4(0.f, 0.f, 1.f, 0.99f); // blue
    d_vertices_states_color[EAnimesh::NOT_DISPLACED      ]= Vec4(1.f, 1.f, 0.f, 0.99f); // yellow
    d_vertices_states_color[EAnimesh::FITTED             ]= Vec4(0.f, 1.f, 0.f, 0.99f); // green
    d_vertices_states_color[EAnimesh::OUT_VERT           ]= Vec4(1.f, 1.f, 1.f, 0.99f); // white
    d_vertices_states_color[EAnimesh::NORM_GRAD_NULL     ]= Vec4(0.f, 0.f, 0.f, 0.99f); // black
    d_vertices_states_color[EAnimesh::PLANE_CULLING      ]= Vec4(0.f, 1.f, 1.f, 0.99f); // cyan
    d_vertices_states_color[EAnimesh::CROSS_PROD_CULLING ]= Vec4(0.5f, 0.5f, 0.5f, 0.99f); // grey

    // Not mandatory but it is supposed to accelerate a little bit animation
    // when activated
#if 0
    // Modified mesh in order to vertices to be ordered by bone cluster
    regroup_mesh_vertices(*m,
                          d_vertices_nearest_bones,
                          h_vertices_nearest_bones,
                          d_vertices_nearest_joint,
                          h_vertices_nearest_joint,
                          vmap_old_new);

    Color cl = Cuda_ctrl::_color.get(Color_ctrl::MESH_POINTS);
    m->set_point_color_bo(cl.r, cl.g, cl.b, cl.a);
#endif

    // Fill '_mesh' attributes in device memory
    copy_mesh_data(*_mesh);


    init_rigid_ssd_weights();
    init_ssd_interpolation_weights();


    update_base_potential();
    //hd_gradient.update_device_mem(); // Allocate device
    //hd_gradient = d_base_gradient ; // init device
    //hd_gradient.update_host_mem(); // copy to host

    // this needs update_base_potential() to be called first because we need
    // d_base_gradient to up to date. '_mesh' must be ready as well
    //compute_mvc();// <- done by update potential
    //

    // this call needs base gradient to be initialized
    init_sum_angles();

    init_cotan_weights();

    init_vertex_bone_proj();

    hd_verts_rots.resize(_mesh->get_nb_vertices(), Mat2::identity());
    hd_verts_3drots.resize(_mesh->get_nb_vertices(), Mat3::identity());

    set_colors(mesh_color);
}

// -----------------------------------------------------------------------------

void Animesh::init_vertex_bone_proj()
{
    for(int i = 0; i < _mesh->get_nb_vertices(); ++i)
    {
        EBone::Id bone_id = h_vertices_nearest_bones[i];
        Vec3 pos = _mesh->get_vertex( i );
        hd_verts_bone_proj[i] = _skel->get_bone(bone_id)->project( pos.to_point3() ).to_vec3();
    }

    //hd_verts_bone_proj.update_device_mem();
}

// -----------------------------------------------------------------------------

void Animesh::init_verts_per_bone()
{
    std::vector< std::vector<Vec3> >& vertices = h_input_verts_per_bone;
    std::vector< std::vector<Vec3> >& normals  = h_input_normals_per_bone;
    std::vector< std::vector<int>  >& vert_ids = h_verts_id_per_bone;
    vertices.clear();
    normals. clear();
    vert_ids.clear();
    vertices.resize(_skel->nb_joints());
    vert_ids.resize(_skel->nb_joints());
    normals. resize(_skel->nb_joints());

    for(int i = 0; i < _mesh->get_nb_vertices(); i++)
    {
        int nearest = h_vertices_nearest_bones[i];

        if(_mesh->is_disconnect(i))
            continue;

        const Vec3 vert = _mesh->get_vertex(i);
        const Vec3 norm = _mesh->get_normal(i);

        vertices[nearest].push_back( Vec3(vert.x,  vert.y,  vert.z)              );
        normals [nearest].push_back( Vec3(norm.x,  norm.y,  norm.z).normalized() );
        vert_ids[nearest].push_back( i );
    }
}

// -----------------------------------------------------------------------------

Animesh::~Animesh()
{
	if (_vbo_input_vert)
		delete _vbo_input_vert;
	if(_nbo_input_normal)
		delete _nbo_input_normal;
}

// -----------------------------------------------------------------------------

void Animesh::init_vert_to_fit()
{
    assert(hd_ssd_interpolation_factor.size() > 0);

    int nb_vert = _mesh->get_nb_vertices();
    std::vector<int> h_vert_to_fit_base;
    h_vert_to_fit_base.reserve(nb_vert);
    int acc = 0;
    for (int i = 0; i < nb_vert; ++i)
    {
        if( !_mesh->is_disconnect(i) && hd_ssd_interpolation_factor[i] < (1.f - 0.00001f) ){
            h_vert_to_fit_base.push_back( i );
            acc++;
        }
    }

    d_vert_to_fit.     resize(acc);
    d_vert_to_fit_base.resize(acc);

    d_vert_to_fit_buff_scan.resize(acc+1);
    d_vert_to_fit_buff.resize(acc);
    h_vert_to_fit_buff.resize(acc);
    h_vert_to_fit_buff_2.resize(acc);

    d_vert_to_fit_base = h_vert_to_fit_base;
    d_vert_to_fit  = h_vert_to_fit_base;
}

// -----------------------------------------------------------------------------

void Animesh::copy_mesh_data(const Mesh& a_mesh)
{
    const int nb_vert = a_mesh.get_nb_vertices();

    const EMesh::Packed_data* d = a_mesh.get_packed_vert_map();
    memcpy( &d_packed_vert_map[0], d, nb_vert);
    if(a_mesh._has_tex_coords)
    {
        Vec3* t = (Vec3*)a_mesh._mesh_attr._tangents.data();
         memcpy( &d_output_tangents[0], t, nb_vert);
    }

    std::vector<Vec3 > input_vertices(nb_vert);
    std::vector<Vec3>   input_normals (nb_vert);
    std::vector<bool>      flip_prop     (nb_vert);
    for(int i = 0; i < nb_vert; i++)
    {
        Point3  pos = a_mesh.get_vertex(i).to_point3();

        input_vertices[i] = pos;
        input_normals [i] = a_mesh.get_normal(i);
        flip_prop     [i] = false;
        hd_velocity   [i] = Vec3::zero();
    }
    //hd_velocity.update_device_mem();

    int n_faces = a_mesh.get_nb_faces();
    std::vector<EMesh::Prim_idx_vertices> h_piv(n_faces);
    for(int i = 0; i < n_faces; i++){
        h_piv[i] = a_mesh.get_piv(i);
    }
    d_piv = h_piv;

    hd_prev_output_vertices =  input_vertices ;
    hd_tmp_vertices         = input_vertices ;
    hd_output_vertices      = input_vertices ;
    d_input_vertices        = input_vertices ;

//    d_input_normals.copy_from(input_normals);
    d_ssd_normals           = input_normals;
	d_base_gradient         = d_ssd_normals;
    d_flip_propagation      = flip_prop;

    _vbo_input_vert   = new GlBuffer_obj<Vec3>( _mesh->get_vbos_size(), GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    _nbo_input_normal = new GlBuffer_obj<Vec3>( _mesh->get_vbos_size(), GL_ARRAY_BUFFER, GL_STATIC_DRAW);

    _vbo_input_vert->bind();
    _nbo_input_normal->bind();

    update_opengl_buffers(nb_vert, d_input_vertices, d_ssd_normals,std::vector<Vec3>() ,
                          _vbo_input_vert, _nbo_input_normal, 0);

    _vbo_input_vert->unbind();
    _nbo_input_normal->unbind();

    ///////////
    // Copy 1st ring neigborhood list
    std::vector<int> h_1st_ring_list(a_mesh.get_size_1st_ring_list());
    std::vector<int> h_1st_ring_list_offsets( 2*nb_vert );
    for(int i = 0; i < a_mesh.get_size_1st_ring_list(); i++){
        h_1st_ring_list[i] = a_mesh.get_1st_ring(i);
    }


    for(int idx = 0; idx < nb_vert; idx++)
    {
        assert( idx < nb_vert );

        int off = a_mesh.get_1st_ring_offset( 2 * idx     );
        int nb  = a_mesh.get_1st_ring_offset( 2 * idx + 1 );
        h_1st_ring_list_offsets[ 2 * idx    ] = off;
        h_1st_ring_list_offsets[ 2 * idx + 1] = nb;

        // Copy index of edges and triangles incident at each vertex
        const std::vector<int>& edge_list = a_mesh.get_1st_ring_edges( idx );
        for(unsigned curr = 0; curr < edge_list.size(); ++curr) {
            hd_1st_ring_edges[off + curr] = edge_list[curr];
        }

        // Compute at each vertex the sum of all incident edges
        Vec3 pos = a_mesh.get_vertex(idx);
        float sum_angles = 0.f;
        int dep      = a_mesh.get_1st_ring_offset(idx*2    );
        int nb_neigh = a_mesh.get_1st_ring_offset(idx*2 + 1);
        int end      = dep + nb_neigh;
        for(int n = dep; n < (dep+nb_neigh); n++)
        {
            int curr = a_mesh.get_1st_ring( n );
            int next = a_mesh.get_1st_ring( (n+1) >= end  ? dep : n+1 );

            Vec3 edge0 = a_mesh.get_vertex( curr ) - pos;
            Vec3 edge1 = a_mesh.get_vertex( next ) - pos;

            sum_angles += edge0.normalized().dot( edge1.normalized() );
        }
        hd_sum_angles[idx] = sum_angles;
    }
    //hd_sum_angles.update_device_mem();

    d_1st_ring_list         = h_1st_ring_list;
    d_1st_ring_list_offsets = h_1st_ring_list_offsets;
    //hd_1st_ring_edges.update_device_mem();

    ///////////
    // Copy faces
    if(a_mesh.get_nb_tris() > 0)
        memcpy( &d_input_tri[0], (const int*)a_mesh._mesh_static.get_tris(), a_mesh.get_nb_tris()*3 );

    if( a_mesh.get_nb_quads() > 0)
        memcpy( &d_input_quad[0], (const int*)a_mesh._mesh_static.get_quads(), a_mesh.get_nb_quads()*4);

    ///////////
    // Compute edge lengths and edge list
    for(int i = 0; i < a_mesh.get_nb_edges(); ++i){
        hd_edge_list   [i] = a_mesh.get_edge( i );
        hd_edge_lengths[i] = Mesh_utils::edge_len(a_mesh, i);
        float angle = Mesh_utils::edge_dihedral_angle(a_mesh, i);
        hd_edge_bending[i] = angle > 2.f*M_PI ? 0.f : angle;
    }

    //hd_edge_bending.update_device_mem();
    //hd_edge_lengths.update_device_mem();
    //hd_edge_list.   update_device_mem();

    ///////////
    // Compute tri areas
    for(int i = 0; i < a_mesh.get_nb_tris(); ++i)
        hd_tri_areas[i] = Mesh_utils::tri_area(a_mesh, i);

    //hd_tri_areas.update_device_mem();

    /////////////
    // Second ring
    std::vector<int> h_2nd_ring_list;
    std::vector<int> h_2nd_ring_list_offset;
    std::vector<float> h_2nd_ring_lengths;

    Std_utils::flatten( _mesh->_mesh_he.get_2nd_ring_verts(), h_2nd_ring_list, h_2nd_ring_list_offset);
    h_2nd_ring_lengths.resize( h_2nd_ring_list.size() );

    for(EMesh::Vert_idx i = 0; i < _mesh->get_nb_vertices(); ++i)
    {
        Vec3 center = _mesh->get_vertex( i );
        int off    = h_2nd_ring_list_offset[i*2];
        int nb_elt = h_2nd_ring_list_offset[i*2+1];
        for(int j = off; j < (off+nb_elt); ++j){
            Vec3 p = _mesh->get_vertex( h_2nd_ring_list[j] );
            h_2nd_ring_lengths[j] = (center-p).norm();
        }
    }

    d_2nd_ring_lengths.resize( h_2nd_ring_lengths.size() );
    d_2nd_ring_list.resize( h_2nd_ring_list.size() );
    d_2nd_ring_list_offsets.resize( h_2nd_ring_list_offset.size() );

    d_2nd_ring_lengths               = h_2nd_ring_lengths     ;
    d_2nd_ring_list                  = h_2nd_ring_list        ;
    d_2nd_ring_list_offsets          =  h_2nd_ring_list_offset;
}

// -----------------------------------------------------------------------------



void Animesh::compute_mvc()
{
#if 0
    HA_Vec3 h_base_grad( d_base_gradient.size() );
    h_base_grad.copy_from( d_base_gradient );

    //Device::Array<Vec3> d_grad( d_input_vertices.size() );
    std::vector<float> edge_lengths(_mesh->get_size_1st_ring_list());
    for(int i = 0; i < _mesh->get_nb_vertices(); i++)
    {
        Point3 pos( _mesh->get_vertex(i)      );
//        Vec3  nor( _mesh->get_mean_normal(i) );
        Vec3  nor( -(h_base_grad[i].normalized()) );

        Mat3 frame = Mat3::coordinate_system( nor ).transpose();
        float sum = 0.f;
        bool  out = false;
        // Look up neighborhood
        int dep      = _mesh->get_1st_ring_offset(i*2    );
        int nb_neigh = _mesh->get_1st_ring_offset(i*2 + 1);
        int end      = (dep+nb_neigh);

        if( nor.norm() < 0.00001f || _mesh->is_vert_on_side(i) ) {
            for(int n = dep; n < end; n++) hd_1st_ring_mvc[n] = 0.f;
        }
        else
        {
            for(int n = dep; n < end; n++)
            {
                int id_curr = _mesh->get_1st_ring( n );
                int id_next = _mesh->get_1st_ring( (n+1) >= end  ? dep   : n+1 );
                int id_prev = _mesh->get_1st_ring( (n-1) <  dep  ? end-1 : n-1 );

                // compute edge length
                Point3  curr( _mesh->get_vertex(id_curr) );
                Vec3 e_curr = (curr - pos);
                edge_lengths[n] = e_curr.norm();

                // compute mean value coordinates
                // coordinates are computed by projecting the neighborhood to the
                // tangent plane
                {

                    #if 1
                    // Project on tangent plane
                    Vec3 e_next = Point3( _mesh->get_vertex(id_next) ) - pos;
                    Vec3 e_prev = Point3( _mesh->get_vertex(id_prev) ) - pos;

                    e_curr = frame * e_curr;
                    e_next = frame * e_next;
                    e_prev = frame * e_prev;

                    e_curr.x = 0.f;
                    e_next.x = 0.f;
                    e_prev.x = 0.f;

                    float norm_curr_2D = e_curr.norm();

                    e_curr.normalize();
                    e_next.normalize();
                    e_prev.normalize();

                    // Computing mvc
                    float anext = std::atan2( -e_prev.z * e_curr.y + e_prev.y * e_curr.z, e_prev.dot(e_curr) );
                    float aprev = std::atan2( -e_curr.z * e_next.y + e_curr.y * e_next.z, e_curr.dot(e_next) );
                    #else
                    Vec3 e_next = nor.proj_on_plane( pos, _mesh->get_vertex(id_next).to_point3() ) /*_mesh->get_vertex(id_next).to_point3()*/ - pos;
                    Vec3 e_prev = nor.proj_on_plane( pos, _mesh->get_vertex(id_prev).to_point3() ) /*_mesh->get_vertex(id_prev).to_point3()*/ - pos;

                    float norm_curr_2D = e_curr.normalize();

                    e_next.normalize();
                    e_prev.normalize();

                    // Computing mvc
                    float anext = acos( e_prev.dot(e_curr) );
                    float aprev = acos( e_curr.dot(e_next) );
                    #endif

                    float mvc = 0.f;
                    if(norm_curr_2D > 0.0001f)
                        mvc = (std::tan(anext*0.5f) + std::tan(aprev*0.5f)) / norm_curr_2D;

                    sum += mvc;
                    hd_1st_ring_mvc[n] = mvc;
                    out = out || mvc < 0.f;
                }
            }
            // we ignore points outside the convex hull
            if( sum  <= 0.f || out || isnan(sum) ) {
                for(int n = dep; n < end; n++) hd_1st_ring_mvc[n] = 0.f;
            }
        }

    }
    d_1st_ring_lengths.copy_from( edge_lengths );
    hd_1st_ring_mvc.update_device_mem();
#endif
}

// -----------------------------------------------------------------------------
#if 0
static bool is_cluster_ssd(const Skeleton* s, int id)
{
    bool ssd = s->bone_type( id ) == Bone_type::SSD;
    const int pt = s->parent( id );
    if( pt > -1 )
    {
        const std::vector<int>& sons = s->get_sons( pt );
        for (unsigned j = 0; j < sons.size(); ++j)
            ssd = ssd && (s->bone_type( sons[j] ) == Bone_type::SSD);
    }

    return ssd;
}
#endif

// -----------------------------------------------------------------------------

void Animesh::init_ssd_interpolation_weights()
{
    int n = d_input_vertices.size();

//    std::vector<float> base_potential(n);
//    base_potential.copy_from(d_base_potential);

    std::vector<float> base_ssd_weights(n);
    base_ssd_weights = hd_ssd_interpolation_factor;
#if 0
    for(int i = 0; i < n; i++)
    {
        base_ssd_weights[i] = 0.f;
#if 1
        const Point3 vert = _mesh->get_vertex(i).to_point3();
        const int nearest = h_vertices_nearest_bones[i];
        Bone_cu b = _skel->get_bone_rest_pose( nearest );
        const float len = b.length() > 0.0001f ? b.length() : 1.f;

        bool ssd = is_cluster_ssd(_skel, nearest);

        bool fit;
        if( ssd )
            fit = false;
         else
        {
            // cluster parent is ssd ?
            int pt = _skel->get_parent( nearest );
            bool pt_ssd = ( pt > -1 && is_cluster_ssd(_skel, pt));

            // cluster son is ssd
            const std::vector<int>& sons = _skel->get_sons( nearest );
            bool s_ssd = ( sons.size() > 0 &&
                           is_cluster_ssd(_skel, sons[0]) &&
                           !_skel->is_leaf( sons[0]));

            if( (b.dist_proj_to( vert )/len) < 0.5f )
                fit = !pt_ssd; // Don't fit if parent cluster ssd
            else
                fit = !s_ssd; // Don't fit if son cluster ssd

        }
        base_ssd_weights[i] = fit ? 0.f : 1.f;
#endif
//        if(base_potential[i] <= 0.f) base_ssd_weights[i] = 1.f;
//        else                         base_ssd_weights[i] = 0.f;

    }
#endif
    //_mesh->diffuse_along_mesh(base_ssd_weights.ptr(), 1.f, 2);

    hd_ssd_interpolation_factor = base_ssd_weights;

    init_vert_to_fit();

    if( mesh_color == EAnimesh::SSD_INTERPOLATION)
        set_colors( mesh_color );
}

// -----------------------------------------------------------------------------

void Animesh::export_off(const char* filename) const
{

    // FIXME: take into acount quads
    // FIXME: use vmap_old_new to write index and vertices in the old order
    assert(false);

    using namespace std;
    ofstream file(filename);

    if(!file.is_open()){
        cerr << "Error exporting file " << filename << endl;
        exit(1);
    }

    file << "OFF" << endl;
    file << _mesh -> get_nb_vertices() << ' ' << _mesh -> get_nb_faces() << " 0" << endl;
    //Vec3* output_vertices;
    //_mesh->_mesh_gl._vbo.cuda_map_to(output_vertices);
    //float* vertices = new float[3 * _mesh -> get_nb_vertices()];
    //CUDA_SAFE_CALL(cudaMemcpy(vertices,
    //                          output_vertices,
    //                          3 * _mesh -> get_nb_vertices() * sizeof(float),
    //                          cudaMemcpyDeviceToHost));
    //_mesh->_mesh_gl._vbo.cuda_unmap();
    //for(int i = 0; i < _mesh -> get_nb_vertices(); i++){
    //    file << vertices[3 * i] << ' '
    //         << vertices[3 * i + 1] << ' '
    //         << vertices[3 * i + 2] << ' ' << endl;
    //}

    //for(int i = 0; i < _mesh -> get_nb_faces(); i++)
    //{
    //    EMesh::Tri_face t = _mesh->get_tri( i );
    //    file << "3 " << t.a << ' ' << t.b << ' ' << t.c << endl;
    //}

    file.close();
}

// -----------------------------------------------------------------------------

void Animesh::export_cluster(const char* filename)
{
    // HACK: update because cluster might have been updated in d_nearest_bone_in_device_mem
    // by the paint method
    h_vertices_nearest_bones =  d_vertices_nearest_bones ;
    init_verts_per_bone();
    update_nearest_bone_joint_in_device_mem();

    using namespace std;
    ofstream file(filename);

    if(!file.is_open()){
        cerr << "Error exporting file " << filename << endl;
        exit(1);
    }

    file << "[CLUSTER]" << std::endl;
    for(int i = 0; i < _mesh->get_nb_vertices(); ++i)
    {
        file << h_vertices_nearest_bones[i] << std::endl;
    }

    file.close();

}

// -----------------------------------------------------------------------------

void Animesh::import_cluster(const char* filename)
{
    using namespace std;
    ifstream file(filename);

    if(!file.is_open()){
        cerr << "Error exporting file " << filename << endl;
        exit(1);
    }

    std::string dummy;
    file >> dummy;
    for(int i = 0; i < _mesh->get_nb_vertices(); ++i)
    {
        file >> h_vertices_nearest_bones[i];
    }

    file.close();

    d_vertices_nearest_bones = h_vertices_nearest_bones ;
    init_verts_per_bone();
    update_nearest_bone_joint_in_device_mem();

}

// -----------------------------------------------------------------------------

void Animesh::draw(bool use_color_array, bool use_point_color) const
{
    if(!use_point_color)
        _mesh->draw_using_buffer_object( _mesh->_mesh_gl._vbo, _mesh->_mesh_gl._normals_bo, use_color_array );
    else
        _mesh->draw_using_buffer_object( _mesh->_mesh_gl._vbo, _mesh->_mesh_gl._normals_bo, _mesh->_mesh_gl._point_color_bo, use_color_array );
}

// -----------------------------------------------------------------------------

void Animesh::draw_rest_pose(bool use_color_array, bool use_point_color) const
{
    if(!use_point_color)
        _mesh->draw_using_buffer_object(*_vbo_input_vert, *_nbo_input_normal, use_color_array );
    else
        _mesh->draw_using_buffer_object(*_vbo_input_vert, *_nbo_input_normal, _mesh->_mesh_gl._point_color_bo, use_color_array );
}

// -----------------------------------------------------------------------------

void Animesh::draw_points_rest_pose() const
{
    _mesh->draw_points_using_buffer_object(*_vbo_input_vert, *_nbo_input_normal, _mesh->_mesh_gl._point_color_bo, true);
}


// -----------------------------------------------------------------------------

float Animesh::compute_nearest_vert_to_bone(int bone_id)
{
    return 1.f;
}

// -----------------------------------------------------------------------------

void Animesh::clusterize_euclidean(std::vector<int>& vertices_nearest_bones,
                                   std::vector<int>& h_vertices_nearest_joint,
                                   std::vector<int>& nb_vert_by_bone)
{
    const int nb_bones = _skel->get_bones().size();
    assert(nb_vert_by_bone.size() == nb_bones);
    for(int i = 0; i<nb_bones; i++)
        nb_vert_by_bone[i] = 0;

    int n = _mesh->get_nb_vertices();
    for(int i = 0; i < n ; i++)
    {
        float d0  = std::numeric_limits<float>::infinity();
        int   nd0 = _skel->root();

        float joint_dist       = std::numeric_limits<float>::infinity();
        int   nearest_joint_id = _skel->root();

        const Point3 current_vertex = _mesh->get_vertex(i).to_point3();
        for(int j = 0; j < _skel->nb_joints(); j++)
        {
            const Bone* b = _skel->get_bone( j );

            if( /*_skel->is_leaf(j) ||*/ !_do_bone_deform[j] ) ////////////////////////////////
                continue;

            // Compute nearest bone
            float dist2 = b->dist_sq_to(current_vertex);

            if(dist2 <= d0){
                d0  = dist2;
                nd0 = j;
            }

            // compute nearest joint
            const Point3 joint = _skel->joint_pos(j).to_point3();
            const Vec3 dir    = current_vertex-joint;
            float dist = dir.norm();
            // works fine but some mesh have corrupted normals so for the moment
            // I don't use this information
            float sign = 1.f;// dir.dot( current_normal );
            if(dist < joint_dist && sign >= 0){
                nearest_joint_id = j;
                joint_dist       = dist;
                hd_nearest_joint_dist[i] = dist;
            }
        }
        h_nearest_bone_dist     [i] = sqrt(d0);
        vertices_nearest_bones  [i] = nd0;
        h_vertices_nearest_joint[i] = nearest_joint_id;
        nb_vert_by_bone[nd0]++;
    }
    //hd_nearest_joint_dist.update_device_mem();
}

// -----------------------------------------------------------------------------
#if 1
void Animesh::clusterize_weights(std::vector<int>& vertices_nearest_bones,
                                 std::vector<int>& h_vertices_nearest_joint,
                                 std::vector<int>& nb_vert_by_bone)
{
    const int nb_bones = _skel->get_bones().size();
    assert(nb_vert_by_bone.size() == nb_bones);
    for(int i = 0; i<nb_bones; i++)
        nb_vert_by_bone[i] = 0;

    int n = _mesh->get_nb_vertices();
    for(int i = 0; i < n; i++)
    {
        int nd0 = _skel->root();

        // float joint_dist       = std::numeric_limits<float>::infinity();
        // int   nearest_joint_id = _skel->root();

        float w_max = -1.f;
        const std::map<int, float>& map = h_weights[i];
        std::map<int, float>::const_iterator it;
        for(it = map.begin(); it != map.end(); ++it)
        {
            const int   k = it->first;
            const float w = it->second;

            if( /*!_skel->is_leaf(k) &&*/ _do_bone_deform[k] ) ///////////////////////////////////
            {
                if( w_max < w){
                    w_max = w;
                    nd0 = k;
                }
            }
        }

        // compute nearest joint
        //h_nearest_bone_dist     [i] = sqrt(d0);
        vertices_nearest_bones  [i] = nd0;
        h_vertices_nearest_joint[i] = nd0; // FIXME : really compute the nearest joint and not the nearest bone
        nb_vert_by_bone[nd0]++;
    }
}
#endif

// -----------------------------------------------------------------------------

void Animesh::select(int x, int y, Select_type<int>* selection_set, bool rest_pose)
{
    selection_set->reset();
    const int nb_points = d_input_vertices.size();

    // Copy vertices to host
    std::vector<float> h_vert(nb_points*3);
    float* d_vertices = 0;
    d_vertices = rest_pose ? (float*)d_input_vertices[0] : (float*)hd_output_vertices[0];
    memcpy(&h_vert[0], d_vertices, nb_points*3);

    GLint viewport[4];
    GLdouble modelview[16];
    GLdouble projection[16];
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
    glGetDoublev(GL_PROJECTION_MATRIX, projection);
    GLdouble vx, vy, vz;
    float depth;
    // Read depth buffer at mouse position
    glReadPixels( x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
    for(int i = 0; i < nb_points; i++)
    {
        gluProject(h_vert[i*3], h_vert[i*3+1], h_vert[i*3+2],
                   modelview, projection, viewport,
                   &vx, &vy, &vz);

        selection_set->test(i,
                            Vec3((float)vx, (float)vy, (float)vz),
                            Vec3((float)x , (float)y , depth    ) );
    }
}

// -----------------------------------------------------------------------------

void Animesh::clusterize(EAnimesh::Cluster_type type)
{
    if( type == EAnimesh::EUCLIDEAN )
    {
        clusterize_euclidean(h_vertices_nearest_bones,
                             h_vertices_nearest_joint,
                             nb_vertices_by_bones);
    }
    else if( type == EAnimesh::FROM_WEIGHTS)
    {
        clusterize_weights(h_vertices_nearest_bones,
                           h_vertices_nearest_joint,
                           nb_vertices_by_bones);
    }

    init_verts_per_bone();
    update_nearest_bone_joint_in_device_mem();
}

// -----------------------------------------------------------------------------

static void compact_to_hd_array(std::vector<int>& hd_array,
                                int target_size,
                                const std::vector<bool>& is_elt_stored)
{
    hd_array.resize( target_size );
    for(unsigned i = 0, acc = 0; i < is_elt_stored.size(); ++i)
        if( is_elt_stored[i] )
            hd_array[acc++] = (int)i;

    //hd_array.update_device_mem();
}

// -----------------------------------------------------------------------------

void Animesh::compute_smooth_parts()
{
    {
        ////////
        // Compute free vertices
        std::vector<bool> is_free_verts(_mesh->get_nb_vertices(), false);
        int nb_free_verts = 0;
#if 0
        for(int i = 0; i < _mesh->get_nb_vertices(); ++i)
        {
            const Point3 pos = _mesh->get_vertex( i ).to_point3();
            for(int j = 0; j < _skel->nb_joints(); j++)
            {
                const Bone* b = _skel->get_bone( j );

                if( _skel->is_leaf(j) || !_do_bone_deform[j] || _skel->root() == j)
                    continue;

                // Hard coded dist for testing purpose$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                //rad_cyl == 11.f
                // rad_dana == 2.f
                // rad_hand == 4.f
                if( !is_free_verts[i] && (b->org() - pos).norm() < 11.f /*(h_junction_radius[j] + h_junction_radius[j] * 1.3f) */){
                    d_input_smooth_factors.set(i, 1.f);
                    is_free_verts[i] = true;
                    nb_free_verts++;
                }
            }
        }
#else
        for(int i = 0; i < _mesh->get_nb_vertices(); ++i)
        {

            int j = h_vertices_nearest_joint[i];

            //if( _skel->is_leaf(j) || !_do_bone_deform[j] || _skel->root() == j) continue;

            if( !is_free_verts[i] )
            {
                //d_input_smooth_factors.set(i, 1.f);
                is_free_verts[i] = true;
                nb_free_verts++;
            }
        }
#endif

        ////////
        // Compact the list and uploaded to GPU

        //compact_to_hd_array(hd_free_vertices, nb_free_verts, is_free_verts);
        hd_free_vertices.resize( nb_free_verts );
        hd_map_verts_to_free_vertices.resize( _mesh->get_nb_vertices() );
        for(int i = 0, acc = 0; i < _mesh->get_nb_vertices(); ++i)
            if( is_free_verts[i] ){
                hd_map_verts_to_free_vertices[i] = acc;
                hd_free_vertices[acc] = i;
                acc++;
            }
            else
                hd_map_verts_to_free_vertices[i] = -1;

        //hd_free_vertices.update_device_mem();
        //hd_map_verts_to_free_vertices.update_device_mem();
    }

    {
        ////////
        // Compute associated potential to free vertices
        hd_free_base_potential.resize( hd_free_vertices.size() );

        /*
         *
         * The function is supposed to compute base potential at rest pose only for the sub list of "free_vertices"
         * I broke the function must recode it or forget about it.
        compute_potential(hd_free_vertices.d_ptr(),
                          hd_free_vertices.size(),
                          hd_free_base_potential.d_ptr() );
*/
        std::cerr << "WARNING: old code to be removed not working anymore" << std::endl;
        //hd_free_base_potential.update_host_mem();
    }

    {
        ////////
        // Compute associated edges to free vertices
        // Look up free verts and its 1st ring
        int nb_free_edges = 0;
        std::vector<bool> is_free_edge( _mesh->get_nb_edges(), false );
        for(int i = 0; i < hd_free_vertices.size(); ++i)
        {
            const int vert_id = hd_free_vertices[i];
            const std::vector<int>& edge_list = _mesh->get_1st_ring_edges(vert_id);
            for(unsigned j = 0; j < edge_list.size(); ++j)
            {
                const int edge_id = edge_list[j];
                if( !is_free_edge[edge_id] ){
                    // tag edges / count them
                    is_free_edge[edge_id] = true;
                    nb_free_edges++;
                }
            }
        }
        // compact the array
        compact_to_hd_array(hd_free_edges, nb_free_edges, is_free_edge);
    }


    {
        ////////
        // Compute associated triangles to free vertices
        // Look up free verts and associated tris
        int nb_free_tris = 0;
        std::vector<bool> is_free_tri( _mesh->get_nb_tris(), false );
        for(int i = 0; i < hd_free_vertices.size(); ++i)
        {
            const int vert_id = hd_free_vertices[i];
            const std::vector<int>& tri_list = _mesh->get_1st_ring_tris(vert_id);
            for(unsigned j = 0; j < tri_list.size(); ++j)
            {
                const int tri_id = tri_list[j];
                if( !is_free_tri[tri_id] ){
                    // tag tris / count them
                    is_free_tri[tri_id] = true;
                    nb_free_tris++;
                }
            }

        }
        // compact the array
        compact_to_hd_array(hd_free_triangles, nb_free_tris, is_free_tri);
    }

    {
        const int nb_free_verts = hd_free_vertices.size();
        hd_energy_terms.resize( nb_free_verts    , 0.f );
        hd_energy_grad. resize( nb_free_verts * 2, 0.f );
        hd_energy_verts.resize( nb_free_verts * 2, 0.f );
    }
}

// -----------------------------------------------------------------------------

void Animesh::update_nearest_bone_joint_in_device_mem()
{
#if 0
    int n = _mesh->get_nb_vertices();
    // Convert host ids to device ids for the nearest joints
    std::vector<DBone_id> tmp (n);
    std::vector<DBone_id> tmp2(n);
    for(int i = 0; i < n; i++){
        tmp [i] = _skel->get_bone_device_idx( h_vertices_nearest_bones[i] );
        tmp2[i] = _skel->get_bone_device_idx( h_vertices_nearest_joint[i] );
    }
    d_nearest_bone_in_device_mem. copy_from(tmp);
    d_nearest_joint_in_device_mem.copy_from(tmp2);

    d_vertices_nearest_bones.copy_from(h_vertices_nearest_bones);
    d_vertices_nearest_joint.copy_from(h_vertices_nearest_joint);
#endif
}

// -----------------------------------------------------------------------------

void Animesh::set_default_bones_radius()
{
    const int nb_verts  = _mesh->get_nb_vertices();
    const int nb_joints = _skel->nb_joints();

    std::vector<float> avg_rad     (nb_joints);
    std::vector<float> nearest_rad (nb_joints);
    std::vector<float> farthest_rad(nb_joints);
    std::vector<int>   nb_smp      (nb_joints);

    const float inf = std::numeric_limits<float>::infinity();
    for(int i = 0; i < nb_joints; i++) {
        nearest_rad [i] = inf;
        farthest_rad[i] = 0.f;
        avg_rad     [i] = 0.f;
        nb_smp      [i] = 0;
    }

    for(int i = 0; i < nb_verts; i++)
    {
        const int j = h_vertices_nearest_bones[i];
        const Point3 vert = _mesh -> get_vertex(i).to_point3();
        float  d = _skel->get_bone(j)->dist_to( vert );

        nearest_rad [j] = std::min(nearest_rad [j], d);
        farthest_rad[j] = std::max(farthest_rad[j], d);
        avg_rad[j] += d;
        nb_smp[j]++;
    }

    for(int i = 0; i < nb_joints; i++)
    {
        // Cylinder radius is average vertices distance
        avg_rad[i] = nb_smp[i] ? avg_rad[i] / nb_smp[i] : 1.f;
        _skel->set_bone_radius(i, avg_rad[i]);

        // HRBF compact support radius is farthest vertex distance
        const float radius = farthest_rad[i] == 0.f ? 1.f : farthest_rad[i];
        _skel->set_bone_hrbf_radius(i, radius);

        // Junction radius is nearest vertex distance
        h_junction_radius[i] = nearest_rad[i] == inf ? 1.f : nearest_rad[i];
    }
}

// -----------------------------------------------------------------------------

float distsqToSeg(const Point3& v, const Point3& p1, const Point3& p2)
{
    Vec3 dir   = p2 - p1;
    Vec3 difp2 = p2 - v;

    if(difp2.dot(dir) < 0.f) return difp2.norm_squared();

    Vec3 difp1 = v - p1;
    float dot = difp1.dot(dir);

    if(dot <= 0.f) return difp1.norm_squared();

    return max(0.f, difp1.norm_squared() - dot*dot / dir.norm_squared());
}

// -----------------------------------------------------------------------------


Point3 projToSeg(const Point3& v, const Point3& p1, const Point3& p2)
{

  Vec3 dir = p2 - p1;

  if( (p2 - v).dot(dir) < 0.f) return p2;

  float dot = (v - p1).dot(dir);

  if(dot <= 0.f) return p1;

  return p1 + dir * (dot / dir.norm_squared());
}

// -----------------------------------------------------------------------------

bool vectorInCone(const Vec3& v, const std::vector<Vec3>& ns)
{
    int i;
    Vec3 avg = Vec3(0.f, 0.f, 0.f);
    for(i = 0; i < (int)ns.size(); ++i)
        avg += ns[i];

    return v.normalized().dot(avg.normalized()) > 0.5f;
}

// -----------------------------------------------------------------------------

void Animesh::heat_diffuse_ssd_weights(float heat)
{

#if 1
    // Heat diffusion code :
//    assert(_mesh->is_closed());
//    assert(_mesh->is_manifold());
    assert(_mesh->get_nb_quads() == 0);

    init_rigid_ssd_weights();

    std::cout << "heat coeff" << heat << std::endl;
    int nv       = _mesh->get_nb_vertices();
    int nb_bones = _skel->nb_joints();

    std::vector< Vec3             > vertices(nv);
    // edges[nb_vertex][nb_vertex_connected_to_it]
    std::vector< std::vector<int>    > edges(nv);
    std::vector< std::vector<double> > boneDists(nv);
    std::vector< std::vector<bool>   > boneVis(nv);
    std::vector<std::vector<std::pair<int, double> > > nzweights(nv);

    // Fill edges and vertices
    for(int i = 0; i<nv; ++i)
    {
        Vec3 v = _mesh->get_vertex(i);
        vertices[i] = Vec3(v.x, v.y, v.z);
        int dep = _mesh->get_1st_ring_offset(i*2    );
        int off = _mesh->get_1st_ring_offset(i*2 + 1);
        for(int j=dep; j< (dep+off); ++j )
            edges[i].push_back(_mesh->get_1st_ring(j));
    }

    // Compute bone visibility and distance from a vertex to each bone
    for(int i = 0; i < nv; ++i)
    {
        boneDists[i].resize(nb_bones, -1);
        boneVis[i].resize(nb_bones, true);
        Point3 cPos( vertices[i] );

        // Compute the normal of each adjacent face
        std::vector<Vec3> normals;
        for(int j = 0; j < (int)edges[i].size(); ++j)
        {
            int nj = (j + 1) % edges[i].size();
            Vec3 v1 = vertices[edges[i][j] ] - cPos;
            Vec3 v2 = vertices[edges[i][nj]] - cPos;
            Vec3 n  = (v1.cross(v2)).normalized();
            normals.push_back( n );
        }

#if 1
        // Compute the nearest bone euclidean distance to the current vertex

        const double inf = std::numeric_limits<double>::infinity();;
        double minDist   = inf;
        for(int j = 0; j < nb_bones; ++j)
        {
            if( _skel->is_leaf(j) || !_do_bone_deform[j] )
            {
                boneDists[i][j] = inf;
                continue;
            }

            Bone_cu bone = _skel->get_bone_rest_pose(j);

            Point3 p1 = bone.org();
            Point3 p2 = bone.org() + bone.dir();
            boneDists[i][j] = sqrt(distsqToSeg(cPos, p1, p2));
            minDist = min(boneDists[i][j], minDist);
        }
#endif

        for(int j = 0; j < nb_bones; ++j)
        {
            //the reason we don't just pick the closest bone is so that if two are
            //equally close, both are factored in.
            if(boneDists[i][j] > minDist * 1.0001)
                continue;

            //Bone_cu bone = _skel->get_bone_rest_pose( j );

            //Point3 pos  = cPos;
            //Point3 p1   = bone.org();
            //Point3 p2   = bone.org() + bone.dir();
            //Point3 proj = projToSeg(pos, p1, p2);

            if( _skel->is_leaf(j) )
                boneVis[i][j] = false;
            else
                // TODO: implement visibility test
                boneVis[i][j] = /*tester->canSee(cPos, p)*/true /*&& vectorInCone(-(pos - proj), normals)*/;
        }
    }

#if 0
    // Fill boneDists with voxel distance
    for(int j = 0; j < nb_bones; ++j)
    {
        std::vector<Vox::Segment> bones;
        Bone_cu bone = _skel->get_bone_rest_pose(j+1);

        Point3 org = bone.org();
        Point3 end = bone.org() + bone->dir();
        bones.push_back( Vox::Segment(Vox::V3d(org.x, org.y, org.z),
                                      Vox::V3d(end.x, end.y, end.z) )
                        );

        Grid grid_closest = Vox::Geo_dist::closest_bone(*_voxels, bones);

        for(int i = 0; i < nv; ++i)
        {
            Vec3    v     = _mesh->get_vertex(i);
            Vox::V3d  x     = Vox::V3d(v.x, v.y, v.z);
            Vox::V3d  x_vox = grid_closest.convert_real_to_voxel_index( x );
            Vox::Idx3 u_vox = x_vox.to_index_3();
            Vox::Geo_dist::Cell c = grid_closest.get( u_vox );
            boneDists[i][j] = c._dist;
        }
    }
#endif

    // Compute weights
//    rig(vertices, edges, boneDists, boneVis, nzweights, heat);

    // convert nzweights to our array representation
    std::vector<float> h_weights; h_weights.reserve(2 * nv);
    std::vector<int>   h_joints; h_joints.reserve( 2 * nv);
    std::vector<int>   h_jpv(2u*nv);

    int acc = 0;
    for(int i=0; i<nv; ++i)
    {
        int size = nzweights[i].size();
        for(int j=0; j<size; j++)
        {
            int   joint_id = nzweights[i][j].first;
            float weight   = (float)nzweights[i][j].second;

            h_joints. push_back( joint_id );
            h_weights.push_back( weight   );
        }
        h_jpv[i*2    ] = acc;  // starting index
        h_jpv[i*2 + 1] = size; // number of bones influencing the vertex
        acc += size;
    }

    d_jpv = h_jpv;
    d_weights.resize(acc);
    d_weights =h_weights;
    d_joints.resize(acc);
    d_joints =h_joints;

    update_host_ssd_weights();

//    std::cout << "COMPUTE NEW CLUSTER FROM WEIGHTS" << std::endl;
//    clusterize( EAnimesh::FROM_WEIGHTS );/////////////////////DEBUG//////////////
#endif
}

// -----------------------------------------------------------------------------

void Animesh::init_rigid_ssd_weights()
{
    int nb_vert = _mesh->get_nb_vertices();

    std::vector<float> weights(nb_vert);
    std::vector<int>   joints (nb_vert);
    std::vector<int>   jpv    (2u*nb_vert);

    for(int i = 0; i < nb_vert; ++i)
    {
        joints [i] = h_vertices_nearest_bones[i];
        weights[i] = 1.f;

        jpv[i*2    ] = i; // starting index
        jpv[i*2 + 1] = 1; // number of bones influencing the vertex

        int start = i;
        int end   = start + 1;

        h_weights[i].clear();
        for(int j = start; j < end ; j++)
            h_weights[i][joints[j]] = weights[j];
    }

    d_jpv=jpv;
    d_weights.resize(nb_vert);
    d_weights= weights;
    d_joints.resize(nb_vert);
    d_joints =joints;
}

// -----------------------------------------------------------------------------

void Animesh::geodesic_diffuse_ssd_weights(int nb_iter, float strength)
{
    std::vector<std::map<int, float> >& weights = h_weights;

    // diffuse weights
    std::vector<std::map<int, float> > new_weights(weights.size());
    std::map<int, float>::iterator it;
    for(int iter = 0; iter < nb_iter; iter++)
    {
        // For all vertices
        for(int i = 0; i < _mesh->get_nb_vertices(); i++)
        {
            Point3 center(_mesh->get_vertex(i));

            std::vector<float> bones_weights(_skel->nb_joints(), 0.f);

            std::map<int, float>& map = weights[i];
            for(unsigned bone_id = 0; bone_id < bones_weights.size(); bone_id++)
            {
                it = map.find(bone_id);
                float vert_weight = 0.f;
                if(it != map.end()) vert_weight = it->second;

                if(vert_weight <= 0.000001f)
                {
                    float min_length = std::numeric_limits<float>::infinity();
                    int dep      = _mesh->get_1st_ring_offset(i*2    );
                    int nb_neigh = _mesh->get_1st_ring_offset(i*2 + 1);
                    for(int n = dep; n < (dep+nb_neigh); n++)
                    {
                        int index_neigh = _mesh->get_1st_ring(n);

                        Point3 neigh(_mesh->get_vertex(index_neigh));
                        float norm = (center-neigh).norm();

                        std::map<int, float>::iterator it_nei;
                        it_nei = weights[index_neigh].find(bone_id);
                        if(it_nei != weights[index_neigh].end())
                        {
                            float w = it_nei->second;
                            if(w > 0.0001f)
                            {
                                if(norm < min_length)
                                {
                                    min_length = norm;
                                    w = w - min_length * strength /** iter*iter*/;
                                    bones_weights[bone_id] = w;
                                }
                            }
                        }
                    }
                }
                else
                    bones_weights[bone_id] = vert_weight;
            }


            for(unsigned j = 0; j < bones_weights.size(); j++)
            {
                float w = bones_weights[j];
                if(w > 0.f){
                    new_weights[i][j] = w;
                }
            }
        }// END nb_vert

        for(int i = 0; i < _mesh->get_nb_vertices(); i++){
            weights[i].swap(new_weights[i]);
            new_weights[i].clear();
        }
    }// END nb_iter

    // Eliminate near zero weigths and normalize :
    for(int i = 0; i < _mesh->get_nb_vertices(); i++)
    {
        std::map<int, float>& map = weights[i];
        float sum = 0.f;
        for(it = map.begin(); it != map.end(); ++it)
        {
            float w = it->second;
            if(w > 0.0001f)
            {
                w = w*w*w;
                new_weights[i][it->first] = w;
                sum += w;
            }
        }

        weights[i].clear();

        std::map<int, float>& n_map = new_weights[i];
        for(it = n_map.begin(); it != n_map.end(); ++it)
        {
            float w = it->second / sum;
            if(w > 0.0001f)
                weights[i][it->first] = w;
            if(w > 1.1f)
            {
                std::cout << w << std::endl;
                assert(false);
            }
        }
    }

    update_device_ssd_weights();
}

// -----------------------------------------------------------------------------

void Animesh::topology_diffuse_ssd_weights(int nb_iter, float strength)
{
    std::vector<std::map<int, float> >& weights = h_weights;

    // diffuse weights
    std::vector<std::map<int, float> > new_weights(weights.size());
    std::map<int, float>::iterator it;
    for(int iter = 0; iter < nb_iter; iter++)
    {
        // For all vertices
        for(int i = 0; i < _mesh->get_nb_vertices(); i++)
        {
            std::vector<float> bones_weights(_skel->nb_joints(), 0);
            // For all vertices' neighborhoods sum the weight of each bone
            int dep      = _mesh->get_1st_ring_offset(i*2    );
            int nb_neigh = _mesh->get_1st_ring_offset(i*2 + 1);
            float cotan_sum = 0.f;
            for(int n = dep; n < (dep+nb_neigh); n++)
            {
                int index_neigh = _mesh->get_1st_ring(n);

                std::map<int, float>& map = weights[index_neigh];
                for(it = map.begin(); it != map.end(); ++it) {
                    bones_weights[it->first] += it->second /** hd_1st_ring_cotan[n]*/;
                    cotan_sum += hd_1st_ring_cotan[n];
                }
            }

            for(unsigned j = 0; j < bones_weights.size(); j++)
            {
                float val = bones_weights[j];
                bones_weights[j] = val * (1.f-strength) + val * strength / (nb_neigh/*cotan_sum*/);
            }

            for(unsigned j = 0; j < bones_weights.size(); j++)
            {
                float w = bones_weights[j];
                if(w > 0.f) new_weights[i][j] = w;
            }
        }// END nb_vert

        for(int i = 0; i < _mesh->get_nb_vertices(); i++){
            weights[i].swap(new_weights[i]);
            new_weights[i].clear();
        }
    }// END nb_iter

    // Eliminate near zero weigths and normalize :
    for(int i = 0; i < _mesh->get_nb_vertices(); i++)
    {
        std::map<int, float>& map = weights[i];
        float sum = 0.f;
        for(it = map.begin(); it != map.end(); ++it)
        {
            float w = it->second;
            if(w > 0.0001f)
            {
                new_weights[i][it->first] = w;
                sum += w;
            }
        }

        weights[i].clear();

        std::map<int, float>& n_map = new_weights[i];
        for(it = n_map.begin(); it != n_map.end(); ++it)
        {
            float w = it->second / sum;
            if(w > 0.0001f)
                weights[i][it->first] = w;
        }
    }

    update_device_ssd_weights();
}

// -----------------------------------------------------------------------------

void Animesh::init_smooth_factors(std::vector<float>& d_smooth_factors)
{
    const int nb_vert = _mesh->get_nb_vertices();
    std::vector<float> smooth_factors(nb_vert);

    for(int i=0; i<nb_vert; i++)
        smooth_factors[i] = 0.0f;

    d_smooth_factors = smooth_factors;
}

// -----------------------------------------------------------------------------

void Animesh::diffuse_attr(int nb_iter, float strength, float *attr)
{
    //Animesh_kers::diffuse_values(attr,
    //                             d_vals_buffer.ptr(),
    //                             d_1st_ring_list,
    //                             d_1st_ring_list_offsets,
    //                             strength,
    //                             nb_iter);
}

// -----------------------------------------------------------------------------

void Animesh::get_anim_vertices_aifo(std::vector<float>& anim_vert)
{
    const int nb_vert = hd_output_vertices.size();
    anim_vert.reserve(nb_vert);
    std::vector<Vec3> h_out_verts(nb_vert);

    //hd_output_vertices.update_host_mem();
    //h_out_verts.copy_from(hd_output_vertices);

    //hd_tmp_vertices.update_host_mem();
    h_out_verts = hd_tmp_vertices;

    for(int i = 0; i < nb_vert; i++)
    {
        Vec3 p = h_out_verts[vmap_new_old[i]];
        anim_vert.push_back(p.x);
        anim_vert.push_back(p.y);
        anim_vert.push_back(p.z);
    }
}

// -----------------------------------------------------------------------------

void Animesh::set_bone_type(int id, int bone_type)
{
    _skel->reset();
    Bone* bone = 0;
    const Bone* prev_bone = _skel->get_bone( id );
    float rad = prev_bone->radius();
    switch(bone_type){
#if 0
    case EBone::PRECOMPUTED:
    {
        // We don't precompute an already precomputed primitive
        assert(_skel->bone_type(id) != EBone::PRECOMPUTED );
        // Precompute a SSD bone is useless and should be forbiden
        assert(_skel->bone_type(id) != EBone::SSD         );

        Bone_precomputed* b = new Bone_precomputed( prev_bone->get_obbox() );
        Precomputed_prim& grid = b->get_primitive();
        grid.fill_grid_with( prev_bone );
        bone = b;
    }break;
    case EBone::HRBF:     bone = new Bone_hrbf(rad);      break;
    case EBone::CYLINDER: bone = new Bone_cylinder();     break;
#endif
    case EBone::SSD:      bone = new Bone_ssd();          break;

    default: //unknown bone type !
        assert(false);
        break;

    }


    bone->set_radius(rad);
    _skel->set_bone(id, bone);

    init_ssd_interpolation_weights();
    _skel->unreset();
}

// -----------------------------------------------------------------------------

float Animesh::get_junction_radius(int bone_id){
    assert(bone_id >=0                 );
    assert(bone_id < _skel->nb_joints());
    return h_junction_radius[bone_id];
}

// -----------------------------------------------------------------------------

void Animesh::set_junction_radius(int bone_id, float rad)
{
    assert(bone_id >=0                 );
    assert(bone_id < _skel->nb_joints());
    h_junction_radius[bone_id] = rad;
}

// -----------------------------------------------------------------------------

void Animesh::set_ssd_weight(int id_vertex, int id_joint, float weight)
{
    id_joint = _skel->parent(id_joint);

    assert(id_vertex < (int)d_input_vertices.size());
    // clamp [0, 1]
    weight = max(0.f, min(weight, 1.f));

    float old_weight = get_ssd_weight(id_vertex, id_joint);
    float delta      = old_weight - weight;

    int start, end;
	start = d_jpv[id_vertex*2];
	end   = d_jpv[id_vertex*2+1];
    //d_jpv.fetch(id_vertex*2  , start);
    //d_jpv.fetch(id_vertex*2+1, end  );

    delta = delta / (float)(end-1);

    for(int i=start; i<(start+end); i++)
    {
        int current_joint;
		current_joint = d_joints[i];
        //d_joints.fetch(i, current_joint);
        if(current_joint == id_joint)
            d_weights[i]= weight;
        else
        {
            float w;
			w = d_weights[i];
            //d_weights.fetch(i, w);
            d_weights[i]= w+delta;
        }
    }
}

// -----------------------------------------------------------------------------

float Animesh::get_ssd_weight(int id_vertex, int id_joint)
{
    assert(id_vertex < d_input_vertices.size());

    int start, end;
	start = d_jpv[id_vertex*2];
	end = d_jpv[id_vertex*2+1];
    //d_jpv.fetch(id_vertex*2  , start);
    //d_jpv.fetch(id_vertex*2+1, end  );

    for(int i=start; i<(start+end); i++)
    {
        int current_joint;
		current_joint = d_joints[i];
        //d_joints.fetch(i, current_joint);
        if(current_joint == id_joint)
        {
            float w;
			w = d_weights[i];
            //d_weights.fetch(i, w);
            return w;
        }
    }

    // Joint "id_joint" is not associated to this vertex
    assert(false);
    return 0.f;
}

// -----------------------------------------------------------------------------

void Animesh::get_ssd_weights(std::vector<std::map<int, float> >& weights)
{
    const int nb_vert = d_input_vertices.size();
    weights.clear();
    weights.resize(nb_vert);

    std::vector<float> h_weights(d_weights.size());
    std::vector<int> h_joints(d_joints.size());
    std::vector<int> h_jpv(d_jpv.size());
    h_weights =d_weights;
    h_joints= d_joints;
    h_jpv = d_jpv;

    for( int i = 0; i < nb_vert; i++)
    {
        int start = h_jpv[i*2];
        int end   = start + h_jpv[i*2 + 1];
        weights[i].clear();
        for(int j = start; j < end ; j++){
            weights[i][h_joints[j]] = h_weights[j];
            //std::cout << h_weights[j] << std::endl;
        }
    }
}

// -----------------------------------------------------------------------------

void Animesh::update_host_ssd_weights()
{
    get_ssd_weights(h_weights);
}

// -----------------------------------------------------------------------------

void Animesh::set_ssd_weights(const std::vector<std::map<int, float> >& in_weights)
{
    const int nb_vert = d_input_vertices.size();
    assert( in_weights.size() == (unsigned)nb_vert );

    std::vector<float> weights;
    std::vector<int>   joints;
    std::vector<int>             jpv(nb_vert*2);

    weights.reserve(nb_vert*2);
    joints.reserve(nb_vert*2);

    int acc = 0;
    for( int i = 0; i < nb_vert; i++)
    {
        const std::map<int, float>& map = in_weights[i];
        jpv[i*2    ] = acc;
        jpv[i*2 + 1] = map.size();
        std::map<int, float>::const_iterator it;
        for(it = map.begin(); it != map.end(); ++it)
        {
            joints.push_back(it->first);
            weights.push_back(it->second);
        }
        acc += map.size();
    }

    d_weights.resize(weights.size());
    d_joints.resize(joints.size());
    d_jpv.resize(jpv.size());
    d_weights= weights;
    d_joints= joints;
    d_jpv = jpv;

//    std::cout << "COMPUTE NEW CLUSTER FROM WEIGHTS" << std::endl;
//    clusterize( EAnimesh::FROM_WEIGHTS );/////////////////////DEBUG//////////////
}

// -----------------------------------------------------------------------------

void Animesh::update_device_ssd_weights()
{
    set_ssd_weights(h_weights);
}

// -----------------------------------------------------------------------------

void Animesh::compute_potential(const Vec3* vert_pos,
                                const int nb_vert,
                                float* d_base_potential,
                                Vec3* d_base_grad)
{
    //const int nb_verts = nb_vert;
    //const int block_size = 256;
    //const int grid_size =
    //        (nb_verts + block_size - 1) / block_size;

    //Animesh_kers::compute_potential<<<grid_size, block_size>>>
    //    (_skel->skel_id(),
    //     (Point3*)vert_pos,
    //     nb_vert,
    //     d_base_potential,
    //     d_base_grad);

    //CUDA_CHECK_ERRORS();
}

// -----------------------------------------------------------------------------

void Animesh::export_weights(const char* filename)
{
    using namespace std;
    ofstream file(filename, ios_base::out|ios_base::trunc);

    if(!file.is_open()){
        cerr << "Error exporting file " << filename << endl;
        exit(1);
    }

    // Copy to host :
    std::vector<int>   h_jpv(d_jpv.size());
    std::vector<int>  h_joints(d_joints.size());
    std::vector<float> h_weights(d_weights.size());

    //h_jpv.copy_from(d_jpv);
    //h_joints.copy_from(d_joints);
    //h_weights.copy_from(d_weights);
	h_jpv = d_jpv;
	h_joints = d_joints;
	h_weights = d_weights;

    for(int i = 0; i < d_input_vertices.size(); i++)
    {
        int start, end;
        float sum_weights = 0.f;
        // vertices are not necessarily
		const int temp = vmap_new_old[i]*2;
		h_jpv[2];
        start = h_jpv[ temp  ];
        end   = h_jpv[temp + 1];

        for(int j=start; j<(start+end); j++)
        {
            float weight = h_weights[j];
            int   bone   = h_joints[j];
            sum_weights += weight;

            file << bone << " " << weight << " ";
        }

        if((sum_weights > 1.0001f) || (sum_weights < -0.0001f)){
            std::cerr << "WARNING: exported ssd weights does not sum to one ";
            std::cerr << "(line " << (i+1) << ")" << std::endl;
        }
        file << endl;
    }
    file.close();
}


// -----------------------------------------------------------------------------

void Animesh::read_weights_from_file(const char* filename,
                                     bool file_has_commas)
{
    // TODO: to be deleted use : parsers/weights_loader.hpp instead
    using namespace std;


    ifstream file(filename);

    int n = _mesh -> get_nb_vertices();
    std::vector<float> h_weights; h_weights.reserve(2*n);
    std::vector<int>   h_joints; h_joints.reserve(2*n);
    std::vector<int>   h_jpv(2*n);

    if(!file.is_open()){
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    int k = 0;
    for(int i = 0; i < n; i++)
    {
        std::string str_line;
        std::getline(file, str_line);
        std::stringbuf current_line_sb(str_line, ios_base::in);

        istream current_line(&current_line_sb);
        int j = 0;
        //int j_old = -1;
        float weight, sum_weights = 0.f;
        int p = 0;
        while(!current_line.eof() && !str_line.empty())
        {
            current_line >> j;
            //if(j == j_old) break;
            if(file_has_commas) current_line.ignore(1,',');

            current_line >> weight;

            if(file_has_commas) current_line.ignore(1,',');

            current_line.ignore(10,' ');

            if(current_line.peek() == '\r') current_line.ignore(1,'\r');

            if(current_line.peek() == '\n') current_line.ignore(1,'\n');

            p++;
            h_weights.push_back( weight ); // SSD weight
            h_joints.push_back(  j     ); // joint number
            //j_old = j;

            sum_weights += weight;
            if(j < 0 || j > _skel->nb_joints()){
                std::cerr << "ERROR: incorrect joint id in imported ssd weights.";
                std::cerr << "Maybe the file does not match the skeleton";
                std::cerr << std::endl;
            }
        }

        if((sum_weights > 1.0001f) || (sum_weights < -0.0001f)){
            std::cerr << "WARNING: imported ssd weights does not sum to one ";
            std::cerr << "(line " << (i+1) << ")" << std::endl;
        }

        // we use vmap_old_new because Animesh does not necessarily
        // stores vertices in the same order as in the off file
        h_jpv[2*vmap_old_new[i]  ] = k; //read start position
        h_jpv[2*vmap_old_new[i]+1] = p; //number of joints modifying that vertex
        k += p;
    } // END FOR NB lINES

    // Copy weights to device mem
    d_jpv = h_jpv;
    d_weights.resize(k);
    d_weights=h_weights;
    d_joints.resize(k);
    d_joints=h_joints;

    update_host_ssd_weights();

    cout << "file \"" << filename << "\" loaded successfully" << endl;
    set_default_bones_radius();
    file.close();

//    std::cout << "COMPUTE NEW CLUSTER FROM WEIGHTS" << std::endl;
//    clusterize( EAnimesh::FROM_WEIGHTS );/////////////////////DEBUG//////////////
}

// -----------------------------------------------------------------------------

int Animesh::pack_vert_to_fit(std::vector<int>& in,
                                   std::vector<int>& out,
                                   int size)
{
 //   Cuda_utils::mem_cpy_dth(in.ptr(), d_vert_to_fit.ptr(), size);	 
	in = d_vert_to_fit;
    int j = 0;
    for(int i = 0; i < size; i++)
    {
        int elt = in[i];
        if(elt != -1)
        {
            out[j] = elt;
            j++;
        }
    }
	out = in;
//    Cuda_utils::mem_cpy_htd(d_vert_to_fit.ptr(), out.ptr(), j);
    return j;
}

// -----------------------------------------------------------------------------

//#include "toolbox/cuda_utils/cuda_utils_thrust.hpp"

static void transform_vert_to_fit(const int* src, int* dst, const int nb_vert)
{
    //const int p = blockIdx.x * blockDim.x + threadIdx.x;
    //if(p < nb_vert) dst[p] = src[p] < 0 ? 0 : 1;
}

/// here src must be different from dst
static
void pack(const int* prefix_sum, const int* src, int* dst, const int nb_vert)
{
    //const int p = blockIdx.x * blockDim.x + threadIdx.x;
    //if(p < nb_vert){
    //    const int elt = src[p];
    //    if(elt >= 0) dst[ prefix_sum[p] ] = elt;
    //}
}

int Animesh::pack_vert_to_fit_gpu(
        std::vector<int>& d_vert_to_fit,
        std::vector<int>& buff,
        std::vector<int>& packed_array,
        int nb_vert_to_fit)
{
#if 0
    if(nb_vert_to_fit == 0) return 0;
    assert(d_vert_to_fit.size() >= nb_vert_to_fit           );
    assert(buff.size()          >= d_vert_to_fit.size() + 1 );
    assert(packed_array.size()  >= d_vert_to_fit.size()     );

    const int block_s = 16;
    const int grid_s  = (nb_vert_to_fit + block_s - 1) / block_s;
    transform_vert_to_fit<<<grid_s, block_s >>>(d_vert_to_fit.ptr(), buff.ptr()+1, nb_vert_to_fit);
    buff.set(0, 0);// First element to zero
    CUDA_CHECK_ERRORS();

    // Compute prefix sum in buff between [1 nb_vert_to_fit]
    Cuda_utils::inclusive_scan(0, nb_vert_to_fit-1, buff.ptr()+1);

    const int new_nb_vert_to_fit = buff.fetch(nb_vert_to_fit);

    pack<<<grid_s, block_s >>>(buff.ptr(), d_vert_to_fit.ptr(), packed_array.ptr(), nb_vert_to_fit);
    CUDA_CHECK_ERRORS();

    //Cuda_utils::mem_cpy_dtd(d_vert_to_fit.ptr(), packed_array.ptr(), new_nb_vert_to_fit);

    return new_nb_vert_to_fit;
#endif
	return -1;
}

// -----------------------------------------------------------------------------

void Animesh::update_opengl_buffers(int nb_vert,
                                    const std::vector<Tbx::Vec3>& vert,  //packed_vert
                                    const std::vector<Tbx::Vec3>& normals, //packed_normals
                                    const std::vector<Tbx::Vec3>& tangents, //packed_tangents
                                    GlBuffer_obj<Vec3>* vbo,
                                    GlBuffer_obj<Vec3>* nbo,
                                    GlBuffer_obj<Vec3>* tbo)
{

	Vec4* color_ptr;
	const int size = _mesh->_mesh_attr._size_unpacked_verts;

	std::vector<Tbx::Vec3>  unpacked_vert(size);  //unpacked_vert
	std::vector<Tbx::Vec3>  unpacked_normals(size); //unpacked_normals
	std::vector<Tbx::Vec3>  unpacked_tangents(size);
//	_mesh->_mesh_attr._packed_vert_map._point_color_bo.map_to(color_ptr, GL_WRITE_ONLY);
	auto& unp_vertmap = _mesh->_mesh_attr._packed_vert_map;
	for (int i = 0; i < nb_vert; i++)
	{
		Vec3 pv = vert[i];
		Vec3 pn = normals[i];
		Vec3 pt;
		if(tangents.size() > i+1)
			pt = tangents[i];
		const EMesh::Packed_data& d = unp_vertmap[i];
		int idx = d._idx_data_unpacked;
		for (int j = 0; j < d._nb_ocurrence; j++)
		{
			unpacked_vert    [idx+j] = pv;
			unpacked_normals [idx+j] = pn;
			if( tangents.size() > i+1)
				unpacked_tangents[idx+j] = pt;
		}
	}
	if(vert.size())
		vbo->bind();
	if(normals.size())
		nbo->bind();
	if(tangents.size())
		tbo->bind();

	if(vert.size())
		vbo->set_data(unpacked_vert);
	if(normals.size())
		nbo->set_data(unpacked_normals);
	if(tangents.size())
		tbo->set_data(unpacked_tangents);	

	if(vert.size())
		vbo->unbind();
	if(normals.size())
		nbo->unbind();
	if(tangents.size())
		tbo->unbind();
}

// -----------------------------------------------------------------------------

void Animesh::reset_flip_propagation(){
    int nb_vert = _mesh->get_nb_vertices();
    std::vector<bool> flip_prop(nb_vert);
    for(int i = 0; i < nb_vert; i++){
        flip_prop[i] = false;
    }

    d_flip_propagation = flip_prop;
}

// -----------------------------------------------------------------------------

void Animesh::init_sum_angles()
{
    hd_1st_ring_angle.resize( d_1st_ring_list.size() );
    int nb_vert = _mesh->get_nb_vertices();
    for(int i = 0; i < nb_vert; i++)
    {
        // Compute at each vertex the sum of all incident edges
        Vec3 pos = _mesh->get_vertex(i);
        float sum_angles = 0.f;
        int dep      = _mesh->get_1st_ring_offset(i*2    );
        int nb_neigh = _mesh->get_1st_ring_offset(i*2 + 1);
        int end      = dep + nb_neigh;
        for(int n = dep; n < (dep+nb_neigh); n++)
        {
            int curr = _mesh->get_1st_ring( n );
            int next = _mesh->get_1st_ring( (n+1) >= end  ? dep : n+1 );

            Vec3 edge0 = _mesh->get_vertex( curr ) - pos;
            Vec3 edge1 = _mesh->get_vertex( next ) - pos;

            float angle = acos(edge0.normalized().dot(edge1.normalized()));//hd_gradient[i].signed_angle( edge0.normalized(), edge1.normalized() );
            //std::cout << angle << std::endl;
            hd_1st_ring_angle[n] = angle;
            sum_angles += angle;
        }

        hd_sum_angles[i] = sum_angles;
    }
    //hd_sum_angles.update_device_mem();
    //hd_1st_ring_angle.update_device_mem();
}

void Animesh::set_color_ssd_weight(int joint_id)
{
	mesh_color = EAnimesh::SSD_WEIGHTS;

	const int vtx_num = d_input_vertices.size();
	EMesh::Packed_data* d_map = &d_packed_vert_map[0];
	
	_mesh->_mesh_gl._color_bo.bind();
	std::vector<Tbx::Vec4> d_colors;
	d_colors.resize( _mesh->_mesh_attr._size_unpacked_verts );
	Animesh_colors::ssd_weights_colors_kernel(d_colors,_mesh->_mesh_attr._packed_vert_map, joint_id, h_weights);
	_mesh->_mesh_gl._color_bo.set_data(d_colors);

	_mesh->_mesh_gl._color_bo.unbind();
}




void Animesh::compute_pcaps(int bone_id, bool use_parent_dir, std::vector<Tbx::Vec3>& out_verts, std::vector<Tbx::Vec3>& out_normals)
{
	cout<<"compute_pcaps"<<endl;
}

void Animesh::compute_jcaps(int bone_id, std::vector<Tbx::Vec3>& out_verts, std::vector<Tbx::Vec3>& out_normals)
{
	cout<<"compute_jcaps"<<endl;
}



void Animesh::set_colors(EAnimesh::Color_type type /*= EBASE_POTENTIAL*/, 
						 float r /*= 0.7f*/, 
						 float g /*= 0.7f*/,
						 float b /*= 0.7f*/, 
						 float a /*= 1.f */)
{
	mesh_color = type;

    const int nb_vert = _mesh->get_nb_vertices();
	std::vector<Tbx::Vec4> d_colors;
	d_colors.resize( _mesh->_mesh_attr._size_unpacked_verts );
    _mesh->_mesh_gl._color_bo.bind();

    switch(type)
    {
    case EAnimesh::BASE_POTENTIAL:
        Animesh_colors::base_potential_colors_kernel(d_colors, _mesh->_mesh_attr._packed_vert_map,d_base_potential);
		_mesh->_mesh_gl._color_bo.set_data(d_colors);
        break;
    case EAnimesh::GRAD_POTENTIAL:
    case EAnimesh::MVC_SUM:
        /////////////////////////////////////////////////////////////////////////// FIXME: TODO: use a proper enum

        //Animesh_colors::gradient_potential_colors_kernel<<<grid_size, block_size>>>
        //    (_skel->skel_id(), d_colors, d_map, (Vec3*)hd_output_vertices.d_ptr(), hd_output_vertices.size());
        /*
        Animesh_colors::mvc_colors_kernels<<<grid_size, block_size>>>
            (d_colors, d_map, d_1st_ring_list.ptr(), d_1st_ring_list_offsets.ptr(), hd_1st_ring_mvc.d_ptr(), nb_vert);
        */

        break;
    case EAnimesh::SSD_INTERPOLATION:
        //Animesh_colors::ssd_interpolation_colors_kernel<<<grid_size, block_size>>>
        //    (d_colors, d_map, hd_ssd_interpolation_factor.device_array());
        break;
    case EAnimesh::SMOOTHING_WEIGHTS:
        //Animesh_colors::smoothing_colors_kernel<<<grid_size, block_size>>>
        //    (d_colors, d_map, d_input_smooth_factors);
        break;
    case EAnimesh::ANIM_SMOOTH_LAPLACIAN:
        //Animesh_colors::smoothing_colors_kernel<<<grid_size, block_size>>>
        //    (d_colors, d_map, d_smooth_factors_laplacian);
        break;
    case EAnimesh::ANIM_SMOOTH_CONSERVATIVE:
        //Animesh_colors::smoothing_colors_kernel<<<grid_size, block_size>>>
        //    (d_colors, d_map, d_smooth_factors_conservative);
        break;
    case EAnimesh::NEAREST_JOINT:
        //Animesh_colors::nearest_joint_colors_kernel<<<grid_size, block_size>>>
        //    (d_colors,  d_map, d_vertices_nearest_joint);
        break;
    case EAnimesh::CLUSTER:
        //Animesh_colors::cluster_colors_kernel<<<grid_size, block_size>>>
        //    (d_colors,  d_map, d_vertices_nearest_bones/*d_nearest_bone_in_device_mem*/);
        break;
    case EAnimesh::NORMAL:
    {
        //Vec3* normals = (Vec3*)d_output_normals.ptr();
        //Animesh_colors::normal_colors_kernel<<<grid_size, block_size>>>
        //    (d_colors, d_map, normals, _mesh->get_nb_vertices());
    }
        break;
    case EAnimesh::USER_DEFINED:
    {
        //float4 color = make_float4(r, g, b, a);
        //Animesh_colors::user_defined_colors_kernel <<<grid_size, block_size>>>
        //        (d_colors, d_map, color,  _mesh->get_nb_vertices());
    }
        break;
    case EAnimesh::VERTICES_STATE:
    {

// Show if a vertex is moved by the arap relaxation scheme:
        //Animesh_colors::displacement_colors_kernel<<<grid_size, block_size>>>
        //    (d_colors,  d_map, hd_displacement.device_array());


    }
        break;
    case EAnimesh::FREE_VERTICES:
    {

        // Color vertices
        //Animesh_colors::free_vert_colors_kernel<<<grid_size, block_size>>>
        //    (d_colors,
        //     d_map,
        //     hd_map_verts_to_free_vertices.d_ptr(),
        //     hd_free_vertices.d_ptr(),
        //     nb_vert);

    }
        break;
    case EAnimesh::EDGE_STRESS:
    {

        //Animesh_colors::edge_stress_colors_kernel<<<grid_size, block_size>>>
        //    (d_colors, d_map, hd_edge_list.d_ptr(), (Vec3*)d_input_vertices.ptr(),
        //     (Vec3*)hd_output_vertices.d_ptr(), d_1st_ring_list_offsets.ptr(),
        //     hd_1st_ring_edges.d_ptr(), nb_vert);

    }
        break;

    case EAnimesh::GAUSS_CURV:
    {
        //Animesh_colors::gaussian_curvature_colors_kernel<<<grid_size, block_size>>>(
        //                                 d_colors,
        //                                 d_map,
        //                                 (Vec3*)hd_output_vertices.d_ptr(),
        //                                 d_1st_ring_list.ptr(),
        //                                 d_1st_ring_list_offsets.ptr(),
        //                                 nb_vert,
        //                                 Cuda_ctrl::_debug._val0);
    }
        break;
    case EAnimesh::SSD_WEIGHTS:
 //       assert(false); // use the method set_color_ssd_weight()
        break;
    default:
        std::cout << "sorry the color scheme you ask is not implemented" << std::endl;
        break;
    }
    _mesh->_mesh_gl._color_bo.unbind();
}

void Animesh::init_cotan_weights()
{
	cout<<"init_cotan_weights"<<endl;
}

// -----------------------------------------------------------------------------

void Animesh::Adhoc_sampling::sample(std::vector<Tbx::Vec3>& out_verts, std::vector<Tbx::Vec3>& out_normals)
{
	cout<<"Adhoc_sampling sample "<<endl;
}

void Animesh::Poisson_disk_sampling::sample(std::vector<Tbx::Vec3>& out_verts, std::vector<Tbx::Vec3>& out_normals)
{
	cout<<"Poisson_disk_sampling sample "<<endl;
}
