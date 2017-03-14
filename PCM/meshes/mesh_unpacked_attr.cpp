#include "mesh_unpacked_attr.hpp"
#include "mesh.hpp"

// -----------------------------------------------------------------------------

void Mesh_unpacked_attr::clear_data()
{
    // Delete opengl textures
    for(unsigned i = 0; i < _material_list.size(); i++)
    {
        delete _material_list[i]._map_ka;
        delete _material_list[i]._map_kd;
        delete _material_list[i]._map_ks;
        delete _material_list[i]._map_bump;
    }

    _packed_vert_map.clear();
    _unpacked_tri.clear();
    _unpacked_quad.clear();
    _normals.clear();
    _tangents.clear();
    _tex_coords.clear();
    _material_grps_tri.clear();
    _material_grps_quad.clear();
    _material_list.clear();
}

// -----------------------------------------------------------------------------

void Mesh_unpacked_attr::set_normals(const std::vector<Tbx::Vec3>& normals)
{
    int n_size = _size_unpacked_verts;
    _normals.resize( n_size * 3 );
    for(int i = 0; i < get_nb_vertices(); i++)
    {
        const EMesh::Packed_data d = _packed_vert_map[i];
        for(int j = 0; j < d._nb_ocurrence; j++)
        {
            const int p_idx = d._idx_data_unpacked + j;
            Tbx::Vec3 n = normals[i];
            _normals[p_idx*3    ] = n.x;
            _normals[p_idx*3 + 1] = n.y;
            _normals[p_idx*3 + 2] = n.z;
        }
    }
}

// -----------------------------------------------------------------------------

void Mesh_unpacked_attr::set_tangents(const std::vector<Tbx::Vec3>& tangents)
{
    int n_size = _size_unpacked_verts;
    _tangents.resize( n_size * 3 );
    for(int i = 0; i < get_nb_vertices(); i++)
    {
        const EMesh::Packed_data d = _packed_vert_map[i];
        for(int j = 0; j < d._nb_ocurrence; j++)
        {
            const int p_idx = d._idx_data_unpacked + j;
            Tbx::Vec3 n = tangents[i];
            _tangents[p_idx*3    ] = n.x;
            _tangents[p_idx*3 + 1] = n.y;
            _tangents[p_idx*3 + 2] = n.z;
        }
    }
}

// -----------------------------------------------------------------------------

void Mesh_unpacked_attr::set_nb_attributes(const std::vector<int>& nb_attr_per_vertex)
{
    int nb_verts = (int)nb_attr_per_vertex.size();
    _packed_vert_map.resize( nb_verts );
    int off = 0;
    for( int i = 0; i < nb_verts; i++)
    {
        int nb_elt = std::max(1, nb_attr_per_vertex[i]);

        EMesh::Packed_data tuple;
        tuple._idx_data_unpacked = off;
        tuple._nb_ocurrence      = nb_elt;

        _packed_vert_map[i] = tuple;

        off += nb_elt;
    }
    _size_unpacked_verts = off;
    _normals.   resize( _size_unpacked_verts * 3 );
    _tangents.  resize( _size_unpacked_verts * 3 );
    _tex_coords.resize( _size_unpacked_verts * 2 );
}



// -----------------------------------------------------------------------------

void Mesh_unpacked_attr::regroup_transcelucent_materials()
{
    // Transcluscent materials must be moved at the end of the material group list
    const int tri_grp_size  = _material_grps_tri. size();
    const int quad_grp_size = _material_grps_quad.size();

    // Switch the material group to the end
    int acc = tri_grp_size-1;
    for( int j = 0; j < acc; j++ )
    {
        int mat_idx = _material_grps_tri[j].mat_idx;
        const EMesh::Material& mat = _material_list[mat_idx];
        float average_transp = (mat._tf[0] + mat._tf[1] + mat._tf[2]) / 3.0f;

        if(average_transp <= (1.f - 0.001f))
        {
            EMesh::Mat_grp temp_grp = _material_grps_tri[j];
            _material_grps_tri[j  ] = _material_grps_tri[acc];
            _material_grps_tri[acc] = temp_grp;
            acc--;
        }
    }

    acc = quad_grp_size-1;
    for( int j = 0; j < acc; j++ )
    {
        int mat_idx = _material_grps_tri[j].mat_idx;
        const EMesh::Material& mat = _material_list[mat_idx];
        float average_transp = (mat._tf[0] + mat._tf[1] + mat._tf[2]) / 3.0f;

        if(average_transp <= (1.f - 0.001f))
        {
            EMesh::Mat_grp temp_grp = _material_grps_quad[j];
            _material_grps_quad[j  ] = _material_grps_quad[acc];
            _material_grps_quad[acc] = temp_grp;
            acc--;
        }
    }
}

// -----------------------------------------------------------------------------

void Mesh_unpacked_attr::regroup_faces_by_material()
{
    std::vector<EMesh::Tri_face>  new_tri          ( get_nb_tris()  );
    std::vector<EMesh::Tri_face>  new_unpacked_tri ( get_nb_tris()  );
    std::vector<EMesh::Quad_face> new_quad         ( get_nb_quads() );
    std::vector<EMesh::Quad_face> new_unpacked_quad( get_nb_quads() );

    int acc = 0;
    for(int mat_i = 0; mat_i < (int)_material_list.size(); mat_i++){
        for(int grp_i = 0; grp_i < (int)_material_grps_tri.size(); grp_i++){
            if(_material_grps_tri[grp_i].mat_idx == mat_i)
            {
                int face_start = _material_grps_tri[grp_i].starting_idx;
                int face_end   = face_start + _material_grps_tri[grp_i].nb_face;
                _material_grps_tri[grp_i].starting_idx = acc;
                for(int face_i = face_start; face_i < face_end; face_i++)
                {
                    new_tri         [acc] = _mesh._mesh_static.get_tri(face_i);
                    new_unpacked_tri[acc] = get_unpacked_tri(face_i);
                    acc++;
                }
            }
        }// END nb_mat_grp
    }// END nb_mat_list

    assert ( acc == _mesh._mesh_static.get_nb_tris() );

    acc = 0;
    for(int mat_i = 0; mat_i < (int)_material_list.size(); mat_i++){
        for(int grp_i = 0; grp_i < (int)_material_grps_quad.size(); grp_i++){
            if(_material_grps_quad[grp_i].mat_idx == mat_i)
            {
                int face_start = _material_grps_quad[grp_i].starting_idx;
                int face_end   = face_start + _material_grps_quad[grp_i].nb_face;
                _material_grps_quad[grp_i].starting_idx = acc;
                for(int face_i = face_start; face_i < face_end; face_i++)
                {
                    new_quad         [acc] = _mesh._mesh_static.get_quad(face_i);
                    new_unpacked_quad[acc] = get_unpacked_quad(face_i);
                    acc++;
                }
            }
        }// END nb_mat_grp
    }// END nb_mat_list

    assert ( acc == _mesh._mesh_static.get_nb_quads() );

    for( int i = 0; i < get_nb_tris(); ++i)
    {
        _mesh._mesh_static.set_tri(i, new_tri[i] );
        set_unpacked_tri(i, new_unpacked_tri[i] );
    }

    for( int i = 0; i < get_nb_quads(); ++i)
    {
        _mesh._mesh_static.set_quad(i, new_quad[i] );
        set_unpacked_quad(i, new_unpacked_quad[i] );
    }
}
