#include "meshes/mesh_utils_loader.hpp"

#include "toolbox/utils.hpp"
#include "toolbox/maths/vec2.hpp"
#include "parsers/tex_loader.hpp"
#include <map>
using namespace Tbx;
// =============================================================================
namespace Mesh_utils {
// =============================================================================



static inline Vec3 to_vec(const Loader::Normal&    n) { return Vec3(n.x, n.y, n.z); }
static inline Vec3 to_vec(const Loader::Vertex&    v) { return Vec3(v.x, v.y, v.z); }
static inline Vec2 to_vec(const Loader::Tex_coord& t) { return Vec2(t.u, t.v);      }

// -----------------------------------------------------------------------------

void build_material_lists(Mesh& out_mesh,
                          const Loader::Abs_mesh& in_mesh,
                          const std::string& mesh_path)
{
    out_mesh._has_materials = true;
    out_mesh._mesh_attr._material_grps_tri.clear();
    out_mesh._mesh_attr._material_grps_quad.clear();
    int last_mat = in_mesh._materials.size();
    for(unsigned i = 0; i < in_mesh._groups.size(); i++)
    {
        const Loader::Group& grp = in_mesh._groups[i];
        // No materials ? we take the previous material or a default one
        if(grp._assigned_mats.size() == 0)
        {
            EMesh::Mat_grp mesh_mat_grp;
            mesh_mat_grp.starting_idx = grp._start_face;
            mesh_mat_grp.nb_face      = grp._end_face - grp._start_face;
            mesh_mat_grp.mat_idx      = last_mat;

            out_mesh._mesh_attr._material_grps_tri.push_back(mesh_mat_grp);
        }

        for(unsigned j = 0; j < grp._assigned_mats.size(); j++)
        {
            const Loader::Material_group& file_mat_grp = grp._assigned_mats[j];
            last_mat = file_mat_grp._material_idx;

            EMesh::Mat_grp mesh_mat_grp;
            mesh_mat_grp.starting_idx = file_mat_grp._start_face;
            mesh_mat_grp.nb_face      = file_mat_grp._end_face - file_mat_grp._start_face;
            mesh_mat_grp.mat_idx      = file_mat_grp._material_idx;

            out_mesh._mesh_attr._material_grps_tri.push_back(mesh_mat_grp);
        }
    }

    const std::string path = mesh_path;

    // Copy materials And load textures :
    out_mesh._mesh_attr._material_list.resize(in_mesh._materials.size()+1);
    out_mesh._has_bumpmap = false;
    for(unsigned i = 0; i < in_mesh._materials.size()+1; i++)
    {
        Loader::Material mat;

        // Last material is a default material :
        if(i == in_mesh._materials.size()) mat = Loader::Material();
        else                            mat = in_mesh._materials[i];

        EMesh::Material mesh_mat;

        mesh_mat._name = mat._name;
        // Copy coeffs
        mesh_mat._bump_strength = mat._Bm;
        for(int j = 0; j < 4; j++){
            mesh_mat._ka[j] = mat._Ka[j];
            mesh_mat._kd[j] = mat._Kd[j];
            mesh_mat._ks[j] = mat._Ks[j];
        }
        for(int j = 0; j < 3; j++)
            mesh_mat._tf[j] = mat._Tf[j];

        mesh_mat._ni = mat._Ni;
        mesh_mat.set_ns(mat._Ns);

        // Load textures (read file and create the opengl texture)
        mesh_mat._map_bump = mat._map_Bump.size() ? Loader::Tex_loader::load(path+mat._map_Bump) : 0;
        mesh_mat._map_ka   = mat._map_Ka.size()   ? Loader::Tex_loader::load(path+mat._map_Ka)   : 0;
        mesh_mat._map_kd   = mat._map_Kd.size()   ? Loader::Tex_loader::load(path+mat._map_Kd)   : 0;
        mesh_mat._map_ks   = mat._map_Ks.size()   ? Loader::Tex_loader::load(path+mat._map_Ks)   : 0;
        // Compy texture names
        mesh_mat._file_path_ka   = mat._map_Ka;
        mesh_mat._file_path_kd   = mat._map_Kd;
        mesh_mat._file_path_ks   = mat._map_Ks;
        mesh_mat._file_path_bump = mat._map_Bump;

        out_mesh._mesh_attr._material_list[i] = mesh_mat;

        out_mesh._has_bumpmap = out_mesh._has_bumpmap || mesh_mat._map_bump != 0;
    }
}

// -----------------------------------------------------------------------------

void load_mesh(Mesh& out_mesh, const Loader::Abs_mesh& in_mesh)
{
    Mesh_unpacked_attr& mesh_attr = out_mesh._mesh_attr;

    out_mesh._is_initialized = false;
    out_mesh.clear_data();

    // For now the loader triangulates every faces
    int nb_verts = in_mesh._vertices.size();
    int nb_tri   = in_mesh._triangles.size();

    out_mesh._mesh_static.resize_vertices( nb_verts );
    out_mesh.resize_tris( nb_tri );

    // Copy vertex coordinates
    for( int i = 0; i < nb_verts; i++)
    {
        Loader::Vertex v = in_mesh._vertices[i];
        out_mesh._mesh_static.set_vertex(i, Vec3(v.x, v.y, v.z));
    }

    // Build the list of texture coordinates indices and normals indices per
    // vertices indices.
    // Also we copy the packed triangle index in '_tri'
    std::vector<std::map<std::pair<int,int>,int> > pair_per_vert(nb_verts);
    std::vector<int> nb_pair_per_vert(nb_verts, 0);
    for( int i = 0; i < nb_tri; i++)
    {
        EMesh::Tri_face tri;
        for(int j = 0; j < 3; j++)
        {
            // Fill triangle index
            int v_idx = in_mesh._triangles[i].v[j]; // Vertex index
            int n_idx = in_mesh._triangles[i].n[j]; // normal index
            int t_idx = in_mesh._triangles[i].t[j]; // texture index

            tri[j] = v_idx;

            std::pair<int, int> pair(t_idx, n_idx);
            std::map<std::pair<int, int>,int>& map = pair_per_vert[v_idx];
            if( map.find(pair) == map.end() )
            {
                map[pair] = nb_pair_per_vert[v_idx];
                nb_pair_per_vert[v_idx]++;
            }
        }

        out_mesh._mesh_static.set_tri(i, tri);
    }

    // We now build the mapping between packed vertex coordinates and unpacked
    // vertex coortinates, so that each vertex in the unpacked form has its own
    // texture coordinates and/or normal direction.
    mesh_attr.set_nb_attributes( nb_pair_per_vert );

    // Copy triangles index, normals and texture coordinates
    out_mesh._has_tex_coords = false;
    out_mesh._has_normals    = false;
    for( int i = 0; i < nb_tri; i++)
    {
        EMesh::Tri_face unpack_tri;
        for( int j = 0; j < 3; j++)
        {
            int v_idx = in_mesh._triangles[i].v[j];
            int n_idx = in_mesh._triangles[i].n[j];
            int t_idx = in_mesh._triangles[i].t[j];

            std::pair<int, int> pair(t_idx, n_idx);
            int off = pair_per_vert[v_idx][pair];

            assert(off < mesh_attr.get_packed_verts_map(v_idx)._nb_ocurrence);

            int v_unpacked = mesh_attr.get_packed_verts_map(v_idx)._idx_data_unpacked;

            // Fill unpacked triangle index
            unpack_tri[j] = v_unpacked + off;

            // Fill normal as there index match the unpacked vertex array
            if( n_idx != -1 )
                mesh_attr.set_normal(v_unpacked, off, to_vec( in_mesh._normals[n_idx] ) );
            else
                mesh_attr.set_normal(v_unpacked, off, to_vec( Loader::Normal() ) );

            // Fill texture coordinates as there index match the unpacked vertex array
            if( t_idx != -1 )
                mesh_attr.set_tex_coords(v_unpacked, off, to_vec( in_mesh._texCoords[t_idx] ) );
            else
                mesh_attr.set_tex_coords(v_unpacked, off, to_vec( Loader::Tex_coord() ) );

            out_mesh._has_tex_coords = out_mesh._has_tex_coords || (t_idx != -1);
            out_mesh._has_normals    = out_mesh._has_normals    || (n_idx != -1);
        }

        mesh_attr.set_unpacked_tri(i, unpack_tri);
    }

    // Copy materials groups :
    if(in_mesh._groups.size())
    {
        //out_mesh._mesh_attr.build_material_lists(in_mesh, in_mesh._mesh_path);
        build_material_lists(out_mesh, in_mesh, in_mesh._mesh_path);
        out_mesh._mesh_attr.regroup_faces_by_material();
        out_mesh._mesh_attr.regroup_transcelucent_materials();
    }

    /////////////FIXME: the loaded armadillo_LAURA normals are wrong see if its the file or FBX loader in the meantime I recompute the normals
    //if( !out_mesh._has_normals )
        out_mesh.compute_normals();

    if(out_mesh._has_tex_coords)
        out_mesh.compute_tangents();
    // Initialize VBOs
    out_mesh._mesh_gl.alloc_gl_buffer_objects();
    out_mesh._mesh_he.update( out_mesh._mesh_static );
    out_mesh._is_initialized = true;
}

// -----------------------------------------------------------------------------

void save_mesh(const Mesh& in_mesh, Loader::Abs_mesh& out_mesh)
{
    // FIXME: quads must be handle as well and copied

    // Copy vertices array
    out_mesh._vertices.clear();
    out_mesh._vertices.resize( in_mesh.get_nb_vertices() );
    for(unsigned i = 0; i < out_mesh._vertices.size(); ++i){
        Vec3 v = in_mesh._mesh_static.get_vertex(i);
        out_mesh._vertices[i] = Loader::Vertex(v.x, v.y, v.z);
    }

    // Copy normals array
    out_mesh._normals.clear();
    out_mesh._normals.resize( in_mesh._mesh_attr._size_unpacked_verts );
    for(unsigned i = 0; i < out_mesh._normals.size(); ++i){
        Vec3 v(in_mesh._mesh_attr._normals[i*3], in_mesh._mesh_attr._normals[i*3+1], in_mesh._mesh_attr._normals[i*3+2]);
        out_mesh._normals[i] = Loader::Normal(v.x, v.y, v.z);
    }

    // Copy texture coordinates array
    out_mesh._texCoords.clear();
    out_mesh._texCoords.resize( in_mesh._mesh_attr._size_unpacked_verts );
    for(unsigned i = 0; i < out_mesh._texCoords.size(); ++i)
        out_mesh._texCoords[i] = Loader::Tex_coord( in_mesh._mesh_attr._tex_coords[i*2], in_mesh._mesh_attr._tex_coords[i*2+1] );

    // Copy face index array
    out_mesh._triangles.clear();
    out_mesh._triangles.resize( in_mesh.get_nb_tris() );
    for(unsigned i = 0; i < out_mesh._triangles.size(); ++i){
        Loader::Tri_face F;
        // Vertex index
        F.v[0] = in_mesh._mesh_static.get_tri(i)[0];
        F.v[1] = in_mesh._mesh_static.get_tri(i)[1];
        F.v[2] = in_mesh._mesh_static.get_tri(i)[2];
        // Normal index
        const int* ptr = (int*)in_mesh._mesh_attr._unpacked_tri.data();
        F.n[0] = ptr[i*3]; F.n[1] = ptr[i*3+1]; F.n[2] = ptr[i*3+2];
        // Tex coords index
        F.t[0] = ptr[i*3]; F.t[1] = ptr[i*3+1]; F.t[2] = ptr[i*3+2];

        out_mesh._triangles[i] = F;
    }

    // Copy Material list
    out_mesh._materials.clear();
    out_mesh._materials.resize( in_mesh._mesh_attr._material_list.size() );
    for(unsigned i = 0; i < out_mesh._materials.size(); ++i) {
        const EMesh::Material& m = in_mesh._mesh_attr._material_list[i];
        Loader::Material M;
        M._name = m._name;
        M._illum = 4;
        Utils::copy(M._Ka, m._ka, 4);
        Utils::copy(M._Kd, m._kd, 4);
        Utils::copy(M._Ks, m._ks, 4);
        Utils::copy(M._Tf, m._tf, 3);
        M._Ni = m._ni;
        M._Ns = m.get_ns();
        M._map_Ka = m._file_path_ka; M._map_Kd = m._file_path_kd;
        M._map_Ks = m._file_path_ks; M._map_Bump = m._file_path_bump;
        M._Bm = m._bump_strength;
        out_mesh._materials[i] = M;
    }

    // copy groups (we don't actually handle groups so we just one big root
    // group which contains all the materials groups)
    out_mesh._groups.clear();
    out_mesh._groups.resize( 1 );

    Loader::Group G;
    G._start_face = 0;
    G._end_face = out_mesh._triangles.size();
    //G.start_point = 0;
    //G._end_point = 0;
    G._name = "_";

    G._assigned_mats.resize( in_mesh._mesh_attr._material_grps_tri.size() );
    for(unsigned i = 0; i < G._assigned_mats.size(); ++i){
        const EMesh::Mat_grp& m = in_mesh._mesh_attr._material_grps_tri[i];
        Loader::Material_group MG;
        MG._start_face = m.starting_idx;
        MG._end_face = m.starting_idx + m.nb_face;
        //MG._start_point = 0;
        //MG._end_point = 0;
        MG._material_idx = m.mat_idx;
        G._assigned_mats[i] = MG;
    }
    out_mesh._groups[0] = G;
}

}// END Mesh_utils NAMESPACE ===================================================
