#ifndef MESH_UNPACKED_ATRR_HPP__
#define MESH_UNPACKED_ATRR_HPP__

#include <vector>

#include "toolbox/maths/vec3.hpp"
#include "toolbox/maths/vec2.hpp"
#include "mesh_types.hpp"
#include "mesh_materials.hpp"

// TODO: forward defs to be deleted
class Mesh;

/** @class Mesh_unpacked_attr
    @name Opengl ready Mesh's unpacked attributes
    This class store mesh attributes (color, normals, positions) and the
    corresponding face list ready to use with OpenGL.

    Opengl can't really fetch multiple attributes for one vertex.
    The class aim to prepare vertices with multiples attributes
    (tex coords at seams for instance) to be duplicated.
    For instance if a vertex has two normals we will duplicate it at the same
    position and assiociate the first normal to the first duplicate and so on.

    TODO: document usage
**/
struct Mesh_unpacked_attr {

    //_material_list(m._material_list) // FIXME: implement recopy for Material -> for GLTex2D as well ***

    Mesh_unpacked_attr(Mesh& mesh) :
        _mesh(mesh),
        _size_unpacked_verts(-1)
    { }

    ~Mesh_unpacked_attr() {
        clear_data();
    }

    //Mesh_unpacked(Mesh m) build from mesh with default _packed_map (no duplicates)


    //Mesh_unpacked(Mesh m, _unpacked_tri*, _unpacked_quad*) build from mesh

    /*
     // todo: to be deleted
    /// fill the attributes 'material_grps_tri' 'material_grps_quad'
    /// 'material_list'
    void build_material_lists(const Loader::Abs_mesh& mesh,
                              const std::string& mesh_path);
                              */

    /// in order to accelerate the rendering we regroup the index of faces
    /// in order to be contigus by material type
    void regroup_faces_by_material();

    /// We use the ugly alpha blending to render transparent materials
    /// this function regroups their rendering at the end.
    void regroup_transcelucent_materials();

    void clear_data();

    // -------------------------------------------------------------------------
    /// @name Accessors
    // -------------------------------------------------------------------------

    EMesh::Tri_face get_unpacked_tri(EMesh::Tri_idx i) const {
        assert( i < get_nb_tris() );
        assert( i >= 0 );
        const std::vector<EMesh::Vert_idx>& tri = _unpacked_tri;
        return EMesh::Tri_face(tri[i*3], tri[i*3 + 1], tri[i*3 + 2]);
    }

    EMesh::Quad_face get_unpacked_quad(EMesh::Quad_idx i) const
    {
        assert( i < get_nb_quads() );
        assert( i >= 0 );
        const std::vector<EMesh::Vert_idx>& quad = _unpacked_quad;
        return EMesh::Quad_face( quad[i*4],
                                 quad[i*4 + 1],
                                 quad[i*4 + 2],
                                 quad[i*4 + 3]);
    }

    void set_unpacked_tri(EMesh::Tri_idx i, const EMesh::Tri_face& tri) {
        assert( i < get_nb_tris() );
        assert( i >= 0 );
        _unpacked_tri[i*3 + 0] = tri.a;
        _unpacked_tri[i*3 + 1] = tri.b;
        _unpacked_tri[i*3 + 2] = tri.c;
    }

    void set_unpacked_quad(EMesh::Quad_idx i, const EMesh::Quad_face& quad)
    {
        assert( i < get_nb_quads() );
        assert( i >= 0 );
        _unpacked_quad[i*4    ] = quad.a;
        _unpacked_quad[i*4 + 1] = quad.b;
        _unpacked_quad[i*4 + 2] = quad.c;
        _unpacked_quad[i*4 + 3] = quad.d;
    }

    void resize_tris(int nb_tris){ _unpacked_tri.resize( nb_tris * 3 ); }

    int get_nb_tris() const { return _unpacked_tri.size() / 3; }

    int get_nb_quads() const { return _unpacked_quad.size() / 3; }

    /// For each vertex copy the vector of "normals". If the vertex is
    /// duplicated (to be able to hold several attributes) we copy the normal
    /// for each duplicates
    void set_normals(const std::vector<Tbx::Vec3>& normals);

    void set_tangents(const std::vector<Tbx::Vec3>& tangents);

    void set_nb_attributes(const std::vector<int>& nb_attr_per_vertex);

    void set_normal(EMesh::Vert_idx i, int attr_num, const Tbx::Vec3& n)
    {
        int id = i + attr_num;
        _normals[id * 3 + 0] = n.x;
        _normals[id * 3 + 1] = n.y;
        _normals[id * 3 + 2] = n.z;
    }

    void set_tangent(EMesh::Vert_idx i, int attr_num, const Tbx::Vec3& t)
    {
        int id = i + attr_num;
        _tangents[id * 3 + 0] = t.x;
        _tangents[id * 3 + 1] = t.y;
        _tangents[id * 3 + 2] = t.z;
    }

    void set_tex_coords(EMesh::Vert_idx i, int attr_num, const Tbx::Vec2& t)
    {
        int id = i + attr_num;
        _tex_coords[id * 2 + 0] = t.x;
        _tex_coords[id * 2 + 1] = t.y;
    }


    int get_nb_vertices() const {  return (int)_packed_vert_map.size(); }

    const EMesh::Packed_data& get_packed_verts_map(EMesh::Vert_idx i) const {
        return _packed_vert_map[i];
    }

    Mesh& _mesh; // <- todo to be deleted ///////////////////////////////////////////////////////////

    // -------------------------------------------------------------------------
    /// @name Attributes
    // -------------------------------------------------------------------------
    // TODO: Should materials be handled separatly ?
//private:
    // TODO: provide accessors with type Vec3 Vec2 and then simply change internal type
    std::vector<float> _normals;     ///< Normal direction list [N0x N0y N0z N1x N1y N1z ...]
    std::vector<float> _tangents;    ///< Tangent direction list [T0x T0y T0z T1x T1y T1z ...]
    std::vector<float> _tex_coords;  ///< Texture coordinates list [T0u T0v T1u T1v ...]

    /// size of the vbo for the rendering. '_normals' and '_tangents' size are
    /// 3*size_unpacked_vert_array and '_tex_coords' is 2*size_unpacked_vert_array
    int _size_unpacked_verts;

    std::vector<int> _unpacked_tri;  ///< unpacked triangle index ( size == nb_tri)
    std::vector<int> _unpacked_quad; ///< unpacked quad index (size == nb_quad)

    /// Mapping between the packed array of vertices 'vert' and the unpacked
    /// array of vertices 'vbo'. because vertices have multiple texture
    /// coordinates we have to duplicate them. we do that for the rendering
    /// by duplicating the original data in 'vert' into the vbos.
    /// packed_vert_map[packed_vert_idx] = mapping to unpacked.
    /// size of 'packed_vert_map' equals 'nb_vert'
    std::vector<EMesh::Packed_data> _packed_vert_map;

    /// List of material bound to a face index group in unpacked_tri
    std::vector<EMesh::Mat_grp>   _material_grps_tri;
    /// List of material bound to a face index group in unpacked_quad
    std::vector<EMesh::Mat_grp>   _material_grps_quad;
    /// List of material definitions (coeffs Ka, Kd, textures, bump map etc.)
    std::vector<EMesh::Material>  _material_list;

};

#endif // MESH_UNPACKED_ATRR_HPP__
