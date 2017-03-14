#ifndef MESH_STATIC_HPP__
#define MESH_STATIC_HPP__

#include <vector>
#include "mesh_types.hpp"
#include "toolbox/maths/vec3.hpp"

/**
 * @class Mesh_static
 * @brief Static mesh stored as arrays of triangles, quads and positions
 */
struct Mesh_static {

    Mesh_static() {}

    ~Mesh_static() { clear_data(); }

    void clear_data();

    void import_off(const char *filename);

    // TODO: delete scale and offset
    void export_off(const char* filename, bool invert_index, float scale, const Tbx::Vec3 &offset) const;


    // -------------------------------------------------------------------------
    /// @name Accessors
    // -------------------------------------------------------------------------

    Tbx::Vec3 get_vertex(EMesh::Vert_idx i) const { return _verts[i]; }

    void set_vertex(EMesh::Vert_idx i, const Tbx::Vec3& p) { _verts[i] = p; }

    int get_nb_tris() const { return _tris.size();  }
    int get_nb_quads() const { return _quads.size(); }

    int get_nb_faces() const { return get_nb_tris() + get_nb_quads(); }

    int get_nb_vertices() const { return _verts.size(); }

    const std::vector<Tbx::Vec3>& get_vertices() const { return _verts; }
    std::vector<Tbx::Vec3>& get_vertices() { return _verts; }

    void resize_vertices(int nb_vertices) { _verts.resize( nb_vertices ); }

    void resize_tris(int nb_tris) { _tris.resize(nb_tris); }

    /// Get three triangle index for the ith triangle
    EMesh::Tri_face get_tri(EMesh::Tri_idx i) const {
        assert( i < get_nb_tris() && i >= 0 );
        return _tris[i];
    }

    /// Get four quad index for the ith quad
    EMesh::Quad_face get_quad(EMesh::Quad_idx i) const {
        assert( i < get_nb_quads() && i >= 0 );
        return _quads[i];
    }

    void set_tri(EMesh::Tri_idx i, const EMesh::Tri_face& tri) {
        assert( i < get_nb_tris() && i >= 0 );
        _tris[i] = tri;
    }

    /// Get four quad index for the ith quad
    void set_quad(EMesh::Quad_idx i, const EMesh::Quad_face& quad)
    {
        assert( i < get_nb_quads() && i >= 0 );
        _quads[i] = quad;
    }

    const EMesh::Tri_face* get_tris() const { return &_tris.front(); }
    const EMesh::Quad_face* get_quads() const { return &_quads.front(); }

    // -------------------------------------------------------------------------
private:

    std::vector<Tbx::Vec3> _verts; /// List of vertex position
    std::vector<EMesh::Tri_face>  _tris;  ///< triangle index
    std::vector<EMesh::Quad_face> _quads; ///< quad index
};


#endif // MESH_STATIC_HPP__
