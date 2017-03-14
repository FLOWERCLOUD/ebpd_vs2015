#ifndef MESH_TYPES_HPP__
#define MESH_TYPES_HPP__

#include "assert.h"
#include "stdio.h"

/**
 * @namespace EMesh
 * @brief Holds data structure used by a #Mesh
 */
// =============================================================================
namespace EMesh {
// =============================================================================

typedef int Vert_idx; ///< Vertex index
typedef int Edge_idx; ///< Edge index
typedef int Face_idx; ///< Face index

typedef int Tri_idx;  ///< Triangle index
typedef int Quad_idx; ///< Quad index

typedef int Mat_idx;  ///< Material index

// -----------------------------------------------------------------------------

struct Tex_coords{

    Tex_coords() : u(0.f), v(0.f) { }

    Tex_coords(float u_, float v_) : u(u_), v(v_) { }
    float u,v;
};

// -----------------------------------------------------------------------------

/// An edge with overloaded == and < to use it in map containers.
struct Edge {
    Vert_idx a, b; ///< vertex index of the edge

    Edge() : a(-1), b(-1) { }


    Edge(Vert_idx a_, Vert_idx b_) : a(a_), b(b_) { }


    bool operator==(const Edge& e) const {
        assert( a != b ); // Corruption detected: un-initialized or self-edge
        return (e.a == a && e.b == b) || (e.b == a && e.a == b);
    }


    bool operator<(const Edge& e) const {
        assert( a != b ); // Corruption detected: un-initialized or self-edge
        return (e.a == a ) ? e.b < b : e.a < a;
    }

    /// Acces through array operator []
    inline const int& operator[](int i) const{
        assert( i < 2);
        return ((int*)this)[i];
    }


    inline int& operator[](int i) {
        assert( i < 2);
        return ((int*)this)[i];
    }
};

// -----------------------------------------------------------------------------

/// @brief triangle face represented with three vertex index
/// you can access the indices with the attributes '.a' '.b' '.c' or the array
/// accessor []
struct Tri_face {

    Tri_face() : a(-1), b(-1), c(-1) { }


    Tri_face(Vert_idx a_, Vert_idx b_, Vert_idx c_) : a(a_), b(b_), c(c_) { }
    /// Get one of the three edges

    Edge edge(int i) const {
        assert( a != b && b != c && c != a ); // Corruption detected: un-initialized or self-tri
        const Edge e[3] = { Edge(a,b), Edge(b,c), Edge(c,a) };
        return e[i];
    }

    /// Acces through array operator []

    inline const int& operator[](int i) const{
        assert( i < 3);
        return ((int*)this)[i];
    }


    inline int& operator[](int i) {
        assert( i < 3);
        return ((int*)this)[i];
    }

    Vert_idx a, b, c;
};

// -----------------------------------------------------------------------------

struct Tri_edges {

    Tri_edges() : a(-1), b(-1), c(-1) { }


    Tri_edges(int a_, int b_, int c_) : a(a_), b(b_), c(c_) { }

    /// Acces through array operator []

    inline const int& operator[](int i) const{
        assert( i < 3);
        return ((int*)this)[i];
    }


    inline int& operator[](int i) {
        assert( i < 3);
        return ((int*)this)[i];
    }

    Edge_idx a, b, c; ///< edge index
};

// -----------------------------------------------------------------------------

struct Quad_face {

    Quad_face() : a(-1), b(-1), c(-1), d(-1) { }


    Quad_face(Vert_idx a_, Vert_idx b_, Vert_idx c_, Vert_idx d_) : a(a_), b(b_), c(c_), d(d_) {}

    /// Acces through array operator []

    inline const int& operator[](int i) const{
        assert( i < 4);
        return ((int*)this)[i];
    }


    inline int& operator[](int i) {
        assert( i < 4);
        return ((int*)this)[i];
    }

    Vert_idx a, b, c, d;
};

// -----------------------------------------------------------------------------

/// @struct PrimIdx
/// The class of a primitive (triangle or quad).
/// Each attribute is the index of a vertex.
struct Prim_idx{
    EMesh::Vert_idx a, b, c, d;
};

// -----------------------------------------------------------------------------

/// @param PrimIdxVertices
/// The offsets for each vertex of each primitive that are used for the
/// temporary storage during the computation of the normals on the GPU
struct Prim_idx_vertices{
    int ia, ib, ic, id;


    Prim_idx_vertices() { }


    Prim_idx_vertices(int ia_, int ib_, int ic_, int id_) :
        ia(ia_), ib(ib_), ic(ic_), id(id_)
    {  }

    void print(){
        printf("%d %d %d %d\n",ia, ib, ic, id);
    }
};

// -----------------------------------------------------------------------------

/** @struct Packed_data
*/
struct Packed_data{
    /// index of the data in the unpacked array of vertices
    EMesh::Vert_idx _idx_data_unpacked;
    /// number of occurences (vertex).
    /// They are consecutively stored in the unpacked array
    /// (starting from idx_data_unpacked)
    int _nb_ocurrence;
};

}// End Namespace EMesh ========================================================

#endif // MESH_TYPES_HPP__
