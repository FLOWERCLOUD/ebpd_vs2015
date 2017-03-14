#ifndef MESH_HALF_EDGE_HPP__
#define MESH_HALF_EDGE_HPP__

#include <vector>
#include "mesh_static.hpp"
#include "mesh_types.hpp"

class Mesh;

/**
 * @class Mesh_half_edge
 * @brief Build an half edge mesh from a standard static mesh
 *
 * This enhance a simple triangle list representation of a mesh by adding
 * more advanced connectivity informations. You will be able to look up vertices
 * in the first or second ring neighbor, look up edges and more.
 *
 *
 *
 */
struct Mesh_half_edge {

    Mesh_half_edge() :
        _is_closed(true),
        _is_manifold(true),
        _nb_total_neigh(0)
    {
    }

    Mesh_half_edge(const Mesh_static& mesh) :
        _is_closed(true),
        _is_manifold(true),
        _nb_total_neigh(0)
    {
        update( mesh );
    }

    ~Mesh_half_edge(){ clear_data(); }

    //TODO: A more effecient copy constructor by factoring look ups
    //Mesh_half_edge( const Mesh_half_edge& mesh_he ) { }

    /// Clear all attributes
    void clear_data();

    /// Updates every connectivity datas according to "mesh"
    void update(const Mesh_static& mesh);

    // -------------------------------------------------------------------------
    /// @name Accessors
    // -------------------------------------------------------------------------

    /// @return false if the mesh have boundaries or true if it's watertight
    bool is_closed() const { return _is_closed; }

    /// @return true if no deffects are detected (like edge shared by more than
    /// 2 triangles and such)
    bool is_manifold() const { return _is_manifold; }

    //  ------------------------------------------------------------------------
    /// @name Faces accessors
    //  ------------------------------------------------------------------------

    /// Get the offsets for primitive no. i
    EMesh::Prim_idx_vertices get_piv(int i) const {
        const std::vector<int>& piv = _piv;
        return EMesh::Prim_idx_vertices(piv[4*i], piv[4*i +1], piv[4*i + 2],  piv[4*i + 3]);
    }

    /// Get a triangle described by a list of three edge indices.
    /// You can then retreive the edge with #get_edge()
    EMesh::Tri_edges get_tri_edges(EMesh::Tri_idx tri_index) const {
        return _tri_edges[tri_index];
    }

    // -------------------------------------------------------------------------
    /// @name Vertices accessors
    // -------------------------------------------------------------------------

    /// @return false if the ith vertex belongs to at least one primitive
    /// i.e triangle or quad
    bool is_disconnect(EMesh::Vert_idx i) const { return !_is_connected[i]; }

    /// @return true if the vertex is connected to a face
    bool is_vert_connected(EMesh::Vert_idx i) const { return _is_connected[i]; }

    /// Is the ith vertex on the mesh boundary
    bool is_vert_on_side(EMesh::Vert_idx i) const { return _is_side[i]; }

    /// List of vertex indices which are connected to not manifold triangles
    const std::vector<EMesh::Vert_idx>& get_not_manifold_list() const { return _not_manifold_verts; }

    /// List of vertices on a side of the mesh
    const std::vector<EMesh::Vert_idx>& get_on_side_list() const { return _on_side_verts; }

    int get_max_faces_per_vertex() const { return _max_faces_per_vertex; }

    //  ------------------------------------------------------------------------
    /// @name 1st ring neighborhood accessors
    //  ------------------------------------------------------------------------

    // TODO: iterators to explore the 1st ring neighbors

    /// clear and fill 'first_ring' with the ordered first neighborhood ring of
    /// the ith vertex
    void get_1st_ring( std::vector< std::vector<EMesh::Vert_idx> >& first_ring ) const{
        first_ring = _1st_ring_verts;
    }

    const std::vector<EMesh::Vert_idx>& get_1st_ring_verts(EMesh::Vert_idx i) const {
        return _1st_ring_verts[i];
    }

    /// list_size == #get_size_1st_ring_list()
    /// Contigus list of 1st ring neighborhood of every vertices
    EMesh::Vert_idx get_1st_ring(int i) const {
        assert(i < _nb_total_neigh);
        return _1st_ring_list[i];
    }

    int get_size_1st_ring_list() const { return _nb_total_neigh; }

    /// Size == 2 * get_nb_vertices()
    int get_1st_ring_offset(EMesh::Vert_idx i) const {
        assert(i < get_nb_vertices()*2);
        return _1st_ring_list_offsets[i];
    }

    /// Valence of the ith vertex
    int get_valence(EMesh::Vert_idx i) const { return _1st_ring_list_offsets[i*2 + 1]; }

    /// Number of 1st ring neighborhood at the ith vertex"same as valence"
    int get_nb_neighbors(EMesh::Vert_idx i) const { return _1st_ring_list_offsets[i*2+1]; }

    /// Get edge indices at the first ring neighborhood of the ith vertex.
    /// @note same order as with first ring of vertices (#get_1st_ring())
    const std::vector<int>& get_1st_ring_edges(EMesh::Vert_idx i) const {
        return _edge_list_per_vert[i];
    }

    /// Get triangle indices at the first ring neighborhood of the ith vertex.
    /// @warning !! this list is unordered unlike the other 1st ring list !!
    // FIXME: add a post-process that re-order this list c.f compute_face_index()
    const std::vector<EMesh::Tri_idx>& get_1st_ring_tris(EMesh::Vert_idx i) const {
        return _tri_list_per_vert[i];
    }

    int get_nb_vertices() const { return (int)_1st_ring_verts.size(); }

    //  ------------------------------------------------------------------------
    /// @name 2nd ring neighborhood accessors
    //  ------------------------------------------------------------------------

    const std::vector<EMesh::Vert_idx>& get_2nd_ring_verts(EMesh::Vert_idx i) {
        assert( i >= 0 && i < get_nb_vertices() );
        return _2nd_ring_verts[i];
    }

    const std::vector< std::vector<EMesh::Vert_idx> >& get_2nd_ring_verts() {
        return _2nd_ring_verts;
    }

    //  ------------------------------------------------------------------------
    /// @name Edges accessors
    //  ------------------------------------------------------------------------

    EMesh::Edge get_edge(EMesh::Edge_idx i) const { return _edge_list[i]; }

    int get_nb_edges() const { return (int)_edge_list.size(); }

    /// List of triangles shared by the edge 'i'. It can be one for boundaries
    /// two for closed objects or another number if the mesh is not 2-manifold
    const std::vector<EMesh::Tri_idx>& get_edge_shared_tris(EMesh::Edge_idx i) const { return _tri_list_per_edge[i]; }

    bool is_side_edge(EMesh::Edge_idx i) const { return _is_side_edge[i]; }

private:
    // -------------------------------------------------------------------------
    /// @name data updates
    // -------------------------------------------------------------------------

    /// Allocate and compute the offsets for the computation of the normals
    /// on the GPU
    void compute_piv(const Mesh_static& mesh);

    /// For each vertex compute the list of faces it belongs to and stores it
    /// in the attributes '_tri_list_per_vert' and '_quad_list_per_vert'.
    void compute_per_vertex_faces(const Mesh_static& mesh);

    /// Compute the first neighborhood list of every vertices
    /// updates '_1st_ring_list' and '_1st_ring_list_offsets'
    void compute_1st_ring(const Mesh_static& mesh,
                          const std::vector<std::vector<EMesh::Tri_idx> >& tri_list_per_vert,
                          const std::vector<std::vector<EMesh::Quad_idx> >& quad_list_per_vert,
                          const std::vector<bool>& is_connected);

    /// Compute the second neighborhood list of every vertices
    void compute_2nd_ring(const std::vector< std::vector<EMesh::Vert_idx> >& fst_ring_verts);

    /// compute the list "_edges_list", "_edge_list_per_vert" and
    /// "_tri_list_per_edge"
    /// @warning must be called after compute_1st_ring() as we use
    /// attribute _1st_ring_list for the computation
    void compute_edges(const std::vector< std::vector<EMesh::Vert_idx> >& fst_ring_verts);

    void compute_tri_list_per_edges(const Mesh_static& mesh,
                                    const std::vector<std::vector<EMesh::Tri_idx> >& tri_list_per_vert,
                                    const std::vector<EMesh::Edge>& edge_list,
                                    const std::vector<bool>& is_vert_on_side);

    void compute_tri_edges(const Mesh_static& mesh,
                           const std::vector<EMesh::Edge>& edge_list,
                           const std::vector<std::vector<EMesh::Tri_idx> >& tri_list_per_edge);


    // -------------------------------------------------------------------------

    bool _is_closed;         ///< is the mesh closed

    /// When false it is sure the mesh is not 2-manifold. But true does not
    /// ensure mesh is 2-manifold (for instance self intersection are not detected
    /// and when is_closed == false, is_manifold remains true). Mainly
    /// topological defects are detected with is_manifold == false.
    bool _is_manifold;

    int _max_faces_per_vertex;
    // TODO: comment this attribute
    /// ?
    std::vector<int> _piv;

    std::vector<bool> _is_connected; ///< Does the ith vertex belongs to a tri or a quad
    std::vector<bool> _is_side;      ///< Does the ith vertex belongs to a mesh boundary

    /// list of triangles index connected to a vertices.
    /// tri_list_per_vert[index_vert][nb_connected_triangles] = index triangle
    /// in attribute '_tri'
    /// @warning triangle list per vert are unordered
    std::vector<std::vector<EMesh::Tri_idx> > _tri_list_per_vert;

    /// list of quads index connected to a vertices.
    /// quad_list_per_vert[index_vert][nb_connected_quads] = index quad in
    /// attribute '_quad'
    /// @warning quad list per vert are unordered
    std::vector<std::vector<EMesh::Quad_idx> > _quad_list_per_vert;

    /// List of packed vertex index which presents topological defects.
    std::vector<EMesh::Vert_idx> _not_manifold_verts;

    /// List of vertex on the side of the mesh
    std::vector<EMesh::Vert_idx> _on_side_verts;

    /// List of triangles which are described using three edge indices
    /// corresponding to the "_edge_list"
    /// _tri_edges[tri_idx] = { 3 edge index of "_edge_list" }
    std::vector<EMesh::Tri_edges> _tri_edges;

    int _nb_total_neigh; ///< nb elt in '_1st_ring_list'

    // TODO: first ring should be a two dimensionnal vector if flatten is needed the user must take care of it
    /// list of neighbours of each vertex
    /// N.B : quads are triangulated before creating this list of neighborhoods
    /// @see 1st_ring_list_offsets
    std::vector<EMesh::Vert_idx> _1st_ring_list;

    /// 1st index and number of neighbours of each vertex in the 1st_ring_list
    /// array size is twice the number of vertices (2*nb_vert).
    /// Usage:
    /// @code
    ///     int dep      = _1st_ring_list_offsets[i*2    ];
    ///     int nb_neigh = _1st_ring_list_offsets[i*2 + 1];
    ///     for(int n = dep; n < (dep+nb_neigh); n++)
    ///         int neigh = _1st_ring_list[n];
    /// @endcode
    /// @see 1st_ring_list
    std::vector<int> _1st_ring_list_offsets;

    /// Ordered list of the 1st ring vertex neighbors
    /// _1st_ring_verts[id_vertex][ith_vert_neighbor] == vert_neighbor_id
    std::vector< std::vector<EMesh::Vert_idx> > _1st_ring_verts;

    /// List of 2nd ring neighborhood vertices per vertex.
    /// @warning list is not ordered
    std::vector< std::vector<EMesh::Vert_idx> > _2nd_ring_verts;

    ///////////////
    // EDGE DATA //
    ///////////////

    /// List of edges
    std::vector<EMesh::Edge> _edge_list;

    /// List of edges per vertices
    /// _edge_list_per_vert[vert_id] == list of edge id in '_edge_list'
    std::vector<std::vector<EMesh::Edge_idx> > _edge_list_per_vert;

    /// List of triangles per edges
    /// _tri_list_per_edge[edge_id] == list of tri id in '_tri'
    std::vector<std::vector<EMesh::Tri_idx> > _tri_list_per_edge;

    /// List of edges on the boundary of the mesh
    std::vector<EMesh::Edge_idx> _on_side_edges;
    /// Does the ith edge belongs to a mesh boundary
    std::vector<bool> _is_side_edge;

};

#endif // MESH_HALF_EDGE_HPP__
