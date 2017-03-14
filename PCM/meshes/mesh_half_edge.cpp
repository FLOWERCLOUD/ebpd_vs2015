#include "mesh_half_edge.hpp"

#include <deque>
#include <set>

#include "toolbox/std_utils/set.hpp"
#include "toolbox/std_utils/vector.hpp"
//#include "toolbox/timer.hpp"
#include "mesh.hpp"

// -----------------------------------------------------------------------------

void Mesh_half_edge::clear_data()
{
    _piv.clear();
    _is_connected.clear();
    _is_side.clear();
    _edge_list.clear();
    _edge_list_per_vert.clear();
    _tri_list_per_vert.clear();
    _quad_list_per_vert.clear();
    _not_manifold_verts.clear();
    _on_side_verts.clear();
    _2nd_ring_verts.clear();
    _tri_list_per_edge.clear();
    _on_side_edges.clear();
    _is_side_edge.clear();
    _1st_ring_list_offsets.clear();
    _1st_ring_list.clear();
    _tri_edges.clear();
}

// -----------------------------------------------------------------------------

void Mesh_half_edge::update(const Mesh_static& mesh)
{
//    Timer t; t.start();

    compute_piv( mesh );
    compute_per_vertex_faces( mesh );
    compute_1st_ring( mesh, _tri_list_per_vert, _quad_list_per_vert, _is_connected);
    compute_2nd_ring( _1st_ring_verts );
    compute_edges( _1st_ring_verts );
    compute_tri_list_per_edges(mesh, _tri_list_per_vert, _edge_list, _is_side);
    compute_tri_edges(mesh, _edge_list, _tri_list_per_edge);

//    std::cout << "Half Edge mesh computed in: " << t.stop() << " sec" << std::endl;
}

// -----------------------------------------------------------------------------

void Mesh_half_edge::compute_piv(const Mesh_static& mesh)
{
    _piv.resize( 4 * (mesh.get_nb_tris() + mesh.get_nb_quads()) );

    std::vector<int> pic( mesh.get_nb_vertices(), 0 ); // Nb faces per vertices

    int imax = 0; // Max number of faces for every vertices

    for(int i = 0; i < mesh.get_nb_tris(); i++)
    {
        EMesh::Tri_face t = mesh.get_tri( i );

        _piv[4*i] = pic[t.a]++;
        if(imax < pic[t.a]) imax = pic[t.a];
        _piv[4*i+1] = pic[t.b]++;
        if(imax < pic[t.b]) imax = pic[t.b];
        _piv[4*i+2] = pic[t.c]++;
        if(imax < pic[t.c]) imax = pic[t.c];
        _piv[4*i+3] = 0;
    }

    for(int i = 0; i < mesh.get_nb_quads(); i++)
    {
        int j = i + mesh.get_nb_quads();
        EMesh::Quad_face q = mesh.get_quad( i );

        _piv[4*j] = pic[q.a]++;
        if(imax < pic[q.a]) imax = pic[q.a];
        _piv[4*j+1] = pic[q.b]++;
        if(imax < pic[q.b]) imax = pic[q.b];
        _piv[4*j+2] = pic[q.c]++;
        if(imax < pic[q.c]) imax = pic[q.c];
        _piv[4*j+3] = pic[q.d]++;
        if(imax < pic[q.d]) imax = pic[q.d];
    }
    _max_faces_per_vertex = imax;
}

// -----------------------------------------------------------------------------

void Mesh_half_edge::compute_per_vertex_faces(const Mesh_static& mesh)
{
    _tri_list_per_vert.clear();
    _quad_list_per_vert.clear();
    _tri_list_per_vert. resize( mesh.get_nb_vertices() );
    _quad_list_per_vert.resize( mesh.get_nb_vertices() );
    _is_connected.resize( mesh.get_nb_vertices() );
    Tbx::Std_utils::fill( _is_connected, false );

    for(int i = 0; i < mesh.get_nb_tris(); i++)
    {
        EMesh::Tri_face tri = mesh.get_tri( i );
        for(int j = 0; j < 3; j++){
            int v = tri[ j ];
            assert(v >= 0);
            _tri_list_per_vert[v].push_back(i);
            _is_connected[v] = true;
        }
    }

    for(int i = 0; i < mesh.get_nb_quads(); i++){
        EMesh::Quad_face quad = mesh.get_quad( i );
        for(int j = 0; j < 4; j++){
            int v = quad[ j ];
            _quad_list_per_vert[v].push_back(i);
            _is_connected[v] = true;
        }
    }

    // FIXME: look up the list _tri_list_per_vert and re-order in order to ensure
    // triangles touching each other are next in the list.
}

// -----------------------------------------------------------------------------

/** given a triangle 'tri' and one of its vertex index 'current_vert'
    return the pair corresponding to the vertex index opposite to 'current_vert'
    @code
      "current_vert"
            *
           / \
          /   \
         *_____*
    second     first  <--- find the opposite pair
    @endcode
 */
static
std::pair<int, int> pair_from_tri(const EMesh::Tri_face& tri,
                                  int current_vert)
{
    int ids[2] = {-1, -1};
    for(int i = 0; i < 3; i++)
    {
        if(tri[i] == current_vert)
        {
            ids[0] = tri[ (i+1) % 3 ];
            ids[1] = tri[ (i+2) % 3 ];
            break;
        }
    }
    return std::pair<int, int>(ids[0], ids[1]);
}

// -----------------------------------------------------------------------------

/// Same as pair_from_tri() but with quads. 'n' is between [0 1] and tells
/// if the first or the seccond pair of vertex is to be choosen.
static
std::pair<int, int> pair_from_quad(const EMesh::Quad_face& quad,
                                   int current_vert,
                                   int n)
{
    assert(n == 1 || n == 0);
    int ids[3] = {-1, -1, -1};
    for(int i = 0; i < 4; i++)
    {
        if(quad[i] == current_vert)
        {
            ids[0] = quad[ (i+1) % 4 ];
            ids[1] = quad[ (i+2) % 4 ];
            ids[2] = quad[ (i+3) % 4 ];
            break;
        }
    }

    return std::pair<int, int>(ids[0+n], ids[1+n]);
}

// -----------------------------------------------------------------------------

static
bool add_to_ring(std::deque<int>& ring, std::pair<int, int> p)
{
    if(ring[ring.size()-1] == p.first)
    {
        ring.push_back(p.second);
        return true;
    }
    else if(ring[ring.size()-1] == p.second)
    {

        ring.push_back(p.first);
        return true;
    }
    else if(ring[0] == p.second)
    {
        ring.push_front(p.first);
        return true;
    }
    else if(ring[0] == p.first)
    {
        ring.push_front(p.second);
        return true;
    }
    return false;
}

// -----------------------------------------------------------------------------

/// Add an element to the ring only if it does not already exists
/// @return true if already exists
static
bool add_to_ring(std::deque<int>& ring, int neigh)
{
    std::deque<int>::iterator it;
    for(it = ring.begin(); it != ring.end(); ++it)
        if(*it == neigh) return true;

    ring.push_back( neigh );
    return false;
}

// -----------------------------------------------------------------------------

void Mesh_half_edge::compute_1st_ring(const Mesh_static& mesh,
                                      const std::vector<std::vector<EMesh::Tri_idx> >& tri_list_per_vert,
                                      const std::vector<std::vector<EMesh::Quad_idx> >& quad_list_per_vert,
                                      const std::vector<bool>& is_connected)
{
//    Timer t;
//    t.start();

    _is_closed = true;
    _is_manifold = true;
    _not_manifold_verts.clear();
    _on_side_verts.clear();

    _is_side.resize( mesh.get_nb_vertices() );

    _nb_total_neigh = 0;
    _1st_ring_verts.clear();
    _1st_ring_verts.resize( mesh.get_nb_vertices() );
    std::vector<std::pair<int, int> > list_pairs;
    list_pairs.reserve(16);
    for(int i = 0; i < mesh.get_nb_vertices(); i++)
    {
        // We suppose the faces are quads to reserve memory
        if( tri_list_per_vert[i].size() > 0)
            _1st_ring_verts[i].reserve(tri_list_per_vert[i].size());

        if( !is_connected[i] ) continue;

        list_pairs.clear();
        // fill pairs with the first ring of neighborhood of quads and triangles
        for(unsigned j = 0; j < tri_list_per_vert[i].size(); j++)
            list_pairs.push_back(pair_from_tri(mesh.get_tri( tri_list_per_vert[i][j] ), i));

        for(unsigned j = 0; j < quad_list_per_vert[i].size(); j++)
        {
            list_pairs.push_back(pair_from_quad(mesh.get_quad( quad_list_per_vert[i][j] ), i, 0));
            list_pairs.push_back(pair_from_quad(mesh.get_quad( quad_list_per_vert[i][j] ), i, 1));
        }

        // Try to build the ordered list of the first ring of neighborhood of i
        std::deque<int> ring;
        ring.push_back(list_pairs[0].first );
        ring.push_back(list_pairs[0].second);
        std::vector<std::pair<int, int> >::iterator it = list_pairs.begin();
        list_pairs.erase(it);
        unsigned int  pairs_left = list_pairs.size();
        bool manifold   = true;
        while( (pairs_left = list_pairs.size()) != 0)
        {
            for(it = list_pairs.begin(); it < list_pairs.end(); ++it)
            {
                if(add_to_ring(ring, *it))
                {
                    list_pairs.erase(it);
                    break;
                }
            }

            if(pairs_left == list_pairs.size())
            {
                // Not manifold we push neighborhoods of vert 'i'
                // in a random order
                add_to_ring(ring, list_pairs[0].first );
                add_to_ring(ring, list_pairs[0].second);
                list_pairs.erase(list_pairs.begin());
                manifold = false;
            }
        }

        if(!manifold)
        {
            std::cerr << "WARNING : The mesh is clearly not 2-manifold !\n";
            std::cerr << "Check vertex index : " << i << std::endl;
            _is_manifold = false;
            _not_manifold_verts.push_back(i);
        }

        if(ring[0] != ring[ring.size()-1]){
            _is_side[i] = true;
            _on_side_verts.push_back(i);
            _is_closed = false;
        }
        else
        {
            _is_side[i] = false;
            ring.pop_back();
        }

        for(unsigned int j = 0; j < ring.size(); j++)
            _1st_ring_verts[i].push_back( ring[j] );

        _nb_total_neigh += ring.size();
    }// END FOR( EACH VERTEX )

    // Copy results on a more GPU friendly layout for future use
    _1st_ring_list.resize( _nb_total_neigh );
    _1st_ring_list_offsets.resize( 2 * mesh.get_nb_vertices() );

    int k = 0;
    for(int i = 0; i < mesh.get_nb_vertices(); i++)
    {
        int size = _1st_ring_verts[i].size();
        _1st_ring_list_offsets[2 * i    ] = k;
        _1st_ring_list_offsets[2 * i + 1] = size;
        for(int j = 0; j <  size; j++)
            _1st_ring_list[k++] = _1st_ring_verts[i][j];
    }

    if(!_is_closed) std::cout << "Mesh is not a closed mesh\n";
	std::cout << "Mesh 1st ring computed in:"<< std::endl;
//    std::cout << "Mesh 1st ring computed in: " << t.stop() << " sec" << std::endl;
}

// -----------------------------------------------------------------------------

void Mesh_half_edge::compute_2nd_ring(const std::vector< std::vector<EMesh::Vert_idx> >& fst_ring_verts)
{
    _2nd_ring_verts.resize( fst_ring_verts.size() );

    std::set<EMesh::Vert_idx> exclude;
    std::vector<EMesh::Vert_idx> list_1st_ring;
    list_1st_ring.reserve( 20 );

    // We don't know if _2nd_ring_neigh is gonna be ordered it depends on the
    // fst_ring_neigh which has to be ordered and always turning the same way
    for(EMesh::Vert_idx i = 0; i < (int)fst_ring_verts.size(); i++)
    {
        exclude.clear();
        list_1st_ring.clear();
        exclude.insert( i );

        // Explore first ring neighbor
        for(unsigned j = 0; j < fst_ring_verts[i].size(); j++)
        {
            int neigh = fst_ring_verts[i][j];
            exclude.insert( neigh );
            list_1st_ring.push_back( neigh );
        }

        // Explore second ring neighbor
        for(unsigned n = 0; n < list_1st_ring.size(); n++)
        {
            int vert_idx = list_1st_ring[n];
            for(unsigned j = 0; j < fst_ring_verts[vert_idx].size(); j++)
            {
                int neigh = fst_ring_verts[vert_idx][j];
                if( !Tbx::Std_utils::exists( exclude, neigh) )
                {
                    _2nd_ring_verts[i].push_back( neigh );
                    exclude.insert( neigh );
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------

/// Seek edge 'e' in 'edge_idx_list' if found return the index of the edge in
/// 'edge_list'
static int seek_edge(const EMesh::Edge& e,
                     const std::vector<int>& edge_idx_list,
                     const std::vector<EMesh::Edge>& edge_list)
{
    int edge_idx = -1;
    std::vector<int>::const_iterator it = edge_idx_list.begin();
    for(;it != edge_idx_list.end(); ++it)
    {
        if( edge_list[ (*it) ] == e )
        {
            edge_idx = (*it);
            break;
        }
    }

    // assert if edge not found:
    // it means there is serious corruption in the data
    assert( edge_idx < (int)edge_list.size() );
    assert( edge_idx > -1                    );

    return edge_idx;
}

// -----------------------------------------------------------------------------

/// @return the number of edges according to the half edge representation in
/// fst_ring_verts[vert_id][ith_neighbor] == id_neighbor
static int compute_nb_edges(const std::vector<std::vector<EMesh::Vert_idx> >& fst_ring_verts)
{
    // Stores wether a vertex has been treated, it's a mean to check wether we
    // have already see an edge or not
    std::vector<bool> done( fst_ring_verts.size(), false );

    // Look up every vertices and compute exact number of edges
    int nb_edges = 0;
    for(unsigned i = 0; i < fst_ring_verts.size(); i++)
    {
        for(unsigned n = 0; n < fst_ring_verts[i].size(); n++)
        {
            int neigh = fst_ring_verts[i][n];
            if( !done[neigh] /*never seen the edge*/)
                nb_edges++;
        }
        done[i] = true;
    }

    return nb_edges;
}

// -----------------------------------------------------------------------------

void Mesh_half_edge::compute_edges(const std::vector<std::vector<EMesh::Vert_idx> >& fst_ring_verts)
{
    int nb_edges = compute_nb_edges( fst_ring_verts );

    int nb_verts = (int)fst_ring_verts.size();
    // Reset vertex flags to false
    std::vector<bool> done(nb_verts, false);
    _edge_list.resize( nb_edges );
    _edge_list_per_vert.resize( nb_verts );

    // Building the list of mesh's edges as well as
    // the list of 1st ring neighbors of edges.
    int idx_edge = 0;
    for(int i = 0; i < nb_verts; i++)
    {
        int nb_neigh = (int)fst_ring_verts[i].size();
        _edge_list_per_vert[i].reserve( nb_neigh );
        // look up 1st ring and add edges to the respective lists
        // if not already done
        for(int n = 0; n < nb_neigh; n++)
        {
            int neigh = fst_ring_verts[i][n];
            const EMesh::Edge e(i, neigh);

            if( !done[neigh] )
            {
                // Edge seen for the first time
                _edge_list[idx_edge] = e;
                _edge_list_per_vert[i].push_back( idx_edge );
                idx_edge++;
            }
            else
            {
                // Edge already created we have to
                // seek for its index
                int found_edge_idx = seek_edge(e, _edge_list_per_vert[neigh], _edge_list);
                _edge_list_per_vert[i].push_back( found_edge_idx );
            }
        }
        // Must be as much neighbors than edges per vertex
        assert( (int)_edge_list_per_vert[i].size() == nb_neigh );
        done[i] = true;
    }
}

// -----------------------------------------------------------------------------

void Mesh_half_edge::compute_tri_list_per_edges(const Mesh_static& mesh,
                                                const std::vector<std::vector<EMesh::Tri_idx> >& tri_list_per_vert,
                                                const std::vector<EMesh::Edge>& edge_list,
                                                const std::vector<bool>& is_vert_on_side)
{
    int nb_edges = (int)edge_list.size();

    // Compute triangle list per edges
    // and tag/store edges on the mesh's boundaries
    _tri_list_per_edge.resize( nb_edges );
    _is_side_edge.resize(nb_edges, false);
    _on_side_edges.reserve( nb_edges );
    for(int i = 0; i < nb_edges; i++)
    {
        const EMesh::Edge edge = edge_list[i];
        // Choose first vertex
        const int vert_id = edge.a;
        // Look up tris of this vertex and seek for the edge 'e'
        const std::vector<int>& tri_list = tri_list_per_vert[vert_id];
        _tri_list_per_edge[i].reserve( is_vert_on_side[vert_id] ? 1 : 2 );
        for(unsigned t = 0; t < tri_list.size(); t++)
        {
            const int tri_idx = tri_list[t];
            EMesh::Tri_face tri = mesh.get_tri( tri_idx );

            for(int e = 0; e < 3; ++e) {
                if( tri.edge(e) == edge ){
                    _tri_list_per_edge[i].push_back( tri_idx );
                    break;
                }
            }
        }

        // Store edges on sides and tag them
        if( is_vert_on_side[edge.a] && is_vert_on_side[edge.b] )
        {
            _on_side_edges.push_back( i );
            _is_side_edge[i] = true;
        }
    }
}

// -----------------------------------------------------------------------------

void Mesh_half_edge::compute_tri_edges(const Mesh_static& mesh,
                                       const std::vector<EMesh::Edge>& edge_list,
                                       const std::vector<std::vector<EMesh::Tri_idx> >& tri_list_per_edge)
{
    int nb_edges = (int)edge_list.size();

    // Find the edge indices in _edge_list for every triangles.
    // We look up every edges and update triangles in the neighborhood.
    _tri_edges.clear();
    _tri_edges.resize( mesh.get_nb_tris() );
    for(int edge_idx = 0; edge_idx < nb_edges; edge_idx++)
    {
        EMesh::Edge edge = edge_list[edge_idx];
        const std::vector<int>& tri_list = tri_list_per_edge[edge_idx];
        for(unsigned t = 0; t < tri_list.size(); ++t)
        {
            const int tri_idx = tri_list[t];
            EMesh::Tri_face tri = mesh.get_tri(tri_idx);

            // Seek the triangle edge matching ours
            int u = 0;
            for(; u < 3; ++u) {
                if( EMesh::Edge( tri[u], tri[(u+1) % 3] ) == edge )
                    break;
            }
            // Update our list
            _tri_edges[tri_idx][u] = edge_idx;
        }
    }

    #ifndef NDEBUG
    for(int i = 0; i < mesh.get_nb_tris(); ++i) {
        for( int j = 0; j < 3; ++j) {
            // Check if every edges has been properly updated
            assert( _tri_edges[i][j] >= 0 );
        }
    }
    #endif
}

// -----------------------------------------------------------------------------
