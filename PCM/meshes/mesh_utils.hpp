#ifndef MESH_TOOLS_HPP__
#define MESH_TOOLS_HPP__

#include "toolbox/maths/vec3.hpp"
#include "toolbox/maths/bbox3.hpp"
#include "mesh.hpp"

/**
 * @namespace Mesh_utils
 * @brief utilities to work on a #Mesh class
 *
 */
// =============================================================================
namespace Mesh_utils {
// =============================================================================

/// @return The 'edge_idx' vector direction.
static inline
Tbx::Vec3 edge_dir(const Mesh& m, int edge_idx)
{
    const EMesh::Edge e = m.get_edge( edge_idx );
    return (m.get_vertex(e.b) - m.get_vertex(e.a));
}

// -----------------------------------------------------------------------------

/// @return The 'edge_idx' length.
static inline
float edge_len(const Mesh& m, int edge_idx)
{
    return edge_dir(m, edge_idx).norm();
}

// -----------------------------------------------------------------------------

/// @return The 'tri_idx' triangle area.
static inline
float tri_area(const Mesh& m, int tri_idx)
{
	using namespace Tbx;
    const EMesh::Tri_face t = m.get_tri( tri_idx );
    const Vec3 va = m.get_vertex( t.a );
    const Vec3 vb = m.get_vertex( t.b );
    const Vec3 vc = m.get_vertex( t.c );

    const Vec3 edge0 = vb - va;
    const Vec3 edge1 = vc - va;

    return (edge0.cross(edge1)).norm() / 2.f;
}

// -----------------------------------------------------------------------------

/// @return normalized normal of triangle 'tri_idx'
static inline
Tbx::Vec3 tri_normal(const Mesh& m, int tri_idx)
{
	using namespace Tbx;
    const EMesh::Tri_face t = m.get_tri( tri_idx );
    const Vec3 va = m.get_vertex( t.a );
    const Vec3 vb = m.get_vertex( t.b );
    const Vec3 vc = m.get_vertex( t.c );

    const Vec3 edge0 = vb - va;
    const Vec3 edge1 = vc - va;

    return edge0.cross(edge1).normalized();
}

// -----------------------------------------------------------------------------

/// @return the dihedral signed angle [-PI PI] between the 2 triangles shared
/// at the edge 'edge_idx'. If the number of shared triangles is not equal to 2
/// we return a value strictly greater than M_PI*2.f to signal it.
/// The sign of the angle is determine by the orientation of the edge
/// 'edge_idx' and the order of triangles in m.get_edge_shared_tris(edge_idx)
static inline
float edge_dihedral_angle(const Mesh& m, int edge_idx)
{
	using namespace Tbx;
    const std::vector<int>& tris = m.get_edge_shared_tris( edge_idx );
    if( tris.size() != 2) return (float)M_PI*2.f + 1.f;

    Vec3 n0 = tri_normal(m, tris[0]);
    Vec3 n1 = tri_normal(m, tris[1]);

    Vec3 ref = edge_dir(m, edge_idx);
    return ref.signed_angle( n0, n1 );
}

// -----------------------------------------------------------------------------

static inline
Tbx::Bbox3 bounding_box(const Mesh& m)
{
    Tbx::Bbox3 bbox;
    for(int i = 0; i < m.get_nb_vertices(); ++i) {
        bbox.add_point( m.get_vertex(i) );
    }
    return bbox;
}

// -----------------------------------------------------------------------------

// TODO compute normals with a more accurate scheme such as laplacian weighting
/// Compute normals of 'mesh' by averaging the faces around vertices
/// @param mesh : mesh used to compute the normals
/// @param normals : list of normals per vertex (same order as in 'mesh')
void normals(const Mesh& mesh, std::vector<Tbx::Vec3>& normals);

// -----------------------------------------------------------------------------

/// Compute tangents of 'mesh' using texture coordinates
void tangents( const Mesh& mesh, std::vector<Tbx::Vec3>& tangents);

// -----------------------------------------------------------------------------

/// Apply scaling of 'scale' to 'mesh'
void scale(Mesh& mesh, const Tbx::Vec3& scale);

// -----------------------------------------------------------------------------

/// Apply translation 'tr' to 'mesh'
void translate(Mesh& mesh, const Tbx::Vec3& tr);

// -----------------------------------------------------------------------------

/// Compute a cotan weight at a single vertex "curr_vert" for its edge
/// "edge_curr"
float laplacian_cotan_weight(const Mesh& m,
                             EMesh::Vert_idx curr_vert,
                             int edge_curr);

// -----------------------------------------------------------------------------

/// Compute cotangent weights for every vertices.
/// Cotans weights are used to compute the laplacian of a function over the
/// mesh's surface
void laplacian_cotan_weights(const Mesh& m,
                             std::vector< std::vector<float> >& per_edge_cotan_weights);

// -----------------------------------------------------------------------------


}// END Mesh_utils NAMESPACE ===================================================

#endif // MESH_TOOLS_HPP__
