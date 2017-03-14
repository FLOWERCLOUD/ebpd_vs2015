#include "mesh_utils.hpp"
using namespace Tbx;
// =============================================================================
namespace Mesh_utils {
// =============================================================================

void normals(const Mesh& m, std::vector<Vec3>& normals)
{
    const Mesh_static& mesh = m._mesh_static;

    normals.clear();
    normals.resize(mesh.get_nb_vertices(), 0.0f);

    for(int i = 0; i < mesh.get_nb_tris(); i++)
    {
        EMesh::Tri_face tri = mesh.get_tri( i );
        Vec3 va = mesh.get_vertex( tri.a );
        Vec3 vb = mesh.get_vertex( tri.b );
        Vec3 vc = mesh.get_vertex( tri.c );

        Vec3 e0 = vb - va;
        Vec3 e1 = vc - va;

        Vec3 n = e0.cross( e1 );

        float norm = -n.norm();
        n /= norm;
        normals[tri.a] += n;
        normals[tri.b] += n;
        normals[tri.c] += n;
    }

    for(int i = 0; i < mesh.get_nb_quads(); i++)
    {
        EMesh::Quad_face quad = mesh.get_quad(i);

        Vec3 va = mesh.get_vertex( quad.a );
        Vec3 vb = mesh.get_vertex( quad.b );
        Vec3 vc = mesh.get_vertex( quad.c );
        Vec3 vd = mesh.get_vertex( quad.d );

        Vec3 e0 = vb - va + vc - vd;
        Vec3 e1 = vd - va + vc - vb;

        Vec3 n = e0.cross( e1 );
        float norm = -n.norm();
        n /= norm;
        normals[quad.a] += n;
        normals[quad.b] += n;
        normals[quad.c] += n;
        normals[quad.d] += n;
    }

    for(int i = 0; i < mesh.get_nb_vertices(); i++)
    {
        Vec3& n = normals[i];
        float norm = n.norm();
        if(norm > 0.f)
            n /= -norm;
    }
}

// -----------------------------------------------------------------------------

void tangents( const Mesh& m, std::vector<Vec3>& tangents)
{
    const Mesh_static& mesh_static = m._mesh_static;
    const Mesh_unpacked_attr& mesh_attr = m._mesh_attr;

    tangents.clear();
    tangents.resize(mesh_static.get_nb_vertices(), 0.0f);

    for(int i = 0; i < mesh_static.get_nb_tris(); i++)
    {
        EMesh::Tri_face tri  = mesh_static.get_tri(i);
        EMesh::Tri_face uidx = ((EMesh::Tri_face*)mesh_attr._unpacked_tri.data())[i];

        Vec3 va = mesh_static.get_vertex(tri.a);
        Vec3 vb = mesh_static.get_vertex(tri.b);
        Vec3 vc = mesh_static.get_vertex(tri.c);

        EMesh::Tex_coords ta = ((EMesh::Tex_coords*)(mesh_attr._tex_coords.data()))[uidx.a];
        EMesh::Tex_coords tb = ((EMesh::Tex_coords*)(mesh_attr._tex_coords.data()))[uidx.b];
        EMesh::Tex_coords tc = ((EMesh::Tex_coords*)(mesh_attr._tex_coords.data()))[uidx.c];

        Vec3 v1 = Vec3(vb.x - va.x, vb.y - va.y, vb.z - va.z);
        Vec3 v2 = Vec3(vc.x - va.x, vc.y - va.y, vc.z - va.z);

        EMesh::Tex_coords st1 = EMesh::Tex_coords(tb.u - ta.u, tb.v - ta.v);
        EMesh::Tex_coords st2 = EMesh::Tex_coords(tc.u - ta.u, tc.v - ta.v);

        float coef = 1.f / (st1.u * st2.v - st2.u * st1.v);
        Vec3 tangent = coef * ((v1 * st2.v)  + (v2 * -st1.v));

        tangents[tri.a] += tangent;
        tangents[tri.b] += tangent;
        tangents[tri.c] += tangent;
    }

    for(int i = 0; i < mesh_static.get_nb_quads(); i++)
    {
        // TODO: check this code with a 'bump mapped' quad mesh
        EMesh::Quad_face quad  = mesh_static.get_quad(i);
        EMesh::Quad_face uidx = ((EMesh::Quad_face*)mesh_attr._unpacked_quad.data())[i];

        Vec3 va = mesh_static.get_vertex(quad.a);
        Vec3 vb = mesh_static.get_vertex(quad.b);
        Vec3 vc = mesh_static.get_vertex(quad.c);
        Vec3 vd = mesh_static.get_vertex(quad.d);

        EMesh::Tex_coords ta = ((EMesh::Tex_coords*)(mesh_attr._tex_coords.data()))[uidx.a];
        EMesh::Tex_coords tb = ((EMesh::Tex_coords*)(mesh_attr._tex_coords.data()))[uidx.b];
        EMesh::Tex_coords tc = ((EMesh::Tex_coords*)(mesh_attr._tex_coords.data()))[uidx.c];
        EMesh::Tex_coords td = ((EMesh::Tex_coords*)(mesh_attr._tex_coords.data()))[uidx.d];

        // Tangent of the triangle abc
        Vec3 v1 = Vec3(vb.x - va.x, vb.y - va.y, vb.z - va.z);
        Vec3 v2 = Vec3(vc.x - va.x, vc.y - va.y, vc.z - va.z);

        EMesh::Tex_coords st1 = EMesh::Tex_coords(tb.u - ta.u, tb.v - ta.v);
        EMesh::Tex_coords st2 = EMesh::Tex_coords(tc.u - ta.u, tc.v - ta.v);

        float coef = 1.f / (st1.u * st2.v - st2.u * st1.v);
        Vec3 tangent_0 = coef * ((v1 * st2.v)  + (v2 * -st1.v));

        // Tangent of the triangle acd
        v1 = Vec3(va.x - vc.x, va.y - vc.y, va.z - vc.z);
        v2 = Vec3(vd.x - vc.x, vd.y - vc.y, vd.z - vc.z);

        st1 = EMesh::Tex_coords(ta.u - tc.u, ta.v - tc.v);
        st2 = EMesh::Tex_coords(td.u - tc.u, td.v - tc.v);

        coef = 1.f / (st1.u * st2.v - st2.u * st1.v);
        Vec3 tangent_1 = coef * ((v1 * st2.v)  + (v2 * -st1.v));

        tangents[quad.a] += tangent_0;
        tangents[quad.b] += tangent_0;
        tangents[quad.c] += tangent_0;

        tangents[quad.a] += tangent_1;
        tangents[quad.c] += tangent_1;
        tangents[quad.d] += tangent_1;
    }

    for(int i = 0; i < mesh_static.get_nb_vertices(); i++) {
        tangents[i] = tangents[i].normalized();
    }

}

// -----------------------------------------------------------------------------

void scale(Mesh& mesh, const Vec3& scale)
{
    for(int i = 0; i < mesh.get_nb_vertices(); ++i) {
        mesh._mesh_static.set_vertex(i, mesh.get_vertex(i).mult(scale) );
    }
    mesh._mesh_gl.update_vertex_buffer_object();
}

// -----------------------------------------------------------------------------

void translate(Mesh& mesh, const Vec3& tr)
{
    for(int i = 0; i < mesh.get_nb_vertices(); ++i) {
        mesh._mesh_static.set_vertex(i, mesh.get_vertex(i) + tr );
    }
    mesh._mesh_gl.update_vertex_buffer_object();
}

// -----------------------------------------------------------------------------

float laplacian_cotan_weight(const Mesh& m,
                             EMesh::Vert_idx curr_vert,
                             int edge_curr )
{
    const int nb_neighs = m.get_1st_ring_verts(curr_vert).size();
    const Vec3 c_pos = m.get_vertex( curr_vert );

    const int edge_next = (edge_curr+1) % nb_neighs;

    const int id_curr = m.get_1st_ring_verts(curr_vert)[ edge_curr ];
    const int id_next = m.get_1st_ring_verts(curr_vert)[ edge_next ];
    const int id_prev = m.get_1st_ring_verts(curr_vert)[ (edge_curr-1) >= 0 ? (edge_curr-1) : nb_neighs-1 ];

    const Vec3 v1 = c_pos                 - m.get_vertex(id_prev);
    const Vec3 v2 = m.get_vertex(id_curr) - m.get_vertex(id_prev);
    const Vec3 v3 = c_pos                 - m.get_vertex(id_next);
    const Vec3 v4 = m.get_vertex(id_curr) - m.get_vertex(id_next);

    // wij = (cot(alpha) + cot(beta)),
    // for boundary edge, there is only one such edge
    float cotan1 = 0.0f;
    float cotan2 = 0.0f;
    if( !m.is_side_edge( m.get_1st_ring_edges(curr_vert)[edge_curr] ) )
    {
        // general case: not a boundary
        cotan1 = (v1.dot(v2)) / (v1.cross(v2)).norm();
        cotan2 = (v3.dot(v4)) / (v3.cross(v4)).norm();
    }
    else // boundary edge, only have one such angle
    {
        if( id_next == id_prev )
        {
            // two angles are the same, e.g. corner of a square
            cotan1 = (v1.dot(v2)) / (v1.cross(v2)).norm();
        }
        else
        {
            // find the angle not on the boundary
            if( !m.is_side_edge( m.get_1st_ring_edges(curr_vert)[edge_next] ) )
                cotan2 = (v3.dot(v4)) / (v3.cross(v4)).norm();
            else
                cotan1 = (v1.dot(v2)) / (v1.cross(v2)).norm();
        }
    }

    return (cotan1 + cotan2);
}

// -----------------------------------------------------------------------------

void laplacian_cotan_weights(const Mesh& m,
                             std::vector< std::vector<float> >& per_edge_cotan_weights)
{
    per_edge_cotan_weights.resize( m.get_nb_vertices() );
    for(int i = 0; i < m.get_nb_vertices(); ++i)
    {
        const int nb_neighs = m.get_1st_ring_verts(i).size();
        per_edge_cotan_weights[i].resize( nb_neighs );
        for(int n = 0; n < nb_neighs; ++n)
        {
            float w = laplacian_cotan_weight(m, i, n);
            //std::cout << "vert: " << i << " cotan wij " << w << std::endl;
            per_edge_cotan_weights[i][n] = w;
        }
    }
}

// -----------------------------------------------------------------------------


}// END Mesh_utils NAMESPACE ===================================================
