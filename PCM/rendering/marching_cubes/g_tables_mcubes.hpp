#ifndef TABLES_MCUBE_H
#define TABLES_MCUBE_H

// =============================================================================
namespace Marching_cubes {
// =============================================================================

/// Edge list of an intersected cube cell with 8bits for each vertex of the cell
/// A bit set to 1 means the vertex is inside the iso-surface.
extern int g_edge_table[256];

/// Given the edge table index gives the triangel configuration.
/// The list of 16 elements is the adjacency between the vertices.
extern int g_tri_table[256][16];

}// END MARCHING_CUBES =========================================================

#endif // TABLES_MCUBE_H
