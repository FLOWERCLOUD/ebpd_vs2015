#include "marching_cubes_cpu.hpp"

/** Code based on Paul Bourke implementation
 * http://paulbourke.net/geometry/polygonise/
 * And Cyril Crassin
 * http://www.icare3d.org/codes-and-projects/codes/opengl_geometry_shader_marching_cubes.html
*/

#include "rendering/marching_cubes/g_tables_mcubes.hpp"
#include "toolbox/portable_includes/port_glew.h"
#include "toolbox/containers/idx3.hpp"

// =============================================================================
namespace Marching_cubes {
// =============================================================================

/// Linearly interpolate the position where an isosurface cuts
/// an edge between two vertices, each with their own scalar value
Vec3 vert_lerp(float iso_lvl, Vec3 p0, Vec3 p1, float val0, float val1)
{
   float  mu = (iso_lvl - val0) / (val1 - val0);
   return p0 + (p1 - p0) * mu;
}

// -----------------------------------------------------------------------------

/**
   Given a grid cell and an isolevel, calculate the triangular
   facets required to represent the isosurface through the cell.
   Return the number of triangular facets, the array "triangles"
   will be loaded up with the vertices at most 5 triangular facets.
    0 will be returned if the grid cell is either totally above
   of totally below the isolevel.
*/
int polygonise(Cell &grid, float iso_lvl, Vec3* triangles)
{
   int cube_idx;
   Vec3 vert_list[12];

   // Determine the index into the edge table which
   // tells us which vertices are inside of the surface
   cube_idx = 0;
   if(grid.val[0] < iso_lvl) cube_idx |= 1;
   if(grid.val[1] < iso_lvl) cube_idx |= 2;
   if(grid.val[2] < iso_lvl) cube_idx |= 4;
   if(grid.val[3] < iso_lvl) cube_idx |= 8;
   if(grid.val[4] < iso_lvl) cube_idx |= 16;
   if(grid.val[5] < iso_lvl) cube_idx |= 32;
   if(grid.val[6] < iso_lvl) cube_idx |= 64;
   if(grid.val[7] < iso_lvl) cube_idx |= 128;

   // Cube is entirely in/out of the surface
   if(g_edge_table[cube_idx] == 0) return 0;

   // Find the vertices where the surface intersects the cube
   if(g_edge_table[cube_idx] & 1   ) vert_list[ 0] = vert_lerp(iso_lvl, grid.pos[0],grid.pos[1], grid.val[0], grid.val[1]);
   if(g_edge_table[cube_idx] & 2   ) vert_list[ 1] = vert_lerp(iso_lvl, grid.pos[1],grid.pos[2], grid.val[1], grid.val[2]);
   if(g_edge_table[cube_idx] & 4   ) vert_list[ 2] = vert_lerp(iso_lvl, grid.pos[2],grid.pos[3], grid.val[2], grid.val[3]);
   if(g_edge_table[cube_idx] & 8   ) vert_list[ 3] = vert_lerp(iso_lvl, grid.pos[3],grid.pos[0], grid.val[3], grid.val[0]);
   if(g_edge_table[cube_idx] & 16  ) vert_list[ 4] = vert_lerp(iso_lvl, grid.pos[4],grid.pos[5], grid.val[4], grid.val[5]);
   if(g_edge_table[cube_idx] & 32  ) vert_list[ 5] = vert_lerp(iso_lvl, grid.pos[5],grid.pos[6], grid.val[5], grid.val[6]);
   if(g_edge_table[cube_idx] & 64  ) vert_list[ 6] = vert_lerp(iso_lvl, grid.pos[6],grid.pos[7], grid.val[6], grid.val[7]);
   if(g_edge_table[cube_idx] & 128 ) vert_list[ 7] = vert_lerp(iso_lvl, grid.pos[7],grid.pos[4], grid.val[7], grid.val[4]);
   if(g_edge_table[cube_idx] & 256 ) vert_list[ 8] = vert_lerp(iso_lvl, grid.pos[0],grid.pos[4], grid.val[0], grid.val[4]);
   if(g_edge_table[cube_idx] & 512 ) vert_list[ 9] = vert_lerp(iso_lvl, grid.pos[1],grid.pos[5], grid.val[1], grid.val[5]);
   if(g_edge_table[cube_idx] & 1024) vert_list[10] = vert_lerp(iso_lvl, grid.pos[2],grid.pos[6], grid.val[2], grid.val[6]);
   if(g_edge_table[cube_idx] & 2048) vert_list[11] = vert_lerp(iso_lvl, grid.pos[3],grid.pos[7], grid.val[3], grid.val[7]);

   // Create the triangle
   int nb_triangles = 0;
   for( int i = 0; g_tri_table[cube_idx][i] != -1; i += 3)
   {
      triangles[nb_triangles    ] = vert_list[ g_tri_table[cube_idx][i    ] ];
      triangles[nb_triangles + 1] = vert_list[ g_tri_table[cube_idx][i + 1] ];
      triangles[nb_triangles + 2] = vert_list[ g_tri_table[cube_idx][i + 2] ];
      nb_triangles += 3;
   }

   return nb_triangles;
}

// -----------------------------------------------------------------------------

/// Software marching cubes polygonization
/// SLow as hell since because we use direct mode rendering in addition of
/// the CPU calculation...
void direct_mode_render_marching_cubes(const Node_implicit_surface* node,
                                       const Vec3 world_start,
                                       Vec3i res,
                                       Vec3 steps,
                                       float iso_lvl)
{
    // Note: if we were using a temporary buffer for the 3D we won't have to recompute a lot of values of the cells ...

    // We could use some threads too for filling and polygonising

    // And VBOs of course

    // Then there is the version that walks onto the surface...

    std::vector<Vec3> triangles( 16 );

    for(Idx3 idx(res, 0); idx.is_in(); ++idx)
    {
        Vec3 pos( idx.to_vec3i() );
        Vec3 org = world_start + pos * steps;

        Cell cell;
        cell.pos[0] = Vec3(org.x          , org.y          , org.z          );
        cell.pos[1] = Vec3(org.x + steps.x, org.y          , org.z          );
        cell.pos[2] = Vec3(org.x + steps.x, org.y          , org.z + steps.z);
        cell.pos[3] = Vec3(org.x          , org.y          , org.z + steps.z);
        cell.pos[4] = Vec3(org.x          , org.y + steps.y, org.z          );
        cell.pos[5] = Vec3(org.x + steps.x, org.y + steps.y, org.z          );
        cell.pos[6] = Vec3(org.x + steps.x, org.y + steps.y, org.z + steps.z);
        cell.pos[7] = Vec3(org.x          , org.y + steps.y, org.z + steps.z);

        for (int i = 0; i < 8; ++i)
            cell.val[i] = node->f( cell.pos[i] );

        int nb_vert = polygonise(cell, iso_lvl, &(triangles[0]));

        Vec3 g;
        glBegin(GL_TRIANGLES);
        for(int n = 0; n < (nb_vert/3); n++)
        {
            g = node->gf( triangles[n * 3 + 0]  );
            glNormal3f(g.x, g.y, g.z);
            glVertex3f( triangles[n * 3 + 0].x, triangles[n * 3 + 0].y, triangles[n * 3 + 0].z );

            g = node->gf( triangles[n * 3 + 1]  );
            glNormal3f(g.x, g.y, g.z);
            glVertex3f( triangles[n * 3 + 1].x, triangles[n * 3 + 1].y, triangles[n * 3 + 1].z );

            g = node->gf( triangles[n * 3 + 2]  );
            glNormal3f(g.x, g.y, g.z);
            glVertex3f( triangles[n * 3 + 2].x, triangles[n * 3 + 2].y, triangles[n * 3 + 2].z );
        }
        glEnd();

    }
}

}// END MARCHING_CUBE ==========================================================
