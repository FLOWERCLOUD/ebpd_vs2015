#include "marching_cubes_cpu.hpp"

#include "toolbox/gl_utils/glassert.h"
#include "toolbox/gl_utils/shader.hpp"
#include "toolbox/gl_utils/glsave.hpp"
#include "toolbox/gl_utils/glbuffer_object.hpp"
#include "toolbox/gl_utils/glbuffer_object.hpp"
#include "global_datas/g_paths.hpp"
#include "rendering/marching_cubes/g_tables_mcubes.hpp"
#include "rendering/marching_cubes/fill_grid_thread.hpp"
#include "toolbox/maths/vec3i.hpp"
#include "toolbox/maths/vec3.hpp"
#include "toolbox/containers/idx3.hpp"
#include "toolbox/maths/bbox3.hpp"

#include  <vector>
#include <QThreadPool>

// =============================================================================
namespace Marching_cubes {
// =============================================================================

// -----------------------------------------------------------------------------
/// @name Host and gl Device Buffers
// -----------------------------------------------------------------------------

/// Shader in charge of the polygonisation of the 3D grid filled on CPU
static Shader_prog* g_shader = 0;

/// The marching cube's 3d grid vertices are stored in a VBO.
/// Points will trigger the shader wich will process each cell of the grid
/// size of vbo == (g_world_resolution - 1)
/// @see init_3D_grid_vbo
static GlBuffer_obj<Vec3>* g_vbo_grid_points = 0;

/// Edge table stored on GPU as a 2d integer texture
/// @see init_mcube_tables_textures g_edge_table
static GLuint g_edge_table_tex; // <- In texture unit 1

/// Triangle table stored on GPU as a 2d integer texture
/// @see init_mcube_tables_textures g_tri_table
static GLuint g_tri_table_tex;  // <- In texture unit 2

/// The scalar field buffer for the marching cube on GPU,
/// stored as an Opengl 3d texture
static GLuint g_3D_grid_tex;    // <- In texture unit 0

/// The scalar field buffer for the marching cube is a 3D grid stored linearly.
static std::vector<float> g_data_field;

// -----------------------------------------------------------------------------
/// @name Scene parameters
// -----------------------------------------------------------------------------

/// Resolution of the 3D grid of the marching cube
static Vec3i g_world_resolution = Vec3i(64); //Default in GUI is 64

/// The axis aligned bounding box defining the area to be polygonised by
/// the marching cube
static Bbox3 g_world_box;

/// Scalar value which represent the iso-surface.
static const float g_iso_level = 0.5f;

// -----------------------------------------------------------------------------

Vec3 get_world_step()
{
    return g_world_box.lengths().div( Vec3(g_world_resolution) );
}

// -----------------------------------------------------------------------------

/// Set the marching cube world axis aligned bounding box
/// Will update the shader uniforms and global variables
void set_world_bbox(const Bbox3& abbox)
{
    g_world_box  = abbox;
    Vec3 len  = g_world_box.lengths();

    GLCurrentProgrammSave save_shader;
    g_shader->use();
    g_shader->set_uniform("dataStep", get_world_step().x, get_world_step().y, get_world_step().z);
    g_shader->set_uniform("world_size", len.x, len.y, len.z);
    g_shader->set_uniform("world_start", g_world_box.pmin.x, g_world_box.pmin.y, g_world_box.pmin.z);
    // Decal for each vertex in a marching cube
    g_shader->set_uniform("vertDecals[0]", 0.0f, 0.0f, 0.0f);
    g_shader->set_uniform("vertDecals[1]", get_world_step().x, 0.0f, 0.0f);
    g_shader->set_uniform("vertDecals[2]", get_world_step().x, get_world_step().y, 0.0f);
    g_shader->set_uniform("vertDecals[3]", 0.0f, get_world_step().y, 0.0f);
    g_shader->set_uniform("vertDecals[4]", 0.0f, 0.0f, get_world_step().z);
    g_shader->set_uniform("vertDecals[5]", get_world_step().x, 0.0f, get_world_step().z);
    g_shader->set_uniform("vertDecals[6]", get_world_step().x, get_world_step().y, get_world_step().z);
    g_shader->set_uniform("vertDecals[7]", 0.0f, get_world_step().y, get_world_step().z);
    g_shader->unuse();
}

// -----------------------------------------------------------------------------

/// Fill the 3D marching cube grid with multiple threads.
/// @param obj : implicit object we fill the grid with
/// @param res : grid resolution for the marching cube
/// @param depth : the octree's depth. World is divided using an octree of depth
/// 'depth' each node is then filled with a thread (according to the resolution
/// 'res')
/// @param offset : a parameter for the recursive calls -> don't touch that!
static void threaded_octree_fill(const Node_implicit_surface* obj,
                                 Vec3i res,
                                 int depth = 1,
                                 Vec3i offset = Vec3i(0))
{

    depth--;
    Vec3i sub_res = res / 2;
    Vec3i curr_offset;
    for(int i = 0; i < 8; ++i)
    {
        curr_offset = sub_res;
        curr_offset.x *= (i      & 0x1);
        curr_offset.y *= (i >> 1 & 0x1);
        curr_offset.z *= (i >> 2 & 0x1);

        curr_offset += offset;

        if(depth > 0)
        {
            threaded_octree_fill(obj, sub_res, depth, curr_offset);
        }
        else
        {
            Fill_grid_thread* thread = new Fill_grid_thread(sub_res,
                                                            curr_offset,
                                                            g_world_resolution,
                                                            g_world_box.pmin,
                                                            get_world_step(),
                                                            g_data_field,
                                                            obj);
            thread->setAutoDelete( true );
            QThreadPool::globalInstance()->start( thread );
        }
    }

}

// -----------------------------------------------------------------------------

void fill_3D_grid_with_scalar_field(const Node_implicit_surface* obj)
{
    set_world_bbox( obj->get_bbox() );


#if 0
    // Generate a distance field to the center of the cube
    //int size = world_resolution.product();
    //std::vector<float> data_field( size );
    for (Idx3 idx(world_resolution, 0); idx.is_in(); ++idx)
    {
        Vec3 pos = g_world_box.pmin + Vec3( idx.to_3d() ).mult(g_world_step) + (g_world_step * 0.5f);
        float d = ((Obj_HRBF*)obj)->f( pos );
        g_data_field[ idx.to_linear() ] = d;
    }
#else

    threaded_octree_fill(obj, g_world_resolution, 2);
    QThreadPool::globalInstance()->waitForDone();
#endif

    // Upload the 3d texture to GPU for future tessalation with marching cubes
    GLActiveTexUnitSave tex_unit_save(GL_TEXTURE0);
    GLEnabledSave tex_3d(GL_TEXTURE_3D, true, true);
    glAssert( glBindTexture(GL_TEXTURE_3D, g_3D_grid_tex) );

    glAssert( glTexImage3D( GL_TEXTURE_3D,
                            0,
                            GL_ALPHA32F_ARB,
                            g_world_resolution.x, g_world_resolution.y, g_world_resolution.z,
                            0,
                            GL_ALPHA,
                            GL_FLOAT,
                            &(g_data_field[0]) )
              );
}

// -----------------------------------------------------------------------------

static void init_shaders()
{
    Shader vs(g_shaders_dir + "/marching_cubes/mcubes.vs", GL_VERTEX_SHADER);
    Shader gs(g_shaders_dir + "/marching_cubes/mcubes.gs", GL_GEOMETRY_SHADER);
    Shader fs(g_shaders_dir + "/marching_cubes/mcubes.fs", GL_FRAGMENT_SHADER);

    g_shader = new Shader_prog();

    g_shader->set_shader(vs);
    g_shader->set_shader(gs);
    g_shader->set_shader(fs);

    int prog_id = g_shader->get_id();

    //Get max number of geometry shader output vertices
    GLint temp;

    // Setup Geometry Shader
    glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT, &temp);
    glAssert( glProgramParameteriEXT(prog_id, GL_GEOMETRY_INPUT_TYPE_EXT , GL_POINTS        ) );
    glAssert( glProgramParameteriEXT(prog_id, GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP) );
    // Marching cube can produce 16 vertices at most
    // This will accelerate a bit
    glAssert( glProgramParameteriEXT(prog_id, GL_GEOMETRY_VERTICES_OUT_EXT, 16) );

    g_shader->link();
}

// -----------------------------------------------------------------------------

static void init_mcube_tables_textures()
{
    ////////////////////////
    // Edge Table texture //

    // Create texture and upload table
    glAssert( glGenTextures(1, &g_edge_table_tex) );
    {
        GLActiveTexUnitSave tex_unit_save(GL_TEXTURE1);
        GLEnabledSave tex_2d(GL_TEXTURE_2D, true, true);

        glAssert( glBindTexture(GL_TEXTURE_2D, g_edge_table_tex) );
        // Integer textures must use nearest filtering mode of course
        glAssert( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)   );
        glAssert( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)   );
        glAssert( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE) );
        glAssert( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE) );
        glAssert( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE) );

        // We create an integer texture with new GL_EXT_texture_integer formats
        // It's a 2D texture so the table is stored as [1][256] hence the additionnal '&'
        glAssert( glTexImage2D( GL_TEXTURE_2D,
                                0,
                                GL_ALPHA16I_EXT,
                                256, 1,
                                0,
                                GL_ALPHA_INTEGER_EXT,
                                GL_INT,
                                &g_edge_table) );
    }

    ////////////////////////////
    // Triangle Table texture //

    // Create texture and upload table
    glAssert( glGenTextures(1, &g_tri_table_tex) );
    {
        GLActiveTexUnitSave tex_unit_save(GL_TEXTURE2);
        GLEnabledSave tex_2d(GL_TEXTURE_2D, true, true);

        glAssert( glBindTexture(GL_TEXTURE_2D, g_tri_table_tex) );
        glAssert( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)   );
        glAssert( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)   );
        glAssert( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE) );
        glAssert( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE) );
        glAssert( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE) );

        glAssert( glTexImage2D( GL_TEXTURE_2D,
                                0,
                                GL_ALPHA16I_EXT,
                                16, 256,
                                0,
                                GL_ALPHA_INTEGER_EXT,
                                GL_INT,
                                &g_tri_table) );
    }
}

// -----------------------------------------------------------------------------

static void init_discreet_scalar_field_tex(const Vec3i& res)
{
    // Allocate the 3D texture which stores discreet values of the scalar field
    // to polygonise
    glAssert( glGenTextures(1, &g_3D_grid_tex) );

    GLActiveTexUnitSave tex_unit_save(GL_TEXTURE0);
    GLEnabledSave tex_3d(GL_TEXTURE_3D, true, true);

    glAssert( glBindTexture(GL_TEXTURE_3D, g_3D_grid_tex) );
    glAssert( glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)    );
    glAssert( glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)    );
    glAssert( glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE) );
    glAssert( glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE) );
    glAssert( glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE) );

    // Allocate and configure on GPU
    glAssert( glTexImage3D( GL_TEXTURE_3D,
                            0,
                            GL_ALPHA32F_ARB,
                            res.x, res.y, res.z,
                            0,
                            GL_ALPHA,
                            GL_FLOAT,
                            0 )
              );

    // Allocate CPU buffer
    g_data_field.resize( res.product() );

    //std::cout << "GPU allocation marching cube grid: ";
    //std::cout << (float)res.product() * 4.f / 1024.f / 1024.f << " Mbytes" << std::endl;
}

// -----------------------------------------------------------------------------

// Build a VBO filled with points at the center of the cube cells.
// it will triger the geometry shader at each cube cells to generate the triangles
static void init_3D_grid_vbo(const Vec3i& res)
{
    g_vbo_grid_points = new GlBuffer_obj<Vec3>(GL_ARRAY_BUFFER);

    // Build the VBO on host: a GL_POINT per cube cell in the grid.
    Vec3i sub_size = res;
    int size = sub_size.product();
    std::vector<Vec3> grid_data( size );
    for(Idx3 idx(sub_size, 0); idx.is_in(); ++idx)
    {
        //Vec3 pos = world_box.pmin + Vec3( idx.to_3d() ).mult( cube_step );
        grid_data[ idx.to_linear() ] = Vec3( idx.to_3d() );
    }

    g_vbo_grid_points->set_data( grid_data, GL_STATIC_DRAW);

    std::cout << "CPU allocation marching cube grid: ";
    std::cout << (float)size * 3.f * 4.f / 1024.f / 1024.f << " Mbytes" << std::endl;
}

// -----------------------------------------------------------------------------

static void init_shader_settings()
{
    g_shader->use();
    g_shader->set_tex_unit("dataFieldTex", 0);
    g_shader->set_tex_unit("edgeTableTex", 1);
    g_shader->set_tex_unit("triTableTex" , 2);

    g_shader->set_uniform("isolevel", g_iso_level);

    const Vec3 world_size(10.f);
    set_world_bbox( Bbox3(Point3(-world_size*0.5f), Point3(world_size*0.5f)) );

    g_shader->unuse();
}

// -----------------------------------------------------------------------------

void init()
{
    init_shaders();
    init_mcube_tables_textures();
    init_discreet_scalar_field_tex( g_world_resolution );
    init_3D_grid_vbo( g_world_resolution );
    init_shader_settings();
}

// -----------------------------------------------------------------------------

void render_scalar_field()
{

#if 1
    //GLPolygonModeSave poly_mode(GL_LINE);

    g_shader->use();
/*
    g_shader->set_tex_unit("dataFieldTex", 0);
    g_shader->set_tex_unit("edgeTableTex", 1);
    g_shader->set_tex_unit("triTableTex" , 2);
    GLActiveTexUnitSave tex_unit_save(GL_TEXTURE0);
    GLEnabledSave tex_3d(GL_TEXTURE_3D, true, true);
    glAssert( glBindTexture(GL_TEXTURE_3D, g_3D_grid_tex) );

    glAssert( glActiveTexture(GL_TEXTURE2) );
    glAssert( glEnable(GL_TEXTURE_2D) );
    glAssert( glBindTexture(GL_TEXTURE_2D, g_tri_table_tex) );

    glAssert( glActiveTexture(GL_TEXTURE1) );
    glAssert( glEnable(GL_TEXTURE_2D) );
    glAssert( glBindTexture(GL_TEXTURE_2D, g_edge_table_tex) );
*/

    g_vbo_grid_points->bind();
    glAssert( glEnableClientState(GL_VERTEX_ARRAY) );
    glAssert( glVertexPointer(3, GL_FLOAT, 0,  0) );

    glAssert( glDrawArrays(GL_POINTS, 0, g_world_resolution.x*g_world_resolution.y*g_world_resolution.z) );
    glAssert( glDisableClientState(GL_VERTEX_ARRAY) );

    g_vbo_grid_points->unbind();
    g_shader->unuse();
#endif


#if 0
    ///////DEBUG
    // Draw grid vertices
    GLPointSizeSave sa(4.f);
    glBegin( GL_POINTS );
    for(Idx3 idx(world_resolution, 0); idx.is_in(); ++idx)
    {
        Vec3 pos = g_world_box.pmin + Vec3( idx.to_3d() ).mult( g_world_step );
        glVertex3f(pos.x, pos.y, pos.z);
    }
    glEnd();
    ///////DEBUG
#endif

}

// -----------------------------------------------------------------------------

void render_scalar_field_cpu(const Node_implicit_surface* obj, const Vec3i& user_res)
{
    set_world_bbox( obj->get_bbox() );

    Vec3i res = (user_res.x < 0 || user_res.y < 0 || user_res.z < 0) ?
                g_world_resolution :
                user_res;

    direct_mode_render_marching_cubes(obj,
                                      g_world_box.pmin,
                                      res,
                                      g_world_box.lengths().div( Vec3(res) ),
                                      g_iso_level);
}

// -----------------------------------------------------------------------------

void set_resolution( const Vec3i& res )
{
    g_world_resolution = res;

    // 3d grid for scalar field reallocation:
    GLActiveTexUnitSave tex_unit_save(GL_TEXTURE0);
    GLEnabledSave tex_3d(GL_TEXTURE_3D, true, true);
    glAssert( glBindTexture(GL_TEXTURE_3D, g_3D_grid_tex) );
    glAssert( glTexImage3D( GL_TEXTURE_3D,
                            0,
                            GL_ALPHA32F_ARB,
                            res.x, res.y, res.z,
                            0,
                            GL_ALPHA,
                            GL_FLOAT,
                            0 )
              );
    // Allocate CPU buffer
    g_data_field.resize( res.product() );

    delete g_vbo_grid_points;
    g_vbo_grid_points = 0;
    init_3D_grid_vbo( res );
    init_shader_settings();
}


// -----------------------------------------------------------------------------

void clean()
{
    delete g_shader;
    delete g_vbo_grid_points;
    g_shader = 0;
    g_vbo_grid_points = 0;
    glAssert( glDeleteTextures(1, &g_edge_table_tex) );
    glAssert( glDeleteTextures(1, &g_tri_table_tex ) );
    glAssert( glDeleteTextures(1, &g_3D_grid_tex   ) );
    g_edge_table_tex = -1;
    g_tri_table_tex  = -1;
    g_3D_grid_tex    = -1;
}

}// END MARCHING_CUBE ==========================================================
