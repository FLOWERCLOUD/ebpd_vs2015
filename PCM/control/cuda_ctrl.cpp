#include "cuda_ctrl.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "toolbox/portable_includes/port_glew.h"
#include "toolbox/maths/quat_cu.hpp"
#include "../global_datas/toolglobals.hpp"
#include "../global_datas/cuda_globals.hpp"
#include "../rendering/opengl_stuff.hpp"
#include "../rendering/depth_peeling.hpp"
#include "../animation/animesh.hpp"
#include "../rendering/rendering.hpp"
#include "toolbox/cuda_utils/cuda_utils_common.hpp"
#include "../parsers/obj_loader.hpp"
#include "../parsers/fbx_loader.hpp"
#include "../parsers/endianess.hpp"
#include "global_datas/g_scene_tree.hpp"
#include "global_datas/g_textures.hpp"
#include "toolbox/gl_utils/gldirect_draw.hpp"
#include "meshes/mesh_utils_loader.hpp"
#include "../parsers/point_cache_export.hpp"
#include "../animation/graph.hpp"

#include <fstream>
#include <sstream>
using Tbx::Color;


// =============================================================================
namespace Cuda_ctrl {
// =============================================================================

Path_ctrl            _paths;
//Potential_plane_ctrl _potential_plane;
Animated_mesh_ctrl*  _anim_mesh = 0;
Skeleton_ctrl        _skeleton;
Display_ctrl         _display;
Debug_ctrl           _debug;
Graph_ctrl           _graph;
//Operators_ctrl       _operators;
Color_ctrl           _color;
Example_mesh_ctrl  _example_mesh;

// -----------------------------------------------------------------------------

void genertateVertices(std::string _file_paths,std::string name)
{
	_example_mesh.genertateVertices(_file_paths,name);
}

void load_mesh(const std::string file_name)
{
    delete _anim_mesh;
    _anim_mesh = 0;
    delete g_animesh;
    g_animesh = 0;

    delete g_mesh;
    g_mesh = 0;

    std::string ext = file_name;
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if( ext.find(".off") != std::string::npos )
        g_mesh = new Mesh(file_name.c_str());
    else if( ext.find(".obj") != std::string::npos )
    {
        g_mesh = new Mesh();
        // Parse file
        Loader::Obj_file loader( file_name );
        Loader::Abs_mesh abs_mesh;
        // compute abstract representation
        loader.get_mesh( abs_mesh );
        // load for opengl
        Mesh_utils::load_mesh( *g_mesh, abs_mesh );
    }
    else if( ext.find(".fbx") != std::string::npos )
    {
        g_mesh = new Mesh();
        Loader::Fbx_file loader( file_name );
        Loader::Abs_mesh abs_mesh;
        loader.get_mesh( abs_mesh );
        Mesh_utils::load_mesh( *g_mesh, abs_mesh);
    }
    else
        assert(false); // Not the correct type of mesh

    //mesh -> add_noise(5, 0.03f);
    g_mesh->center_and_resize(1.f);
    g_mesh->check_integrity();

    Color cl = _color.get(Color_ctrl::MESH_POINTS);
    g_mesh->set_point_color_bo(cl.r, cl.g, cl.b, cl.a);

    delete g_anim_cache;
    g_anim_cache = new Loader::Point_cache_file(g_mesh->get_nb_vertices(), 100);

    std::cout << "mesh loaded" << std::endl;
}

// -----------------------------------------------------------------------------

void load_mesh( Mesh* mesh )
{
    delete _anim_mesh;
    _anim_mesh = 0;
    delete g_animesh;
    g_animesh = 0;

    delete g_mesh;
    g_mesh = 0;

    g_mesh = mesh;

    //g_mesh->add_noise(5, 0.03f);

    g_mesh->center_and_resize(1.f);
    g_mesh->check_integrity();

    Color cl = _color.get(Color_ctrl::MESH_POINTS);
    g_mesh->set_point_color_bo(cl.r, cl.g, cl.b, cl.a);

    delete g_anim_cache;
    g_anim_cache = new Loader::Point_cache_file(g_mesh->get_nb_vertices(), 100);

    std::cout << "mesh loaded" << std::endl;
}

// -----------------------------------------------------------------------------

bool is_mesh_loaded(){
    return g_mesh != 0 && g_mesh->get_nb_vertices() != 0;
}

// -----------------------------------------------------------------------------

bool is_animesh_loaded(){
    return _anim_mesh != 0;
}

// -----------------------------------------------------------------------------

bool is_skeleton_loaded(){ return g_skel != 0; }

// -----------------------------------------------------------------------------

void erase_graph(){
    delete g_graph;
    g_graph = new Graph(g_mesh->get_offset(), g_mesh->get_scale());
}

// -----------------------------------------------------------------------------

void load_animesh_and_ssd_weights(const char* filename)
{
    delete g_animesh;
    g_animesh = new Animesh(g_mesh, g_skel);
    delete _anim_mesh;
    _anim_mesh = new Animated_mesh_ctrl(g_animesh);

    if(filename){
        // Check extension :
        int len = strlen(filename);
        bool has_commas = (filename[len-4] == '.') &
                (filename[len-3] == 'c') &
                (filename[len-2] == 's') &
                (filename[len-1] == 'v');
        std::cout << "reading weights\n";
        g_animesh->read_weights_from_file(filename, has_commas);
        std::cout << "weights ok" << std::endl;
    }

    g_animesh->update_base_potential();
}

// -----------------------------------------------------------------------------

void load_animesh()
{
    delete g_animesh;
    g_animesh = new Animesh(g_mesh, g_skel);
    delete _anim_mesh;
    _anim_mesh = new Animated_mesh_ctrl(g_animesh);
}

// -----------------------------------------------------------------------------

void reload_shaders()
{
    load_shaders();
}

// -----------------------------------------------------------------------------

void get_mem_usage(double& total, double& free)
{
 //   Cuda_utils::get_device_memory_usage(free, total);
}

// -----------------------------------------------------------------------------

void init_host()
{
    std::cout << "Initialize constants\n";
    //Constants::init();
    std::cout << "Initialize endianness system\n";
    Endianess::init();
    std::cout << "Initialize fbx SDK environment\n";
    Loader::init_fbx_sdk();

    //g_scene_tree = new Scene_tree();

    std::cout << "Done\n";
}

// -----------------------------------------------------------------------------

void set_default_controller_parameters()
{
//#if 0
//    //for bulge-free blending skinning (elbow)
//    Constants::set(Constants::F0, 1.f);
//    Constants::set(Constants::F1, 0.43f);
//    Constants::set(Constants::F2, 1.f);
//    Constants::set(Constants::B0, 0.2f);
//    Constants::set(Constants::B1, 0.7f);
//    Constants::set(Constants::B2, 1.2f);
//    Constants::set(Constants::POW0, 1.f);
//    Constants::set(Constants::POW1, 1.f);
//#else
//    Constants::set(Constants::F0, 0.5f );
//    Constants::set(Constants::F1, 0.5f );
//    Constants::set(Constants::F2, 0.5f );
//    Constants::set(Constants::B0, 0.2f );
//    Constants::set(Constants::B1, 0.7f );
//    Constants::set(Constants::B2, 1.2f );
//    Constants::set(Constants::POW0, 1.f);
//    Constants::set(Constants::POW1, 1.f);
//#endif
    //Blending_env::update_opening(); <- // TODO: to be deleted
    //Blending_env::set_global_ctrl_shape(shape);
}

// -----------------------------------------------------------------------------

GLuint init_tex_operator( int width, int height )
{
    //float* tex = compute_tex_operator(width, height,
    //                                  Cuda_ctrl::_display._operator_type,
    //                                  Cuda_ctrl::_display._operator_mode,
    //                                  Cuda_ctrl::_display._opening_angle,
    //                                  Cuda_ctrl::_display._custom_op_id );

    //glAssert( glBindBuffer(GL_ARRAY_BUFFER, 0) );
    //GLuint tex_id;
    //glAssert( glGenTextures(1, &tex_id) );
    //glAssert( glBindTexture(GL_TEXTURE_2D, tex_id) );
    //glAssert( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP) );
    //glAssert( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP) );
    //glAssert( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST) );
    //glAssert( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST) );

    ////printf("tex ptr %d\n", tex);
    //glAssert( glTexImage2D(GL_TEXTURE_2D,
    //                       0,
    //                       GL_RGBA,
    //                       width,
    //                       height,
    //                       0,
    //                       GL_RGBA,
    //                       GL_FLOAT,
    //                       tex) );

    //glAssert( glBindTexture(GL_TEXTURE_2D, 0) );
    //delete[] tex;
    //return tex_id;
	return -1;
}

// -----------------------------------------------------------------------------

void init_opengl_cuda()
{
    // NOTE this function should not call ANY cuda API functions
    g_mesh  = new Mesh();
    g_graph = new Graph(g_mesh->get_offset(), g_mesh->get_scale());

    g_op_tex = init_tex_operator(100, 100);
	init_host();
}

// -----------------------------------------------------------------------------

//void cuda_start(const std::vector<Blending_env::Op_t>& op)
//{
//    using namespace Cuda_ctrl;
//
//#ifndef NDEBUG
//    std::cout << "WARNING: you're still in debug mode" << std::endl;
//#endif
//
//    init_host();
//    init_cuda( op );
//    set_default_controller_parameters();
//}

// -----------------------------------------------------------------------------

//void cleanup()
//{
//    cudaDeviceSynchronize();
//    CUDA_CHECK_ERRORS();
//
//    // OpenGL ---------------
//    glAssert( glBindTexture(GL_TEXTURE_2D, 0) );
//    glAssert( glBindBuffer(GL_ARRAY_BUFFER, 0) );
//    glAssert( glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0) );
//    glUseProgram( 0 );
//
//    glDeleteBuffers(1, &g_gl_quad);
//
//    glAssert( glDeleteTextures(1, &g_ctrl_frame_tex) );
//    glAssert( glDeleteTextures(1, &g_op_frame_tex) );
//    glAssert( glDeleteTextures(1, &g_op_tex) );
//
//    for (int i = 0; i < NB_TEX; ++i)
//        glDeleteTextures(NB_TEX, g_gl_Tex);
//
//    erase_shaders();
//    Gl::Direct_draw::erase_shader();
//    // End OpenGL ----------
//
//    Textures_env::clean_env();
//    Constants::free();
//
//    delete g_skel; // Skeleton must be deleted before blending env
//    delete g_animesh;
//    delete g_graph;
//    delete g_anim_cache;
//    delete g_mesh;
//    delete g_scene_tree;
//    g_scene_tree  = 0;
//    g_skel        = 0;
//    g_animesh     = 0;
//    g_graph       = 0;
//    g_mesh        = 0;
//
//    Loader::clean_fbx_sdk();
//    Blending_env::clean_env();
//    HRBF_env::clean_env();
//    Precomputed_env::clean_env();
//    Skeleton_env::clean_env();
//
//    CUDA_CHECK_ERRORS();
//}

}// END CUDA_CTRL NAMESPACE  ===================================================
