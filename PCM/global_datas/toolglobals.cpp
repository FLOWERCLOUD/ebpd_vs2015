 #include "toolglobals.hpp"
#include "toolbox/gl_utils/vbo_primitives.hpp"
//#include "global_datas/g_scene_tree.hpp"
#include "global_datas/g_shaders.hpp"
#include "global_datas/g_textures.hpp"
// -----------------------------------------------------------------------------
//Blending_env::Op_id CUSTOM_TEST = -1;

// -----------------------------------------------------------------------------

Loader::Point_cache_file* g_anim_cache = 0;

// -----------------------------------------------------------------------------

//Scene_tree* g_scene_tree = 0;

// -----------------------------------------------------------------------------

Tbx::VBO_primitives g_primitive_printer;
Tbx::Prim_id g_sphere_lr_vbo;
Tbx::Prim_id g_sphere_vbo;
Tbx::Prim_id g_circle_vbo;
Tbx::Prim_id g_arc_circle_vbo;
Tbx::Prim_id g_circle_lr_vbo;
Tbx::Prim_id g_arc_circle_lr_vbo;
Tbx::Prim_id g_grid_vbo;
Tbx::Prim_id g_cylinder_vbo;
Tbx::Prim_id g_cylinder_cage_vbo;
Tbx::Prim_id g_cube_vbo;

GLuint g_gl_quad;

// -----------------------------------------------------------------------------

GLuint g_gl_Tex[NB_TEX];
GLuint g_ctrl_frame_tex;
GLuint g_op_frame_tex;
GLuint g_op_tex;

// -----------------------------------------------------------------------------

Tbx::Shader_prog* g_dummy_quad_shader = 0;
Tbx::Shader_prog* g_points_shader = 0;
Tbx::Shader_prog* g_normal_map_shader = 0;
Tbx::Shader_prog* g_ssao_shader = 0;

namespace Tex_units{
	const int KD   = 3;
	const int KS   = 4;
	const int BUMP = 5;
}

Tbx::Shader_prog* g_phong_list[NB_PHONG_SHADERS];
//std::string write_dir = "/export/home/magritte/vaillant/ppm_img";
std::string g_shaders_dir = "./rendering/shaders";
std::string g_write_dir  = "./resource";
std::string g_cache_dir  = "./resource/app_cache";
std::string g_config_dir = "./resource/app_config";
std::string g_icons_dir  = "./resource/icons";
std::string g_icons_theme_dir = "./resource/icons/blue_theme";

bool g_shooting_state = false;
bool g_save_anim = false;
Tbx::GlShoot* g_oglss;

// -----------------------------------------------------------------------------

Mesh* g_mesh = 0;


