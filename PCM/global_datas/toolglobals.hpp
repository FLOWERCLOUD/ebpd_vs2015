#ifndef GLOBAL_HPP__
#define GLOBAL_HPP__

#include <string>
#include "toolbox/portable_includes/port_glew.h"
#include "toolbox/gl_utils/glshoot.hpp"
#include "macros.hpp"
class Mesh;
class GlShoot;
namespace  Loader
{
	class Point_cache_file;
}
/** @file globals.hpp
    @brief Various global variables
*/

// TODO: encapsulate in namespace when necessary, and provice init() clean() methods
// also add hpp to export specifis variables or group


//#include "blending_env_type.hpp"
//extern Blending_env::Op_id CUSTOM_TEST;

// -----------------------------------------------------------------------------

extern Loader::Point_cache_file* g_anim_cache;

// -----------------------------------------------------------------------------

extern bool g_shooting_state;
extern bool g_save_anim;
extern Tbx::GlShoot* g_oglss;

// TODO: to be deleted
/// The current mesh
extern Mesh* g_mesh;

// -----------------------------------------------------------------------------

#endif // GLOBAL_HPP__
