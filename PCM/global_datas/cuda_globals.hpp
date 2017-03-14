#ifndef CUDA_GLOBALS_HPP__
#define CUDA_GLOBALS_HPP__

#include "toolbox/maths/vec3.hpp"
#include "toolbox/maths/point3.hpp"
#include "toolbox/containers/material_cu.hpp"

struct Graph;
struct Skeleton;
struct Animesh;
namespace Tbx
{
	class Camera;
}

extern Tbx::Material_cu g_ray_material;

/** Various global variables used in cuda source files
 */

/// The graph of the current skeleton
extern Graph* g_graph;

/// The current skeleton
extern Skeleton* g_skel;

/// The current animated mesh
extern Animesh* g_animesh;

/// Is cuda context initialized?
extern bool g_cuda_context_is_init;

/// Return true if one of the given parameters has changed or some other scene
/// parameters since the last call of this function.
/// N.B: the first call returns always true
bool has_changed(const Tbx::Camera& cam,
                 const Tbx::Vec3& v_plane,
                 const Tbx::Point3& p_plane,
                 int width,
                 int height);

#endif // CUDA_GLOBALS_HPP__
