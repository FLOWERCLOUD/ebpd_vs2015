#ifndef MESH_UTILS_LOADER_HPP__
#define MESH_UTILS_LOADER_HPP__

#include "../parsers/loader_mesh.hpp"
#include "meshes/mesh.hpp"

// =============================================================================
namespace Mesh_utils {
// =============================================================================

/// Build a user-friendly mesh structure (OpenGL half-edges etc.) from
/// the abstract representation of a mesh file
/// @param[out] out_mesh : user friendly mesh to be built from the abstract mesh
/// @param[in] in_mesh : abstract mesh loaded from a file (.obj, .fbx etc.)
void load_mesh(Mesh& out_mesh, const Loader::Abs_mesh& in_mesh);

void save_mesh(const Mesh& in_mesh, Loader::Abs_mesh& out_mesh);

}// END Mesh_utils NAMESPACE ===================================================

#endif // MESH_UTILS_LOADER_HPP__
