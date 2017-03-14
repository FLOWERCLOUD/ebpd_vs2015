#ifndef FBX_UTILS_HPP__
#define FBX_UTILS_HPP__

#include <string>
#include <fbxsdk/fbxsdk.h>
#include <fbxsdk/fileio/fbxiosettings.h>
#include <fbxsdk/core/fbxmanager.h>
#include <fbxsdk/scene/fbxscene.h>
#include <fbxsdk/fileio/fbximporter.h>
//#include <fbxsdk/fbxfilesdk_nsuse.h>

#include "toolbox/maths/transfo.hpp"
#include "loader_mesh.hpp"

/**
 * @name Fbx_utils
 * @brief Utilities for the FBX SDK and interacting with our own datas.
 */
// =============================================================================
namespace Fbx_utils {
// =============================================================================

/// @param filename : file path and name to load
/// @param fbx_scene : the allocated scene by FbxManager
/// @param g_manager : the global SDK manager initialized
/// @return If the scene has been succesfully loaded
bool load_scene(const std::string& filename,
                FbxScene* fbx_scene,
                FbxManager* g_manager);

// -----------------------------------------------------------------------------
/// @name Handling the FBX tree data.
// -----------------------------------------------------------------------------

/// Find the first node of the attribute type attrib in the tree
FbxNode* find_root(FbxNode* root, FbxNodeAttribute::EType attrib);

// -----------------------------------------------------------------------------
/// @name Printing various FBX informations
// -----------------------------------------------------------------------------

/// Print the node hierachy
void print_hierarchy(FbxScene* pScene);

/// Print the list of anim stacks names in the given scene
void print_anim_stacks(FbxScene* pScene);

// -----------------------------------------------------------------------------
/// @name Transformations
// -----------------------------------------------------------------------------

/// Function to get a node's global default position.
/// As a prerequisite, parent node's default local position must be already set.
void set_global_frame(FbxNode* pNode, FbxMatrix pGlobalPosition);

/// Recursive function to get a node's global default position.
/// As a prerequisite, parent node's default local position must be already set.
FbxMatrix get_global_frame(const FbxNode* pNode);

/// Get the geometry deformation local to a node.
/// It is never inherited by the children.
FbxAMatrix geometry_transfo(FbxNode* pNode);

// -----------------------------------------------------------------------------
/// @name Conversion FBX types to our types
// -----------------------------------------------------------------------------

/// copy 'd3' in array 't'
void copy(float* t[3], const FbxDouble3& d3);

/// copy FBX phong material to our Material type
void copy( Loader::Material& mat, const FbxSurfacePhong* phong );

/// copy FBX lambert material to our Material type
void copy( Loader::Material& m, const FbxSurfaceLambert* lambert );

/// FbxX matrix to our transformation type
Tbx::Transfo to_transfo(const FbxMatrix& mat);

/// Fbx matrix to our transformation type
Tbx::Transfo to_transfo(const FbxMatrix& mat);

/// FBxVector4 to our loader vertex type
Loader::Vertex to_lvertex(const FbxVector4& vec);

/// FBxVector4 to our loader normal type
Loader::Normal to_lnormal(const FbxVector4& vec);

/// FBxVector2 to our loader texture coordinate type
Loader::Tex_coord to_ltexcoord(const FbxVector2& vec);

/// @return the string corresponding to the mapping mode
std::string to_string(FbxGeometryElement::EMappingMode type);

/// @return the string corresponding to the attribute type
std::string to_string(FbxNodeAttribute::EType type);

}// END NAMESPACE FBX_UTILS ====================================================

#endif // FBX_UTILS_HPP__
