#ifndef LOADER_ENUM_HPP__
#define LOADER_ENUM_HPP__


// =============================================================================
namespace Loader {
// =============================================================================

/// @enum Loader_t
/// @brief The file formats recognized by our loader module
enum Loader_t {
    FBX,          ///< .fbx
    OBJ,          ///< .obj
    OFF,          ///< .off
    SKEL,         ///< .skel
    NOT_HANDLED
};


}// END LOADER =================================================================

#endif // LOADER_ENUM_HPP__
