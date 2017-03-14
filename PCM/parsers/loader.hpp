#ifndef LOADER_HPP
#define LOADER_HPP

#include <string>
#include <vector>
#include "toolbox/maths/transfo.hpp"
#include "parsers/loader_enum.hpp"
#include "parsers/loader_mesh.hpp"
#include "parsers/loader_skel.hpp"
#include "parsers/loader_anims.hpp"


/**
  @namespace Loader
  @brief this namespace holds the intermediate representation for file loading

  Because Files representation are versatiles we choose to represent in the same
  intermediate data structure the output of a file loader. Hence every loader
  of this project (fbx, obj, off, Collada etc.) will have to parse the files and
  convert to this representation. This way the copy to the real project
  representation will be the same for every file formats.

  The representation is extremly simplified: triangular mesh. Only one instance
  of a mesh/skeleton skin can be loaded. If a file contains more instances they
  will be ignored or concatenated for meshes.

  One can add new loaders by inheriting from Base_loader.
  Don't forget to update the make_loader() to take into acount your new loader

  @see Generic_file Obj_file Off_file Fbx_file
*/

// =============================================================================
namespace Loader {
// =============================================================================

/// @class Base_loader
/// @brief Abstract class for file loading
class Base_loader {
protected:
    Base_loader() {}
    Base_loader(const std::string& file_path);
public:

    virtual ~Base_loader(){ }

    /// The loader type (fbx, obj, off, etc.)
    virtual Loader_t type() const { return NOT_HANDLED; }

    /// Update File paths according to the new 'file_path'
    void update_paths(const std::string& file_path);

    /// parse and load the file into memory
    /// @return true if succeed
    /// @warning call update_paths() when overriding this method
    virtual bool import_file(const std::string& file_path) = 0;

    /// save data to disk
    /// @return true if succeed
    /// @warning call update_paths() when overriding this method
    virtual bool export_file(const std::string& file_path) = 0;

    /// Fill the scene tree with the specified object types.
    /// @param obj_flag : bit flag which specify which object type has to be
    /// loaded use flags defined in EScene_graph::Obj_t
    /// ( for instance (EScene_graph::MESH | EScene_graph::SKELETON) will only
    /// load skeleton and mesh objects into the scene graph). Note that 0 means
    /// every object type must be loaded.
    /// @return wether at least one entity has been loaded into the tree
//    virtual EObj::Flags fill_scene(Scene_tree& tree, EObj::Flags flags = 0) = 0;

    /// @param anims parsed animations
    virtual void get_anims(std::vector<Base_anim_eval*>& anims) const = 0;

    /// @name attributes
    /// the name of the file last loaded is saved internally since the
    /// name may be needed to find a path to a material or texture file.
    std::string _file_path; ///< Path + file name
    std::string _path;      ///< Path without file name
};

//------------------------------------------------------------------------------

/// Factory method
/// Allocate the correct loader given the file type and parse the file
/// @note supported formats : .obj, .off, .skel and .fbx
Base_loader* make_loader(const std::string& file_name);

} // END namespace Loader ======================================================

#endif // LOADER_HPP
