#ifndef FBX_LOADER_HPP__
#define FBX_LOADER_HPP__

#include "loader.hpp"

#include <map>

/**
  @file fbx_loader.hpp
  @brief Holds data structure and utilities to store and parse en FBX file

  This module use the FBX SDK to parse and store meshes/skeleton/skin/scenes.
  The SDK requires a global initialization at the programm startup. This
  is done with Loader::init_fbx_sdk().
  Releasing memory of the SDK is done with Loader::clean_fbx_sdk().

  A specialization of Loader::Base_loader is provided to import/export FBX files
  into an intermediate data representation.
*/

// Forward def to avoid the depencie of FBX SDK in this header
namespace fbxsdk_2015_1 {
class FbxNode;
class FbxScene;
}

// =============================================================================
namespace Loader {
// =============================================================================

/// InitiLoaderfbx SDK memory manager. This is mandatory to use the SDK
/// and must be done once at the programm startup
/// @see clean()
void init_fbx_sdk();

/// This erase the fbx SDK memory manager and must be called once when the
/// application is closed
void clean_fbx_sdk();

/**
    @class Fbx_file
    @brief main interface for an alias fbx file.
*/
class Fbx_file : public Base_loader {
public:

    //friend class Fbx_anim_eval;

    Fbx_file (const std::string& file_name) :
        Base_loader(file_name),
        _fbx_scene(0)
    {
        import_file(file_name);
    }

    ~Fbx_file(){ free_mem(); }


    bool import_file(const std::string& file_path);
    bool export_file(const std::string& file_path){
        Base_loader::update_paths( file_path );
        // TODO
        return false;
    }

    //EObj::Flags fill_scene(Scene_tree& tree, EObj::Flags flag = 0);

    //EObj::Obj_t fill_with_mesh(Scene_tree& tree, Obj_mesh*& obj_mesh);

    //EObj::Obj_t fill_with_skel(Scene_tree& tree, Obj_skeleton*& obj_skel, Abs_skeleton& abs_skel);

    /// @return parsed animations or NULL.
    void get_anims(std::vector<Base_anim_eval*>& anims) const {
        get_animations(anims);
    }

    /// Transform internal FBX representation into our skeleton representation
    void get_skeleton(Abs_skeleton& skel) const;

    /// Get fbx animation evaluator. Animation are concatenated in 'anims'
    void get_animations(std::vector<Base_anim_eval*>& anims) const;

    /// transform internal fbx representation into generic representation
    void get_mesh(Abs_mesh& mesh);

    /// transform generic representation into internal fbx representation
    void set_mesh(const Abs_mesh& mesh);

    void free_mem();

private:
    /// compute attributes '_offset_verts' and '_size_mesh'
    void compute_size_mesh();

    /// The FBX file data once parsed with load_file()
    fbxsdk_2015_1::FbxScene* _fbx_scene;

    /// Stores for each FBX mesh node the offset introduced in vertices index
    /// because we concatenate meshes
    std::map<const fbxsdk_2015_1::FbxNode*, int> _offset_verts;

    /// Nb vertices of all the FBX meshes concatenated
    int _size_mesh;
};

}// END namespace LOADER =======================================================


#endif // FBX_LOADER_HPP__

