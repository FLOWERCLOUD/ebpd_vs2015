#ifndef GENERIC_LOADER_HPP__
#define GENERIC_LOADER_HPP__

#include "parsers/loader.hpp"

// =============================================================================
namespace Loader {
// =============================================================================

/**
 * @class Generic_file
 * @brief Generic file loader. Use it for automatic deletion of the loader
 *
 * Look Base_loader doc for more infos on the methods of Generic_file
 * @see Base_loader
 */
class Generic_file {
public:
    /// Create and parse file
    Generic_file(const std::string& file_name) {
        _file = make_loader(file_name);
    }

    ~Generic_file(){ delete _file; }

    Loader_t type() const { return _file->type(); }

    bool import_file(const std::string& file_path){
        delete _file;
        _file = make_loader(file_path);
        return _file != 0;
    }

    bool export_file(const std::string& file_path){
        return _file->export_file( file_path );
    }

    /// @return the type of objects that have been loaded
    EObj::Flags fill_scene(Scene_tree& tree, EObj::Flags flags = 0){
        return _file->fill_scene(tree, flags);
    }

    void get_anims(std::vector<Loader::Base_anim_eval*>& anims) const {
        return _file->get_anims(anims);
    }

    /// @return if the file format is supported
    bool supported() const { return _file != 0; }

    std::string file_path() const { return _file->_file_path; }

private:
    Base_loader* _file;
};

} // END LOADER NAMESPACE ======================================================

#endif // GENERIC_LOADER_HPP__
