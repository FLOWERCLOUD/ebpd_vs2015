#include "parsers/loader.hpp"

//#include "parsers/fbx_loader.hpp"
#include "parsers/obj_loader.hpp"
#include "parsers/off_loader.hpp"
#include "parsers/graph_loader.hpp"
#include "toolbox/std_utils/string.hpp"

static std::string get_file_path(const std::string& path)
{
    std::string res;
    unsigned pos = path.find_last_of('/');

    if( (pos+1) == path.size())
        return path;
    else
    {
        if( pos < path.size()) res = path.substr(0, pos+1);
        else                   res = "";
    }

    return res;
}

// =============================================================================
namespace Loader {
// =============================================================================

	using namespace Tbx;
Base_loader* make_loader(const std::string& file_name)
{
    std::string ext = Std_utils::to_lower( Std_utils::file_ext(file_name) );

    if( ext == ".fbx"){
		return 0;
	}
//        return new Loader::Fbx_file(file_name);
    else if( ext == ".obj")
        return new Loader::Obj_file(file_name);
    else if( ext == ".off")
        return new Loader::Off_file(file_name);
    else if( ext == ".skel")
        return new Loader::Graph_file(file_name);
    else
        return 0;
}

// CLASS Base_loader ===========================================================

Base_loader::Base_loader(const std::string& file_path)
{
    update_paths(file_path);
}

//------------------------------------------------------------------------------

void Base_loader::update_paths(const std::string& file_path)
{
    _file_path = file_path;
    _path      = get_file_path(file_path);
}

// =============================================================================
} // namespace Loader
// =============================================================================
