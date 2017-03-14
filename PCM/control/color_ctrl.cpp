#include "color_ctrl.hpp"

#include "../meshes/mesh.hpp"
extern Mesh* g_mesh;

#include "path_ctrl.hpp"
namespace Cuda_ctrl{
    extern Path_ctrl _paths;
}

#include "toolbox/class_saver.hpp"

// -----------------------------------------------------------------------------

void Color_ctrl::load_class_from_file()
{
    std::string path = Cuda_ctrl::_paths._configuration;
    path += "config_colors.dat";
    Tbx::load_class(this, path);
}


// -----------------------------------------------------------------------------

void Color_ctrl::save_class_to_file()
{
    std::string path = Cuda_ctrl::_paths._configuration;
    path += "config_colors.dat";
    Tbx::save_class(this, path);
}

// -----------------------------------------------------------------------------

Tbx::Color Color_ctrl::get(int enum_field){
    return _list[enum_field];
}

// -----------------------------------------------------------------------------

void  Color_ctrl::set(int enum_field, const Tbx::Color& cl){
    _list[enum_field] = cl;

    // TODO: use the scene graph implement a method in Mesh object to change color
    if(enum_field == MESH_POINTS && g_mesh != 0)
        g_mesh->set_point_color_bo(cl.r, cl.g, cl.b, cl.a);

    save_class_to_file();
}

// -----------------------------------------------------------------------------
