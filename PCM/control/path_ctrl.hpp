#ifndef PATH_CTRL_HPP__
#define PATH_CTRL_HPP__

#include <string>

/** @class Path_ctrl
    @brief This class holds various paths used into the application
*/
class Path_ctrl {
public:

    Path_ctrl() :
        _meshes("./resource/meshes/"),
        _textures("resource/textures/"),
        _screenshots("./resource/"),
        _configuration("./resource/app_config/")
    {  }

    std::string _meshes;
    std::string _textures;
    std::string _screenshots;
    std::string _configuration;
};

#endif // PATH_CTRL_HPP__
