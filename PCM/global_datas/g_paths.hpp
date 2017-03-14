#ifndef G_PATHS_HPP__
#define G_PATHS_HPP__

/** @file g_paths.hpp
    @brief export the global variable related to application paths
    @see globals.cpp
*/

#include <string>

/// Shader directory
extern std::string g_shaders_dir;
/// The directory where screenshots are written
extern std::string g_write_dir;
/// Path to store various caches (mostly blending operators)
extern std::string g_cache_dir;
/// Path to store the app configuration
extern std::string g_config_dir;
/// Path to the general icons folder
extern std::string g_icons_dir;
/// Path to the current theme folder
extern std::string g_icons_theme_dir;

#endif // G_PATHS_HPP__
