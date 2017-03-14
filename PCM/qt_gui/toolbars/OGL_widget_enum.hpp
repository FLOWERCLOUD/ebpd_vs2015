#ifndef OGL_WIDGET_ENUM_HPP
#define OGL_WIDGET_ENUM_HPP

// =============================================================================
namespace EOGL_widget {
// =============================================================================

/// @deprecated as io will be generic for all types of objects
/// @name IO_t
/// @brief Input/output mode defines what the behavior and responses of
/// mouse/keyboard events
enum IO_t {
    RBF,       ///< Enables RBF sample editing
    DISABLE,   ///< Ignores all io events
    GRAPH,     ///< Editing the skeleton (add/remove/move joints)
    SKELETON,  ///< Skeleton manipulation (select/move)
    MESH_EDIT,  ///< Mesh editing (point selection)
    BLOB		///< Blob editing
};

/// @name Pivot_t
/// @brief Defines the center of rotation mode
enum Pivot_t {
    JOINT = 0,     ///< rotate around a joint
    BONE = 1,      ///< rotate around the midlle of a bone
    SELECTION = 2, ///< Rotate around the cog of selected elments
    USER = 3,      ///< Rotate around a point defined by the user
    FREE = 4       ///< no center of rotation
};

/// @name Select_t
/// @brief Defines the selection mode
enum Select_t {
    MOUSE,      ///< Use mouse cursor
    CIRCLE,     ///< Use circle area
    BOX,        ///< Use box area
    FREE_FORM   ///< Use free form
};

} // END OGL_widget ============================================================

#endif // OGL_WIDGET_ENUM_HPP
