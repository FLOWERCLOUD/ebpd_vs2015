#ifndef ANIMATED_MESH_ENUM_HPP__
#define ANIMATED_MESH_ENUM_HPP__

/// @brief Enumerants used by Animesh
// =============================================================================
namespace EAnimesh {
// =============================================================================

/// Define some flags corresponding to different cases a vertex is in when
/// fitted into the implicit surfaces with the gradient march.
/// This helps to see what is happening.
enum Vert_state {
    POTENTIAL_PIT,         ///< the vertex is going away from its base potential
    GRADIENT_DIVERGENCE,   ///< collision detected
    NB_ITER_MAX,           ///< stopped before reaching base potential because the surface is too far
    NOT_DISPLACED,         ///< The vertex is already on the right iso-surface
    FITTED,                ///< The vertex has been succesfully fitted on its base potential
    OUT_VERT,              ///< Vertex starts inside another surface than its own
    NORM_GRAD_NULL,        ///< Gradient norm is null can't set a proper marching direction
    PLANE_CULLING,
    CROSS_PROD_CULLING,
    // Always keep this at the end ------------------------
    NB_CASES
};

// -----------------------------------------------------------------------------

/// Different Possibilities to compute the vertex color of the animesh
enum Color_type {
    CLUSTER,                  ///< Color different by bone cluster
    NEAREST_JOINT,            ///< Color different by joint cluster
    BASE_POTENTIAL,           ///< Color based on the potential
    GRAD_POTENTIAL,           ///< The current gradient potential of the vertex
    SSD_INTERPOLATION,        ///< Color based on the ssd interpolation factor
    SMOOTHING_WEIGHTS,        ///< Color based on the smoothing weights
    ANIM_SMOOTH_LAPLACIAN,    ///< Color based on the animated smoothing weights
    ANIM_SMOOTH_CONSERVATIVE, ///< Color based on the animated smoothing weights
    NORMAL,                   ///< Color based on mesh current animated normals
    USER_DEFINED,             ///< Uniform color defined by user
    SSD_WEIGHTS,              ///< SSD weights are painted on the mesh
    VERTICES_STATE,           ///< Color based on the stop case encounter while fitting @see EAnimesh::Vert_state
    MVC_SUM,                  ///< Color mean value coordinates sum at each vertex
    FREE_VERTICES,            ///< Distinguish vertices deformed by non linear energy from others (rigidly transformed)
    EDGE_STRESS,              ///< Color vertices with average edges streching/compression from rest pose
    AREA_STRESS,              ///< Color vertices with average tris areas streching/compression from rest pose
    GAUSS_CURV,               ///< Color vertices with gaussian curvature
    DISPLACEMENT              ///< binnary Color for vertex moved previously by the deformation algorithm
};

// -----------------------------------------------------------------------------

enum Smooth_type {
    LAPLACIAN,     ///< Based on the centroid of the first ring neighborhood
    CONSERVATIVE,  ///< Try to minimize changes with the rest position
    TANGENTIAL,    ///< Laplacian corrected with the mesh normals
    HUMPHREY       ///< Laplacian corrected with original points position
};

// -----------------------------------------------------------------------------

/// Geometric deformations mode of the vertices
enum Blending_type {
    DUAL_QUAT_BLENDING = 0, ///< Vertices are deformed with dual quaternions
    MATRIX_BLENDING,        ///< Vertices are deformed with the standard SSD
    RIGID                   ///< Vertices follows rigidely the nearest bone
};

// -----------------------------------------------------------------------------

enum Cluster_type {
    /// Clusterize the mesh computing the euclidean dist to each bone
    EUCLIDEAN,
    /// Use bone weights to define the clusters. A vertex will be associated to
    /// highest bone weight influence.
    FROM_WEIGHTS
};

// -----------------------------------------------------------------------------

/// Painting modes for the different mesh attributes
enum Paint_type {
    PT_SSD_INTERPOLATION,
    PT_SSD_WEIGHTS,
    PT_CLUSTER
};

// -----------------------------------------------------------------------------



}
// END EAnimesh NAMESPACE ======================================================

#endif // ANIMATED_MESH_ENUM_HPP__
