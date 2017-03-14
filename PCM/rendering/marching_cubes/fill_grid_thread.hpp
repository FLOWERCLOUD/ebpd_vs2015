#ifndef FILL_GRID_THREAD_HPP__
#define FILL_GRID_THREAD_HPP__

#include <QRunnable>

#include "toolbox/containers/idx3.hpp"
#include <vector>

#include "scene_tree/implicit_surfaces/node_implicit_surface.hpp"

// =============================================================================
namespace Marching_cubes {
// =============================================================================

/**
 * @class Fill_grid_thread
 * @brief thread encapsulation to fill a sub part of a 3d grid with a
 * potential field
 *
 *
 */
class Fill_grid_thread : public QRunnable {
public:

    /// @param sub_res_ : size of the sub grid to be looked up
    /// @param offset_ : offset inside the grid from which we start the look up
    /// @param world_resolution_ : size of the actual grid (ie. data_field_)
    /// @param box_start_ : 3d position of the grid in the scene.
    /// @param world_step_ : lengths of the grid's cells
    /// @param data_field_ : the 3d grid stored linearly of size
    /// world_resolution_.x * world_resolution_.y * world_resolution_.z
    /// @param obj_ : the implicit objet / tree with which we fill the grid
    /// with its scalar values.
    Fill_grid_thread(Vec3i sub_res_,
                     Vec3i offset_,
                     Vec3i world_resolution_,
                     Vec3 box_start_,
                     Vec3 world_step_,
                     std::vector<float>& data_field_,
                     const Node_implicit_surface* obj_);

protected:
    /// Look up the sub-grid and fill it
    void run();

    Vec3i _sub_res;
    Vec3i _offset;

    Vec3i _world_resolution;
    Vec3 _box_start;
    Vec3 _world_step;

    std::vector<float>& _data_field;
    const Node_implicit_surface* _obj;
};

}// END MARCHING_CUBE ==========================================================

#endif // FILL_GRID_THREAD_HPP__
