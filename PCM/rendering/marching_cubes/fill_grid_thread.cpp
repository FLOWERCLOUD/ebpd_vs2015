#include "rendering/marching_cubes/fill_grid_thread.hpp"

// =============================================================================
namespace Marching_cubes {
// =============================================================================

Fill_grid_thread::Fill_grid_thread(
        Vec3i sub_res_,
        Vec3i offset_,
        Vec3i world_resolution_,
        Vec3 box_start_,
        Vec3 world_step_,
        std::vector<float>& data_field_,
        const Node_implicit_surface* obj_) :
    _sub_res(sub_res_),
    _offset(offset_),
    _world_resolution(world_resolution_),
    _box_start(box_start_),
    _world_step(world_step_),
    _data_field(data_field_),
    _obj(obj_)
{

}

// -----------------------------------------------------------------------------

void Fill_grid_thread::run()
{
    Idx3 offset(_world_resolution, _offset);
    Idx3 idx;
    for(Idx3 sub_idx(_sub_res, 0); sub_idx.is_in(); ++sub_idx)
    {
        idx = offset + sub_idx.to_vec3i();
        Vec3 pos = _box_start + Vec3( idx.to_3d() ).mult(_world_step) + (_world_step * 0.5f);
        _data_field[ idx.to_linear() ] = _obj->f( pos );
    }
}

}// END MARCHING_CUBE ==========================================================
