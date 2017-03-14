
#include "skeleton.hpp"
#include "animesh_kers.hpp"


void compute_pot_launch_ker(const Cuda_utils::DA_Vec3& d_verts,
                            Cuda_utils::DA_float& d_pot)
{
    const int nb_verts = d_verts.size();
    const int block_size = 256;
    const int grid_size = (nb_verts + block_size - 1) / block_size;

    Animesh_kers::compute_base_potential<<<grid_size, block_size>>>
        (0, // HACK: always skeleton id zero
         (Point3*)d_verts.ptr(),
         d_verts.size(),
         d_pot.ptr(),
         0);
}


