#include "debug_ctrl.hpp"

#include "../global_datas/toolglobals.hpp"

#include "toolbox/cuda_utils/cuda_utils.hpp"
#include "toolbox/gl_utils/gl_utils.hpp"
#include "../meshes/mesh.hpp"

using Tbx::Vec3;
// -----------------------------------------------------------------------------

void Debug_ctrl::draw_gradient(const std::vector<int>& selected_points,
                               const std::vector<Vec3>& d_gradient,
                               const std::vector<Vec3>& d_gradient_energy)
{
    Vec3* vert = 0;
    g_mesh->_mesh_gl._vbo.map_to(vert, GL_READ_ONLY);

#if 1
    glBegin(GL_LINES);
    for(unsigned i = 0; i<selected_points.size(); i++)
    {
        Vec3 n, n_energy;
        //Cuda_utils::mem_cpy_dth(&n, d_gradient + selected_points[i], 1);
        //Cuda_utils::mem_cpy_dth(&n_energy, d_gradient_energy + selected_points[i], 1);

        n.normalize();
        n = n * -5.f;
        const EMesh::Packed_data d = g_mesh->get_packed_vert_map()[selected_points[i]];

        Vec3 v = vert[ d._idx_data_unpacked ];

        glColor4f(0.f, 0.f, 1.f, 1.f);
        glVertex3f(v.x      , v.y      , v.z      );
        glVertex3f(v.x + n.x, v.y + n.y, v.z + n.z);

        n = n_energy.normalized() * -5.f;

        glColor4f(0.f, 1.f, 0.f, 1.f);
        glVertex3f(v.x      , v.y      , v.z      );
        glVertex3f(v.x + n.x, v.y + n.y, v.z + n.z);

    }
    glEnd();
#else
    for(unsigned i = 0; i<selected_points.size(); i++)
    {
        Vec3 n;
        Cuda_utils::mem_cpy_dth(&n, d_gradient + selected_points[i], 1);

        n.normalize();
        n = n * -5.f;
        const EMesh::Packed_data d = g_mesh->get_packed_vert_map()[selected_points[i]];

        Vec3 v = vert[ d._idx_data_unpacked ];
        Mat3 m = Mat3::coordinate_system( n.normalized() );
        Gl_utils::draw( Transfo(m, v) );
    }
#endif
    g_mesh->_mesh_gl._vbo.unmap();
}

// -----------------------------------------------------------------------------
