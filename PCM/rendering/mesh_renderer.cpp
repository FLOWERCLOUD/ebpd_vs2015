#include "rendering/mesh_renderer.hpp"

#include "toolbox/gl_utils/glbuffer_object.hpp"
#include "toolbox/gl_utils/glsave.hpp"
#include "toolbox/gl_utils/gltex2D.hpp"
#include "meshes/mesh.hpp"
#include "global_datas/g_shaders.hpp"



// =============================================================================
namespace Mesh_renderer {
// =============================================================================

/// Setting up the current shader state knowing the current material 'mat'
/// @param tangent_attr if the material has bump we need to set a tangent
/// attribute. this parameter return the index of the attribute bound to th
/// tangent.
void shader_setup(const Mesh& mesh, const EMesh::Material mat, int& tangent_attr)
{
    if( mesh.has_tex_coords() )
    {
        Tbx::Shader_prog* prog;
        if(mat._map_kd != 0 && mat._map_bump != 0)
        {
            prog = g_phong_list[MAP_KD_BUMP];
            prog->use();

            glActiveTexture(GL_TEXTURE0 + Tex_units::KD);
            mat._map_kd->bind();
            prog->set_uniform("map_Kd", Tex_units::KD);

            tangent_attr = prog->get_attribute_location("attr_Tangent");

            if(tangent_attr != -1)
            {
                glAssert( glEnableVertexAttribArray(tangent_attr) );
                mesh._mesh_gl._tangents_bo.bind();
                glAssert( glVertexAttribPointer(tangent_attr, 3, GL_FLOAT, GL_FALSE, 0, 0) );
            }

            glActiveTexture(GL_TEXTURE0 + Tex_units::BUMP);
            mat._map_bump->bind();
            prog->set_uniform("map_Bump", Tex_units::BUMP);
        }
        else if( mat._map_kd != 0 )
        {
            prog = g_phong_list[MAP_KD];
            prog->use();

            glActiveTexture(GL_TEXTURE0 + Tex_units::KD);
            mat._map_kd->bind();
            prog->set_uniform("map_Kd", Tex_units::KD);
        }
        else
            g_phong_list[NO_TEX]->use();
    }
    else
        g_phong_list[NO_TEX]->use();

}

// -----------------------------------------------------------------------------

void enable_client_state(const Mesh& mesh,
                         const Tbx::GlBuffer_obj<Vec3>& vbo,
                         const Tbx::GlBuffer_obj<Vec3>& nbo)
{
    glAssert( glEnableClientState(GL_VERTEX_ARRAY) );
    vbo.bind();
    glAssert( glVertexPointer(3, GL_FLOAT, 0, 0) );

    assert(mesh.has_normals());
    nbo.bind();
    glAssert( glEnableClientState(GL_NORMAL_ARRAY) );
    glAssert( glNormalPointer(GL_FLOAT, 0, 0) );

    GLEnabledSave save_color_mat(GL_COLOR_MATERIAL, true, true);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    if(mesh.has_tex_coords())
    {
        glAssert( glEnableClientState(GL_TEXTURE_COORD_ARRAY) );
        mesh._mesh_gl._tex_bo.bind();
        glAssert( glTexCoordPointer(2, GL_FLOAT, 0, 0) );
    }

}

// -----------------------------------------------------------------------------

void disable_client_state(const Mesh& mesh)
{
    glAssert( glBindBuffer(GL_ARRAY_BUFFER, 0) );

    glAssert( glVertexPointer(3, GL_FLOAT, 0, 0) );
    glAssert( glNormalPointer(GL_FLOAT, 0, 0) );
    glAssert( glDisableClientState(GL_VERTEX_ARRAY) );
    glAssert( glDisableClientState(GL_NORMAL_ARRAY) );

    if(mesh.has_tex_coords())
    {
        glAssert( glTexCoordPointer(2, GL_FLOAT, 0, 0) );
        glAssert( glDisableClientState(GL_TEXTURE_COORD_ARRAY) );
    }
}
// -----------------------------------------------------------------------------

void draw(const Mesh& mesh,
          const Tbx::GlBuffer_obj<Vec3>& vbo,
          const Tbx::GlBuffer_obj<Vec3>& nbo,
          bool enable_texture)
{
    if(mesh.get_nb_vertices() == 0) return;

    if(!mesh.has_materials() || !enable_texture)
    {
        g_phong_list[NO_TEX]->use();
        mesh.draw_using_buffer_object(vbo, nbo, mesh._mesh_gl._color_bo, false);
        g_phong_list[NO_TEX]->unuse();
        return;
    }

    enable_client_state(mesh, vbo, nbo);

    GLEnabledSave save_tex(GL_TEXTURE_2D, false);

    if(mesh.has_tex_coords()) glEnable (GL_TEXTURE_2D);
    else                      glDisable(GL_TEXTURE_2D);


    const std::vector<EMesh::Mat_grp>&  mat_grps = mesh.get_mat_grps_tri();
    const std::vector<EMesh::Material>& mat_list = mesh.get_mat_list();

    GLEnabledSave save_blend(GL_BLEND, false);
    GLEnabledSave save_color_mat(GL_COLOR_MATERIAL, true, true);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    int tangent_attr = -1;

    glColor4f(1.f, 1.f, 1.f, 1.f);
    mesh._mesh_gl._index_bo_tri.bind();

    int prev_mat = -1;
    bool is_transluscent = false;
    for(unsigned i = 0; i < mat_grps.size(); i++)
    {
        EMesh::Mat_grp grp = mat_grps[i];

        if(prev_mat != grp.mat_idx)
        {
            EMesh::Material mat = mat_list[grp.mat_idx];
            mat.setup_opengl_materials();
            is_transluscent = ((mat._tf[0]+mat._tf[1]+mat._tf[2])/3.0f) <= (1.f-0.001f);

            shader_setup(mesh, mat, tangent_attr);
        }

        if(is_transluscent)
        {
            // We need some sort of basic sorting
            GLEnabledSave save_culling(GL_CULL_FACE, true, true);
            glCullFace(GL_FRONT);
            glAssert( glDrawElements(GL_TRIANGLES, grp.nb_face * 3, GL_UNSIGNED_INT, (GLvoid*)(0+grp.starting_idx*3*sizeof(int))) );
            glCullFace(GL_BACK);
            glAssert( glDrawElements(GL_TRIANGLES, grp.nb_face * 3, GL_UNSIGNED_INT, (GLvoid*)(0+grp.starting_idx*3*sizeof(int))) );
        }
        else
            glAssert( glDrawElements(GL_TRIANGLES, grp.nb_face * 3, GL_UNSIGNED_INT, (GLvoid*)(0+grp.starting_idx*3*sizeof(int))) );

        prev_mat = grp.mat_idx;
    }
    mesh._mesh_gl._index_bo_tri.unbind();

    Shader_prog::unuse();

    // TODO: when obj_loder will support quads materials for quads must be used
    //for(unsigned i = 0; i < material_grps_tri.size(); i++)
    //_index_bo_quad.bind();
    //glAssert( glDrawElements(GL_QUADS, 4 * _nb_quad, GL_UNSIGNED_INT, 0) );
    //_index_bo_quad.unbind();

    disable_client_state(mesh);
    GlTex2D::unbind();

    if( tangent_attr != -1 ) glAssert( glDisableVertexAttribArray(tangent_attr) );
}


}// END Mesh_renderer ==========================================================
