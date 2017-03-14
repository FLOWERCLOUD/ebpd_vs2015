#include "mesh_materials.hpp"

#include "toolbox/portable_includes/port_glew.h"

// =============================================================================
namespace EMesh {
// =============================================================================

Material::Material() :
    _map_ka(0),
    _map_kd(0),
    _map_ks(0),
    _map_bump(0),
    _bump_strength(0)
{
    set_ka(0.1f, 0.1f, 0.1f, 1.f);
    set_kd(0.8f, 0.5f, 0.0f, 1.f);
    set_ks(0.5f, 0.5f, 0.5f, 1.f);
    set_ns(5.f);
    set_tf(1.f);
}

// -------------------------------------------------------------------------

void Material::setup_opengl_materials()
{
    float average_transp = (_tf[0] + _tf[1] + _tf[2]) / 3.0f;
    _ka[3] = _kd[3] = _ks[3] = average_transp;
    glAssert( glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,   _ka) );
    glAssert( glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,   _kd) );
    glAssert( glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  _ks) );
    glAssert( glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, _ns) );

    if(average_transp <= (1.f - 0.001f)) glAssert( glEnable(GL_BLEND)  );
    else                                 glAssert( glDisable(GL_BLEND) );
}

}// End Namespace EMesh ========================================================
