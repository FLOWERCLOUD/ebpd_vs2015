#ifndef POTENTIAL_PLANE_CTRL_HPP__
#define POTENTIAL_PLANE_CTRL_HPP__

#include "toolbox/maths/vec3.hpp"
#include "toolbox/gl_utils/glsave.hpp"

class Potential_plane_ctrl {
public:
    Potential_plane_ctrl() : _setup(false)
    {
        _normal.set(0.f, 1.f, 0.f);
        _org.set(0.f, 0.f, 0.f);
        _decal = 0.f;
    }

    void draw()
    {
        if(_setup){
            draw_plane(2000.f);
        }
    }

    bool       _setup;
    Tbx::Vec3 _normal;
    Tbx::Vec3 _org;
    float _decal;

//private:
    void draw_plane(float size)
    {
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();

        GLfloat newR[16];
        Tbx::Vec3 tmp = _normal;
        if(tmp.x != tmp.y || tmp.y != tmp.z || tmp.x != tmp.z)
            tmp = tmp + 1.f;
        else
            tmp = Tbx::Vec3(1.f,0.f,0.f);

        Tbx::Vec3 Rz = tmp.cross(_normal);
        Tbx::Vec3 Rx = _normal.cross(Rz);
        Rz.normalize(); _normal.normalize(); Rx.normalize();

        newR[0] = Rx.x; newR[4] = _normal.x; newR[ 8] = Rz.x; newR[12] = _org.x + _decal*_normal.x;
        newR[1] = Rx.y; newR[5] = _normal.y; newR[ 9] = Rz.y; newR[13] = _org.y + _decal*_normal.y;
        newR[2] = Rx.z; newR[6] = _normal.z; newR[10] = Rz.z; newR[14] = _org.z + _decal*_normal.z;
        newR[3] = 0.f ; newR[7] = 0.f;       newR[11] = 0.f ; newR[15] = 1.f;

        glMultMatrixf(newR);

        //GLEnabledSave save_light(GL_LIGHTING, true, true);
        glColor4f(1.f, 0.f, 0.f, 0.99f);
        glBegin (GL_QUADS);
        glNormal3f(0.f, 1.f, 0.f);
        glVertex3f(-(size/2.f), 0.f, -(size/2.f));
        glVertex3f(-(size/2.f), 0.f,  (size/2.f));
        glVertex3f( (size/2.f), 0.f,  (size/2.f));
        glVertex3f( (size/2.f), 0.f, -(size/2.f));
        glEnd();

        glPopMatrix();
    }
};

#endif // POTENTIAL_PLANE_CTRL_HPP__

