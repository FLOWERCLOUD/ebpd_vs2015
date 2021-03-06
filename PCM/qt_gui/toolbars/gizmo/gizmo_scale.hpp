#ifndef GIZMO_SCALE_HPP__
#define GIZMO_SCALE_HPP__

#include "gizmo.hpp"
#include "toolbox/portable_includes/port_glew.h"
#include "toolbox/maths/vec3.hpp"
#include "toolbox/gl_utils/glsave.hpp"
#include "toolbox/maths/intersections/intersection.hpp"
#include "toolbox/maths/vec2.hpp"
#include "toolbox/gl_utils/glpick.hpp"
#include "../../global_datas/g_vbo_primitives.hpp"
#include "../../global_datas/toolglobals.hpp"
#include "toolbox/maths/color.hpp"

namespace Tbx
{
	class Camera;
}

//#define USE_SCALE // acts when gizmo_rot => WTF!!!

/**
  @name Gizmo_scale
  @brief 3D frame to provide GUI for object scaling.
  This class provide methods for scaling objects in the GUI.
  It memories the current scale and compute its new scale
  given the mouse position and axis/plane constraint. It also gives means to
  draw a custom oriented frame which is represented by arrows. Each arrow can be
  selected.

  @see Gizmo
*/

// TODO: to be implemented

class Gizmo_scale : public Gizmo {
public:

    enum Constraint_t {AXIS, ALL};
    enum Axis_t {X , Y , Z, CIRCLE, NOAXIS };

    Gizmo_scale(): Gizmo(),
        _length(0.13f),
        _rad_handles(2.0f),
        _ortho_factor(20.f),
        _constraint(AXIS),
        _selected_axis(NOAXIS)
    { _frame = Tbx::Transfo::identity(); }

    /// draw the current selected point and frame. Frame is represented with
    /// arrows (green, red and blue) made of a cylinder and a large point for the tip.
    /// Frame is drawn at the position specified by '_frame.get_translation()'.
    /// X, Y and Z arrows orientations are defined by 'set_frame()'.
    void draw(const Tbx::Camera& cam)
    {
#ifdef USE_SCALE
        if(!_show) return;

        glPushMatrix();
        GLEnabledSave save_light(GL_LIGHTING  , true, false);
        GLEnabledSave save_depth(GL_DEPTH_TEST, true, false);
        GLEnabledSave save_blend(GL_BLEND     , true, false);
        GLEnabledSave save_alpha(GL_ALPHA_TEST, true, false);
        GLEnabledSave save_textu(GL_TEXTURE_2D, true, false);
        GLLineWidthSave save_line_width;

        const Vec3 org = _frame.get_translation();
        const float dist_cam =  !cam.is_ortho() ? (org-cam.get_pos()).norm() : _ortho_factor;

        glMultMatrixf(_frame.transpose().m);
        glScalef(dist_cam, dist_cam, dist_cam);

        /* Axis X */
        if ((_constraint == AXIS && _selected_axis == X) || (_constraint == ALL && _selected_axis == CIRCLE) )
            glColor3f(1.f, 1.f, 0.f);
        else
            glColor3f(1.f, 0.f, 0.f);
        glPushMatrix();
        glRotatef(90.f, 0.f, 1.f, 0.f);
        draw_arrow(_length);
        glPopMatrix();

        /* Axis Y */
        if ((_constraint == AXIS && _selected_axis == Y ) || (_constraint == ALL && _selected_axis == CIRCLE) )
            glColor3f(1.f, 1.f, 0.f);
        else
            glColor3f(0.f, 1.f, 0.f);
        glPushMatrix();
        glRotatef(-90.f, 1.f, 0.f, 0.f);
        draw_arrow(_length);
        glPopMatrix();

        /* Axis Z */
        if ((_constraint == AXIS && _selected_axis == Z ) || (_constraint == ALL && _selected_axis == CIRCLE) )
            glColor3f(1.f, 1.f, 0.f);
        else
            glColor3f(0.f, 0.f, 1.f);
        glPushMatrix();
        draw_arrow(_length);
        glPopMatrix();

        glPopMatrix();

        // Circle parallel to the camera // todo : fix
        glPushMatrix();
        if(_pick.is_pick_init()){
            glLineWidth( _rad_handles*20.0f ); // So that picking is easier
            glDisable(GL_LINE_SMOOTH);
        }
        else
            glLineWidth( _rad_handles );
        draw_circle(cam.get_dir(), dist_cam / 50.f * 6.5f, Color::white());
        glPopMatrix();
#endif
    }

    /// select the x or y or z frame axis given a camera and a mouse position
    /// updates attributes _constraint
    /// @return true if one of the axis is selected
    bool select_constraint(const Tbx::Camera& cam, int px, int py)
    {
#ifdef USE_SCALE
        if(!_show){
            _selected_axis = NOAXIS;
            return false;
        }

        _old_frame = _frame;
        _org.x = px;
        _org.y = py;
        // try to select axes
        using namespace Inter;
        float t0, t1, t2;
        t0 = t1 = t2 = std::numeric_limits<float>::infinity();

        const Vec3 org = _frame.get_translation();
        const float dist_cam = !cam.is_ortho() ?
                               (org-cam.get_pos()).norm() : _ortho_factor;

        // TODO: use picking and not ray primitive intersection
        Ray_cu r = cam.cast_ray(px, py);

        const float s_radius = dist_cam * 0.3f/50.f;
        const float s_length = dist_cam * _length;
        Inter::Cylinder cylinder_x(s_radius, s_length, _frame.x(), org );
        Inter::Cylinder cylinder_y(s_radius, s_length, _frame.y(), org );
        Inter::Cylinder cylinder_z(s_radius, s_length, _frame.z(), org );
        Line cam_ray(r._dir, r._pos );
        Point nil;
        bool res = false;

        res = res || cylinder_line( cylinder_x, cam_ray, nil, t0 );
        res = res || cylinder_line( cylinder_y, cam_ray, nil, t1 );
        res = res || cylinder_line( cylinder_z, cam_ray, nil, t2 );

        GLint viewport[4];
        GLdouble modelview[16];
        GLdouble projection[16];
        glGetIntegerv(GL_VIEWPORT, viewport);
        glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
        glGetDoublev(GL_PROJECTION_MATRIX, projection);

        GLdouble cx, cy, cz;
        gluProject(org.x, org.y, org.z,
                   modelview, projection, viewport,
                   &cx, &cy, &cz);

        cy = cam.height() - cy;
        _pix_diff.x = (int)(cx-px);
        _pix_diff.y = (int)(cy-py);

        if ( res ){
            _constraint = AXIS;
            if(t0 < t1 && t0 < t2){_selected_axis = X; return true;}
            if(t1 < t0 && t1 < t2){_selected_axis = Y; return true;}
            if(t2 < t1 && t2 < t0){_selected_axis = Z; return true;}
            return false;
        } else {
            // try to pick circle
            GLfloat m[16];
            glGetFloatv(GL_PROJECTION_MATRIX, m);
            _pick.begin( m, (GLfloat)px, (GLfloat)(cam.height() - py) );
            draw( cam );
            int idx = _pick.end();
            if ( idx  == (int)CIRCLE ){
                _constraint = ALL;
                _selected_axis = CIRCLE;
                return true;
            } else {
                _constraint = AXIS;
                _selected_axis = NOAXIS;
                return false;
            }
        }
#else
        return false;
#endif
    }

    /// reset the selected constraint set by select_constraint()
    void reset_constraint()
    {
        _constraint = AXIS;
        _selected_axis = NOAXIS;
    }

    /// @brief scale object given the current constraint (plane or axis)
    /// If a constraint has been selected this will scale the selected object
    /// following the constraint and keeping the frame arrow
    /// as close as possible under the mouse position (px, py)
    /// @return the scaling made by the frame in world coordinates
    /// @see select_constraint()
    Tbx::TRS slide(const Tbx::Camera& cam, int px, int py)
    {
#ifdef USE_SCALE
        if( _selected_axis == CIRCLE) {
            return  slide_axis(cam, px, py);
        } else if (_constraint == AXIS && _selected_axis != NOAXIS){
            if( Vec2((float)px-_org.x, (float)py-_org.y).norm() < 0.0001f )
                return TRS();
            else {
                return slide_axis(cam, px, py);
            }

        }
        return TRS();
#else
        return TRS();
#endif
    }

private:

    void draw_arrow(float length){
        Tbx::GLPointSizeSave pt_s( 10 );
        Tbx::GLLineWidthSave ln_s(1.f);
        glBegin(GL_LINES);
        glVertex3f(0.f, 0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.8f*length);
        glAssert( glEnd() );
        glBegin(GL_POINTS);
        glVertex3f(0.f, 0.f, 0.8f*length);
        glAssert( glEnd() );
    }

    void draw_circle(const Tbx::Vec3& cam_dir,
                     float scale,
                     const Tbx::Color& cl)
    {
#ifdef USE_SCALE
        glPushMatrix();
        {
            Vec3 fy, fz;
            cam_dir.coordinate_system(fy, fz);
            Transfo tr(Mat3_cu(cam_dir, fy, fz), _frame.get_translation());

            glMultMatrixf( tr.transpose().m );
            glScalef(scale, scale, scale);

            if( _selected_axis == CIRCLE )
                Color::yellow().set_gl_state();
            else
                cl.set_gl_state();

            _pick.set_name(CIRCLE);
            glRotatef(90.f, 0.f, 1.f, 0.f);
            g_primitive_printer.draw( g_circle_vbo );

            if( _pick.is_pick_init() )
                draw_thick_prim( g_circle_lr_vbo );
        }
        glPopMatrix();
#endif
    }

    void draw_thick_prim( Prim_id id )
    {
#ifdef USE_SCALE
        for (int i = 0; i < 8; ++i) {
            float s = i*0.01f;
            glPushMatrix();
            glTranslatef(0.f, 0.f, s);
            glScalef( 1.f+s, 1.f+s, 1.f+s );
            g_primitive_printer.draw( id );
            glRotatef(-15.f, 0.f, 0.f, 1.f);
            g_primitive_printer.draw( id );
            glRotatef(30.f, 0.f, 0.f, 1.f);
            g_primitive_printer.draw( id );
            glPopMatrix();

            glPushMatrix();
            glTranslatef(0.f, 0.f, -s);
            glScalef( 1.f-s, 1.f-s, 1.f-s );
            g_primitive_printer.draw( id );
            glRotatef(-15.f, 0.f, 0.f, 1.f);
            g_primitive_printer.draw( id );
            glRotatef(30.f, 0.f, 0.f, 1.f);
            g_primitive_printer.draw( id );
            glPopMatrix();
        }
#endif
    }

    Tbx::TRS slide_axis(const Tbx::Camera& cam, int px, int py)
    {
        // todo : use gizmo center to set scale > 1 or < 1
        // todo : project/unproject points for better-suited scale along axes
        float scale = sqrtf( (float)(_pix_diff.x*_pix_diff.x + _pix_diff.y*_pix_diff.y) ) / 100;
        Tbx::Vec3 org = _frame.get_translation();
        org = cam.project( org.to_point3() );
        // to be improved
        if ( (org.x-_org.x)*(org.x-_org.x)+(org.y-_org.y)*(org.y-_org.y) <
             (org.x-px)*(org.x-px)+(org.y-py)*(org.y-py) )
            scale = 1.f/scale;
        if (_selected_axis == X){
            return TRS::scale( _old_frame.fast_invert() * Vec3(scale,     1,     1) );
        } else if (_selected_axis == Y){
            return TRS::scale( _old_frame.fast_invert() * Vec3(    1, scale,     1) );
        } else if (_selected_axis == Z){
            return TRS::scale( _old_frame.fast_invert() * Vec3(    1,     1, scale) );
        } else {
            return TRS::scale( _old_frame.fast_invert() * Vec3(scale, scale, scale) );
        }
    }

    float _length;
    float _rad_handles;   ///< Radius of the tore reprensenting the handles
    float _ortho_factor;

    Constraint_t _constraint;     ///< current constraint (axis or sphere movement)
    Axis_t       _selected_axis;  ///< current selected axis
    Pix          _org;            ///< pixel clicked when a constraint is selected
    Pix          _pix_diff;       ///< pixels to substract when sliding

    GLPick _pick;
};

#endif // GIZMO_SCALE_HPP__
