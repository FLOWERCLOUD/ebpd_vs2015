#include "gizmo_trans.hpp"

#include <limits>

#include "toolbox/gl_utils/glsave.hpp"
#include "toolbox/maths/intersections/intersection.hpp"
#include "toolbox/maths/vec3.hpp"
#include "toolbox/maths/vec2.hpp"
#include "../../control/cuda_ctrl.hpp"
#include "../../rendering/camera.hpp"
using namespace Cuda_ctrl;
using namespace Tbx;
// -----------------------------------------------------------------------------

static void draw_arrow(float radius, float length){
	if(0) 
	{
		GLUquadricObj* quad = gluNewQuadric();

		glLineWidth(1.f);
		//gluCylinder(quad, radius/3.f, radius/3.f, 0.8f * length, 10, 10);
		glBegin(GL_LINES);
		glVertex3f(0.f, 0.f, 0.f);
		glVertex3f(0.f, 0.f, 0.8f*length);
		glAssert( glEnd() );
		glTranslatef(0.f, 0.f, 0.8f * length);
		gluCylinder(quad, radius, 0.0f  , 0.2f * length, 10, 10);

		gluDeleteQuadric(quad);
	}else
	{//gizmo2
		GLUquadricObj* quad = gluNewQuadric();
		glPushMatrix();
		glLineWidth(2.f);
		glBegin(GL_LINES);
		glVertex3f(0.f, 0.f, 0.f);
		glVertex3f(0.f, 0.f, 0.8f*length);
		glAssert( glEnd() );

		glTranslatef(0.f, 0.f, 0.8f * length);
		gluCylinder(quad, radius, 0.0f  , 0.2f * length, 10, 10);
		glPopMatrix();

		gluDeleteQuadric(quad);
	}

}

static void draw_quad( float l )
{
	l *= 0.2f;

	glBegin(GL_QUADS);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(l  , 0.f, 0.f);
	glVertex3f(l  , l, 0.f);
	glVertex3f(0.f , l, 0.f);
	glEnd();

	GLPolygonModeSave mode( GL_LINE );
	GLLineWidthSave width(1.0f);
	glColor4f( 0.3f, 0.3f, 0.3f, 1.f);
	glBegin(GL_LINES);
	glVertex3f(l  , 0.f, 0.f);
	glVertex3f(l  , l, 0.f);
	glVertex3f(l  , l, 0.f);
	glVertex3f(0.f , l, 0.f);
	glEnd();
}

// -----------------------------------------------------------------------------

void Gizmo_trans::draw_quads()
{
	glPushMatrix();
	{
		/* PLANE XY */
		if(_selected_axis == XY )
			glColor4f(1.f, 1.f, 0.f, 0.5f);
		else
			glColor4f( 0.4f, 0.4f, 0.4f, 0.5f);

		if( _picker.is_pick_init() ) _picker.set_name( XY );
		draw_quad( _length );

		/* PLANE XZ */
		if(_selected_axis == XZ )
			glColor4f(1.f, 1.f, 0.f, 0.5f);
		else
			glColor4f( 0.4f, 0.4f, 0.4f, 0.5f);

		glRotatef(90.f, 1.f, 0.f, 0.f);
		if( _picker.is_pick_init() ) _picker.set_name( XZ );
		draw_quad( _length );

		/* PLANE YZ*/
		if(_selected_axis == YZ )
			glColor4f(1.f, 1.f, 0.f, 0.5f);
		else
			glColor4f( 0.4f, 0.4f, 0.4f, 0.5f);

		glRotatef(90.f, 0.f, 1.f, 0.f);
		if( _picker.is_pick_init() ) _picker.set_name( YZ );
		draw_quad( _length );
	}
	glPopMatrix();
}

// -----------------------------------------------------------------------------

/// @return deem factor arrow (proportionnal to the angle between dir and axis)
static float alpha_arrow(const Vec3& dir, const Vec3& axis)
{
	float angle_x = std::abs( dir.dot( axis) );
	float alpha = 1.f - (std::max( angle_x, 0.99f) - 0.99f) * 100.f;
	return alpha;
}


//void Gizmo_trans::draw(const Camera& cam)
//{
//    if(!_show) return;
//
//    glPushMatrix();
//    GLEnabledSave save_light(GL_LIGHTING  , true, false);
//    GLEnabledSave save_depth(GL_DEPTH_TEST, true, false);
//    GLEnabledSave save_blend(GL_BLEND     , true, false);
//    GLEnabledSave save_alpha(GL_ALPHA_TEST, true, false);
//    GLEnabledSave save_textu(GL_TEXTURE_2D, true, false);
//
//    const Vec3 org = _frame.get_translation();
//    const float dist_cam =  !cam.is_ortho() ? (org-cam.get_pos()).norm() : _ortho_factor;
//
//    glMultMatrixf(_frame.transpose().m);
//    glScalef(dist_cam, dist_cam, dist_cam);
//
//    /* Axis X */
//    if(_constraint == AXIS && _selected_axis == X )
//        glColor3f(1.f, 1.f, 0.f);
//    else
//        glColor3f(1.f, 0.f, 0.f);
//    glPushMatrix();
//    glRotatef(90.f, 0.f, 1.f, 0.f);
//    draw_arrow(_radius, _length);
//    glPopMatrix();
//
//    /* Axis Y */
//    if(_constraint == AXIS && _selected_axis == Y )
//        glColor3f(1.f, 1.f, 0.f);
//    else
//        glColor3f(0.f, 1.f, 0.f);
//    glPushMatrix();
//    glRotatef(-90.f, 1.f, 0.f, 0.f);
//    draw_arrow(_radius, _length);
//    glPopMatrix();
//
//    /* Axis Z */
//    if(_constraint == AXIS && _selected_axis == Z )
//        glColor3f(1.f, 1.f, 0.f);
//    else
//        glColor3f(0.f, 0.f, 1.f);
//    glPushMatrix();
//    draw_arrow(_radius, _length);
//    glPopMatrix();
//
//    glPopMatrix();
//}
void Gizmo_trans::draw(const Camera& cam)
{
	if(!_show) return;

	glPushMatrix();
	GLEnabledSave save_light(GL_LIGHTING  , true, false);
	GLEnabledSave save_depth(GL_DEPTH_TEST, true, false);
	GLEnabledSave save_blend(GL_BLEND     , true, true);
	GLBlendSave save_blend_eq;
	glAssert( glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) );
	GLEnabledSave save_alpha(GL_ALPHA_TEST, true, true);
	GLEnabledSave save_textu(GL_TEXTURE_2D, true, false);
	GLLineWidthSave save_line;
	if( _picker.is_pick_init() )
		glLineWidth(10.f);
	else
		glLineWidth(1.f);

	const Vec3 org = _frame.get_translation();
	const float dist_cam =  !cam.is_ortho() ? (org-cam.get_pos()).norm() : cam.get_ortho_zoom() * _ortho_factor;

	glMultMatrixf(_frame.transpose().m);
	glScalef(dist_cam, dist_cam, dist_cam);

	float alpha_x = alpha_arrow( cam.get_dir(), _frame.x() );
	float alpha_y = alpha_arrow( cam.get_dir(), _frame.y() );
	float alpha_z = alpha_arrow( cam.get_dir(), _frame.z() );

	/* Axis X */
	if(_selected_axis == X )
		glColor3f(1.f, 1.f, 0.f);
	else
		glColor4f(1.f, 0.f, 0.f, alpha_x);
	glPushMatrix();
	glRotatef(90.f, 0.f, 1.f, 0.f);
	if( _picker.is_pick_init() ) _picker.set_name( X );
	draw_arrow(_radius, _length);
	glPopMatrix();

	/* Axis Y */
	if(_selected_axis == Y )
		glColor3f(1.f, 1.f, 0.f);
	else
		glColor4f(0.f, 1.f, 0.f, alpha_y);
	glPushMatrix();
	glRotatef(-90.f, 1.f, 0.f, 0.f);
	if( _picker.is_pick_init() ) _picker.set_name( Y );
	draw_arrow(_radius, _length);
	glPopMatrix();

	/* Axis Z */
	if(_selected_axis == Z )
		glColor3f(1.f, 1.f, 0.f);
	else
		glColor4f(0.f, 0.f, 1.f, alpha_z);
	glPushMatrix();
	if( _picker.is_pick_init() ) _picker.set_name( Z );
	draw_arrow(_radius, _length);
	glPopMatrix();

	draw_quads();

	glPopMatrix();
}
// -----------------------------------------------------------------------------

//bool Gizmo_trans::select_constraint(const Camera& cam, int px, int py)
//{
//    if(!_show){
//        _selected_axis = NOAXIS;
//        return false;
//    }
//
//    _old_frame = _frame;
//    _org.x = px;
//    _org.y = py;
//
//    using namespace Inter;
//    float t0, t1, t2;
//    t0 = t1 = t2 = std::numeric_limits<float>::infinity();
//
//    const Vec3 org = _frame.get_translation();
//    const float dist_cam = !cam.is_ortho() ?
//                           (org-cam.get_pos()).norm() : _ortho_factor;
//
//    // TODO: use picking and not ray primitive intersection
//    Ray_cu r = cam.cast_ray(px, py);
//
//    const float s_radius = dist_cam * _radius;
//    const float s_length = dist_cam * _length;
//    Cylinder cylinder_x(s_radius, s_length, _frame.x(), org );
//    Cylinder cylinder_y(s_radius, s_length, _frame.y(), org );
//    Cylinder cylinder_z(s_radius, s_length, _frame.z(), org );
//    Line3d cam_ray(r._dir, r._pos.to_vec3() );
//    Point3d nil;
//    bool res = false;
//
////    if( cam->is_ortho() ){
////        Plane p(cam.get_dir(), cam.get_pos());
////        cylinder_x.project(p);
////        cylinder_y.project(p);
////        cylinder_z.project(p);
////    }
//
//    res = res || cylinder_line( cylinder_x, cam_ray, nil, t0 );
//    res = res || cylinder_line( cylinder_y, cam_ray, nil, t1 );
//    res = res || cylinder_line( cylinder_z, cam_ray, nil, t2 );
//
//    // TODO: use camera.project(org)
//    GLint viewport[4];
//    GLdouble modelview[16];
//    GLdouble projection[16];
//    glGetIntegerv(GL_VIEWPORT, viewport);
//    glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
//    glGetDoublev(GL_PROJECTION_MATRIX, projection);
//
//    GLdouble cx, cy, cz;
//    gluProject(org.x, org.y, org.z,
//               modelview, projection, viewport,
//               &cx, &cy, &cz);
//
//    cy = cam.height() - cy;
//    _pix_diff.x = (int)(cx-px);
//    _pix_diff.y = (int)(cy-py);
//
//    if( res )
//    {
//        _constraint = AXIS;
//        if(t0 < t1 && t0 < t2){_axis = _frame.x(); _selected_axis = X; return true;}
//        if(t1 < t0 && t1 < t2){_axis = _frame.y(); _selected_axis = Y; return true;}
//        if(t2 < t1 && t2 < t0){_axis = _frame.z(); _selected_axis = Z; return true;}
//    }
//
//    return res;
//}

bool Gizmo_trans::select_constraint(const Camera& cam, int px, int py)
{
	if(!_show){
		_selected_axis = NOAXIS;
		return false;
	}

	py = cam.height() - py; /////////// Invert because of opengl
	slide_from(_frame, Vec2i(px, py) );
	GLfloat m[16];
	glGetFloatv(GL_PROJECTION_MATRIX, m);
	//Transfo mvp = (cam.get_proj_transfo()*cam.get_eye_transfo()).transpose();
	_picker.begin(m, px ,py);
	//_picker.begin(mvp.m, px ,py);
	draw( cam );
	int id = _picker.end();


	const Vec3 org = _frame.get_translation();
	Point3 c = cam.project( org.to_point3() );
	c.y = cam.height() - c.y;
	_pix_diff.x = (int)(c.x-px);
	_pix_diff.y = (int)(c.y-py);

	_selected_axis = NOAXIS;
	if( id > -1 )
	{
		_selected_axis = (Axis_t)id;
		switch(_selected_axis){
		case X : _axis = _frame.x(); break;
		case Y : _axis = _frame.y(); break;
		case Z : _axis = _frame.z(); break;
		case XY: _axis = _frame.z(); break;
		case XZ: _axis = _frame.y(); break;
		case YZ: _axis = _frame.x(); break;
		default: assert(false); break;
		}
		return true;
	}
	else
		return false;
}
// -----------------------------------------------------------------------------

TRS Gizmo_trans::slide(const Camera& cam, int px, int py)
{
	Vec2i pix(px, py);
    if(_constraint == AXIS && _selected_axis != NOAXIS)
    {
        if( Vec2((float)px-_org.x, (float)py-_org.y).norm() < 0.0001f )
            return TRS();
        else if( !is_plane_constraint() )
            return slide_axis(_axis, cam, px, py);
		else
			return slide_plane(_axis, cam, pix);
    }

    // TODO: why bother discriminating cases between X Y Z when we can use
    // the attribute _axis and _plane to. This gives rise to the use of useless
    // switchs ...
    // TODO: plane constraint :
    /*
        else
            switch(_selected_axis){
            case(XY): return slide_plane(_frame_z, cam, px, py); break;
            case(XZ): return slide_plane(_frame_y, cam, px, py); break;
            case(YZ): return slide_plane(_frame_x, cam, px, py); break;
            }
    */

    return TRS();
}

// -----------------------------------------------------------------------------

TRS Gizmo_trans::slide_axis(Vec3 axis_dir, const Camera& cam, int px, int py)
{
    px += _pix_diff.x;
    py += _pix_diff.y;

    // TODO:handle orthogonal projection
    Ray_cu  ray = cam.cast_ray(px, py);
    Vec3 up  = cam.get_y();
    Vec3 dir = cam.get_dir();

    // Find a plane passing through axis_dir and as parrallel as possible to
    // the camera image plane
    Vec3 ortho = dir.cross(axis_dir);
    Vec3 p_normal;
    if(ortho.norm() < 0.00001) p_normal = axis_dir.cross( up  );
    else                       p_normal = axis_dir.cross(ortho);


    // Intersection between that plane and the ray cast by the mouse
    const Vec3 org = _old_frame.get_org();
    Inter::Plane axis_plane(p_normal, org);
    Inter::Line3d l_ray(ray._dir, ray._pos.to_vec3());
    Vec3 slide_point;
    bool inter = Inter::plane_line(axis_plane, l_ray, slide_point);

    // Project on axis_dir the intersection point
    dir = axis_dir.normalized();
    slide_point = org + dir * dir.dot(slide_point-org);

    if(inter )
        return TRS::translation( _old_frame.fast_invert() * (slide_point-org) );
    else
        return TRS();
}

TRS Gizmo_trans::slide_plane(Vec3 normal  , const Camera& cam, Tbx::Vec2i pix)
{
	pix.x +=_pix_diff.x;
	pix.y += _pix_diff.y;

	Ray_cu  ray = cam.cast_ray(pix.x, pix.y);

	// Intersection between that plane and the ray cast by the mouse
	const Vec3 org = _start_frame.get_org();
	Inter::Plane axis_plane(normal, org);
	Inter::Line3d l_ray(ray._dir, ray._pos);
	Vec3 slide_point;
	bool inter = Inter::plane_line(axis_plane, l_ray, slide_point);

	if(inter )
		return TRS::translation( slide_point-org );
	else
		return TRS();
}
