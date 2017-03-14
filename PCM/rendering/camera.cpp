#include "rendering/camera.hpp"

#include "toolbox/portable_includes/port_glew.h"
#include "toolbox/maths/quat_cu.hpp"
#include "QGLViewer/camera.h"
#include <iostream>
#include <cassert>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923f
#endif

using namespace Tbx;
using std::cout;
using std::endl;
const float OTHO_ZOOM = 50.0f;
const int _offx =0;
const int _offy =0;
const bool _ortho_proj = false;
static const bool isdebug = false;
// -----------------------------------------------------------------------------
namespace Tbx
{
	Camera::Camera() :
    //_pos (0.f, 0.f, 0.f ),
    //_dir (0.f, 0.f, 1.f),
    //_x (1.f, 0.f, 0.f),
    //_y (0.f, 1.f, 0.f),
    //_fov(M_PI * 50.f / 180.f),
    //_near(1.f),
    //_far(1000.f),
    //_ortho_zoom(50.f),
    //_ortho_proj(false),
    //_width(100),
    //_height(100),
    //_offx(0),
    //_offy(0),
	m_camera(NULL)
	{	}

	Camera::Camera( qglviewer::Camera* _camera):
		m_camera(_camera)
	{
		if(isdebug)
			std::cout<<"Camera: camera convert"<<std::endl;
	}

	Camera::Camera( const qglviewer::Camera* _camera):
		m_camera( (qglviewer::Camera*)_camera)
	{
		if(isdebug)
			std::cout<<"Camera: camera convert"<<std::endl;
	}

	Tbx::Camera& Camera::operator = ( const Tbx::Camera& _camera)
		
	{
		this->m_camera = _camera.m_camera;
		return *this;
	}
// -----------------------------------------------------------------------------

	void Camera::update_dir(float yaw, float pitch, float roll)
	{
		if(isdebug)
			cout<<"Camera:update_dir"<<endl;
		Quat_cu rot = Quat_cu( get_y(), yaw) * Quat_cu( get_x(), pitch) * Quat_cu(get_dir(), roll);

		Vec3& dir = rot.rotate(get_dir()).normalized();
		Vec3& x   = rot.rotate(get_x()).normalized();
		Vec3& up   = rot.rotate(get_y()).normalized();
		qglviewer::Quaternion q;
		q.setFromRotatedBasis( qglviewer::Vec(x.x ,x.y,x.z) ,qglviewer::Vec(up.x ,up.y,up.z),qglviewer::Vec(dir.x ,dir.y,dir.z));
		m_camera->setOrientation( q);

	}

	// -----------------------------------------------------------------------------

	void Camera::set_fov_deg(float f)
	{
		m_camera->setFieldOfView( M_PI * f / 180.f);
	}

	// -----------------------------------------------------------------------------

	void Camera::set_fov_rad(float f)
	{
//		_fov = f;
		m_camera->setFieldOfView(f);
	}

	// -----------------------------------------------------------------------------

	void Camera::set_dir(const Vec3& dir)
	{
		if(isdebug)
			cout<<"Camera:set_dir"<<endl;
		//_dir = dir;
		//_dir.normalize();
		//Vec3 axe = std::abs( _dir.dot(Vec3::unit_y()) ) > 0.9f ? Vec3::unit_x() : Vec3::unit_y();
		//_x = axe.cross( _dir );
		//_x.normalize();
		//_y = _dir.cross(_x);
		m_camera->setViewDirection( qglviewer::Vec( dir.x ,dir. y , dir.z) );
	}

	// -----------------------------------------------------------------------------

	/// update the position from a direction in local space
	void Camera::update_pos(Vec3 mod_local_)
	{
		Vec3& x = get_x();
		Vec3& y = get_y();
		Vec3& dir = get_dir();
		Vec3& np = get_pos() + (x * mod_local_.x + y * mod_local_.y + dir * mod_local_.z);
		m_camera->setPosition( qglviewer::Vec( np.x ,np.y ,np.z) ) ;
	}

	// -----------------------------------------------------------------------------

	void Camera::set_pos(const Vec3& org){
		//_pos = org;
		m_camera->setPosition( qglviewer::Vec( org.x , org.y , org.z));
	}

	// -----------------------------------------------------------------------------

	/// dir and up must be orthogonal to each other
	void Camera::set_dir_and_up(const Vec3& dir, const Vec3& up)
	{
		if(isdebug)
			cout<<"Camera:set_dir_and_up"<<endl;
		// check othogonal
		assert( std::abs( dir.dot(up) ) < 0.001f );
		Vec3 x   = up.cross(dir);
		qglviewer::Quaternion q;
		q.setFromRotatedBasis( qglviewer::Vec(x.x ,x.y,x.z) ,qglviewer::Vec(up.x ,up.y,up.z),qglviewer::Vec(dir.x ,dir.y,dir.z));
		m_camera->setOrientation( q);
	}

	// -----------------------------------------------------------------------------

	void Camera::lookat(const Vec3& aimed)
	{
		if(isdebug)
			cout<<"Camera:lookat"<<endl;
		Vec3 dir = aimed - get_pos();
//		set_dir( dir );
		m_camera->setViewDirection( qglviewer::Vec( dir.x ,dir. y , dir.z) );
	}

	// -----------------------------------------------------------------------------

	void Camera::lookat() const
	{
		if(isdebug)
			cout<<"Camera:lookat()"<<endl;
		Vec3 o = get_pos() + get_dir()*2.f;
		//gluLookAt(get_pos().x, get_pos().y, get_pos().z,
		//			 o.x,    o.y,    o.z,
		//			get_y().x,   get_y().y,   get_y().z);
		qglviewer::Vec o2( o.x ,o. y , o.z);
		m_camera->lookAt( o2);
	}

	// -----------------------------------------------------------------------------

	void Camera::gl_ortho_mult() const
	{
		if(isdebug)
			cout<<"Camera:gl_ortho_mult()"<<endl;
		float dx = OTHO_ZOOM * 0.5f;
		float dy = (float)m_camera->screenHeight() * dx / (float)m_camera->screenWidth();
		glOrtho(-dx, dx, -dy, dy, m_camera->zNear(), m_camera->zFar());
	}

	// -----------------------------------------------------------------------------

	void Camera::gl_perspective_mult() const
	{
		if(isdebug)
			cout<<"Camera:gl_perspective_mult()"<<endl;
		//gluPerspective(_fov*180.0f / M_PI,
		//			   (float)_width/(float)_height,
		//			   _near,
		//			   _far);
		gluPerspective( m_camera->fieldOfView()*180.0f / M_PI,
			m_camera->screenWidth()/m_camera->screenHeight(),
			m_camera->zNear(),
			m_camera->zFar());
	}

	// -----------------------------------------------------------------------------

	void Camera::gl_mult_projection() const
	{
		if(isdebug)
			cout<<"Camera:gl_mult_projection()"<<endl;
		if(_ortho_proj) gl_ortho_mult();
		else            gl_perspective_mult();
	}

	// -----------------------------------------------------------------------------

	void Camera::roll(float theta)
	{
		if(isdebug)
			cout<<"Camera:roll"<<endl;
		float ct = cosf(theta);
		float st = sinf(theta);
		Vec3 x = (get_x() * ct) + (get_y() * st);
		Vec3 y = (get_y() * ct) - (get_x() * st);
		Vec3 dir = x.cross(y);
		qglviewer::Quaternion q;
		q.setFromRotatedBasis( qglviewer::Vec(x.x ,x.y,x.z) ,qglviewer::Vec(y.x ,y.y,y.z),qglviewer::Vec(dir.x ,dir.y,dir.z));
		m_camera->setOrientation( q);
	}

	// -----------------------------------------------------------------------------

	void Camera::local_strafe(float x, float y, float z)
	{
		Vec3 v(x, y, z);
		update_pos(v);
	}

	// -----------------------------------------------------------------------------

	Vec3 Camera::get_pos() const { 
		return Vec3( m_camera->position().x,m_camera->position().y,m_camera->position().z); }

	// -----------------------------------------------------------------------------

	Vec3 Camera::get_dir() const 
	{ 
		if(isdebug)
			cout<<"Camera:get_dir()"<<endl;
		return Vec3( m_camera->viewDirection().x,  m_camera->viewDirection().y,  m_camera->viewDirection().z); 
	}

	// -----------------------------------------------------------------------------

	Vec3 Camera::get_y() const
	{
		if(isdebug)
			cout<<"Camera:get_y()"<<endl;
		double rm[3][3];
		m_camera->orientation().getRotationMatrix(rm);
		qglviewer::Vec up = m_camera->upVector();
		if(isdebug)cout<<"up "<< up.x<<up.y << up.z<<endl;
		if(isdebug)cout<<"rm "<< rm[0][1]<<rm[1][1] << rm[2][1]<<endl;
		Tbx::Vec3 y(rm[0][1] ,rm[1][1], rm[2][1]);
		return y; }

	void Camera::set_viewport(int x, int y, int w, int h)
	{
		if(isdebug)
			cout<<"Camera:set_viewport"<<endl;
		////_width  = w; _height = h; _offy   = y; _offx   = x;
		//m_camera->getViewport();
		m_camera->setScreenWidthAndHeight(w,h);
	}

	int Camera::width() const
	{
		return m_camera->screenWidth();
	}

	int Camera::height() const
	{
		return m_camera->screenHeight();
	}

	float Camera::get_near() const
	{
		return m_camera->zNear();
	}

	float Camera::get_far() const
	{
		return m_camera->fieldOfView();
	}

	float Camera::fovy() const
	{
		return m_camera->fieldOfView();
	}

	float Camera::fovx() const
	{
		return fovy() * ((float)m_camera->screenWidth() / (float)m_camera->screenHeight());
	}

	bool Camera::is_ortho() const
	{
		return _ortho_proj;
	}

	void Camera::set_ortho(bool s)
	{
		//_ortho_proj = s;
		if(isdebug)
			cout<<"Camera:set ortho"<<endl;
	}

	void Camera::set_near(float n)
	{
		//_near = n;
		if(isdebug)
			cout<<"Camera:set znear"<<endl;

	}

	void Camera::set_far(float f)
	{
		if(isdebug)
			cout<<"Camera:set zfar"<<endl;
	}


	// -----------------------------------------------------------------------------

	Vec3 Camera::get_x() const 
	{
		if(isdebug)
			cout<<"Camera:get_x()"<<endl;
		double rm[3][3];
		m_camera->orientation().getRotationMatrix(rm);
		qglviewer::Vec right = m_camera->rightVector();
		if(isdebug)cout<<"up "<< right.x<<right.y << right.z<<endl;
		if(isdebug)cout<<"rm "<< rm[0][0]<<rm[1][0] << rm[2][0]<<endl;
		Tbx::Vec3 x(rm[0][0] ,rm[1][0], rm[2][0]);
		return x; 
	}

	// -----------------------------------------------------------------------------

	Transfo Camera::get_frame() const{

		if(isdebug)
			cout<<"Camera:get_frame()"<<endl;
		return Transfo(Mat3(get_x(), get_y(), get_dir()), get_pos());
	}

	// -----------------------------------------------------------------------------

	void Camera::transform(const Transfo& gtransfo)
	{
		if(isdebug)
			cout<<"Camera:transform"<<endl;
		Vec3 x   = gtransfo * get_x();
		Vec3 y   = gtransfo * get_y();
		Vec3 dir = gtransfo * get_dir();

		qglviewer::Quaternion q;
		q.setFromRotatedBasis( qglviewer::Vec(x.x ,x.y,x.z) ,qglviewer::Vec(y.x ,y.y,y.z),qglviewer::Vec(dir.x ,dir.y,dir.z));
		m_camera->setOrientation( q);
		Vec3 pos = (Vec3)(gtransfo * get_pos().to_point3());
		m_camera->setPosition( qglviewer::Vec( pos.x,pos.y,pos.z ));
	
	}

	// -----------------------------------------------------------------------------

	float Camera::get_ortho_zoom() const { return OTHO_ZOOM; }

	// -----------------------------------------------------------------------------

	void  Camera::set_ortho_zoom(float width)
	{
		if(isdebug)
			cout<<"Camera:set_ortho_zoom"<<endl;
		assert(width > 0.0001f);
		if(isdebug)
			cout<<"set_ortho_zoom"<<endl;
		//_ortho_zoom = width;
	}

	// -----------------------------------------------------------------------------

	inline static float cot(float x){ return tan(M_PI_2 - x); }

	Transfo Camera::get_proj_transfo() const
	{
		if(isdebug)
			cout<<"Camera:get_proj_transfo()"<<endl;
		if(!_ortho_proj)
		{
			// Compute projection matrix as describe in the doc of gluPerspective()
			const float f     = cot( m_camera->fieldOfView() * 0.5f);
			const float ratio = (float)m_camera->screenWidth() / (float)m_camera->screenHeight();
			const float diff  = m_camera->zNear() -  m_camera->zFar();

			return Transfo( f / ratio, 0.f,               0.f, 0.f,
							0.f      , f  ,               0.f, 0.f,
							0.f      , 0.f, (m_camera->zFar()+m_camera->zNear())/diff, (2.f*m_camera->zFar()*m_camera->zNear())/diff,
							0.f      , 0.f,              -1.f, 0.f
							);
		}
		else
		{
			const float dx = OTHO_ZOOM * 0.5f;
			const float dy = (float)m_camera->screenHeight() * dx / (float)m_camera->screenWidth();
			// ------------
			// Compute projection matrix as describe in the doc of gluPerspective()
			const float l = -dx; // left
			const float r =  dx; // right
			const float t =  dy; // top
			const float b = -dy; // bottom

			Vec3 tr = Vec3( -(r+l) / (r-l), -(t+b) / (t-b), - (m_camera->zFar()+m_camera->zNear()) / (m_camera->zFar()-m_camera->zNear())  );

			return Transfo( 2.f / (r-l), 0.f          ,  0.f               , tr.x,
							0.f        , 2.f / (t-b)  ,  0.f               , tr.y,
							0.f        , 0.f          , -2.f / (m_camera->zFar()-m_camera->zNear()), tr.z,
							0.f        , 0.f          ,  0.f               , 1.f
							);
		}
	}

	// -----------------------------------------------------------------------------

	Transfo Camera::get_eye_transfo() const
	{
		if(isdebug)
			cout<<"Camera:get_eye_transfo() "<<endl;
		/*
		We use the glulookat implementation :

		Let :
						centerX - eyeX
					F = centerY - eyeY
						centerZ - eyeZ

				   Let UP be the vector (upX, upY, upZ).

				   Then normalize as follows: f = F/ || F ||

				   UP' = UP/|| UP ||

				   Finally, let s = f X UP', and u = s X f.

				   M is then constructed as follows:

						 s[0]    s[1]    s[2]    0
						 u[0]    u[1]    u[2]    0
					M = -f[0]   -f[1]   -f[2]    0
						  0       0       0      1

				   and gluLookAt is equivalent to

				   glMultMatrixf(M);
				   glTranslated (-eyex, -eyey, -eyez);
		*/

		// Implementation :
		const Vec3 eye = get_pos();
		const Vec3 f   = get_dir().normalized();
		const Vec3 up  = get_y().normalized();

		const Vec3 s = f.cross( up );
		const Vec3 u = s.cross( f  );

		const Transfo trans = Transfo::translate( -eye );

		return  Transfo(  s.x,  s.y,  s.z, 0.f,
						  u.x,  u.y,  u.z, 0.f,
						 -f.x, -f.y, -f.z, 0.f,
						  0.f,  0.f,  0.f, 1.f) * trans;
	}

	// -----------------------------------------------------------------------------

	Transfo Camera::get_viewport_transfo() const
	{
		if(isdebug)
			cout<<"Camera:get_viewport_transfo() "<<endl;
		Transfo tr  = Transfo::translate( 1.f, 1.f, 1.f );
		Transfo sc  = Transfo::scale((float)m_camera->screenWidth() * 0.5f, (float)m_camera->screenHeight() * 0.5f, 0.5f);
		Transfo off = Transfo::translate((float)_offx, (float)_offy, 0.f );

		return off * sc * tr;
	}

	// -----------------------------------------------------------------------------

	Point3 Camera::project(const Point3& p) const
	{
		if(isdebug)
			cout<<"Camera:project "<<endl;
		qglviewer::Vec wp( p.x ,p.y ,p.z);
		qglviewer::Vec& img_p = m_camera->projectedCoordinatesOf( wp);
		return Point3(img_p.x ,img_p.y ,img_p.z);
		//return get_viewport_transfo() * get_proj_transfo().project( get_eye_transfo() * p);
	}

	// -----------------------------------------------------------------------------

	Point3 Camera::un_project(const Point3& p) const
	{
		if(isdebug)
			cout<<"Camera:un_project "<<endl;
		//Transfo t = (get_viewport_transfo() * get_proj_transfo() * get_eye_transfo()).full_invert();
		//return  t.project( p ); // Multiply and do perspective division
		qglviewer::Vec img_p( p.x ,p.y ,p.z);
		qglviewer::Vec& wp = m_camera->unprojectedCoordinatesOf( img_p);
		return Point3(wp.x ,wp.y ,wp.z);

	}

	// -----------------------------------------------------------------------------

	Ray_cu Camera::cast_ray(int px, int py) const
	{
		if(isdebug)
			cout<<"Camera:cast ray "<<endl;
		Ray_cu ray;
		Vec3 cam_dir = get_dir();
		Vec3 cam_pos = get_pos();
		Vec3 cam_hor = -get_x();
		Vec3 cam_ver = -get_y();

		if(_ortho_proj)
		{
			float zoom = OTHO_ZOOM;
			ray.set_dir(cam_dir);
			float dx = (px * 1.f/m_camera->screenWidth() - 0.5f) * zoom;
			float dy = (py * 1.f/m_camera->screenHeight() - 0.5f) * zoom * (m_camera->screenHeight() *  1.f/m_camera->screenWidth());
			ray.set_pos( Point3(cam_pos + cam_hor * dx + cam_ver * dy) );
		}
		else
		{
			Vec3 dep =  cam_dir * (0.5f/tanf(0.5f*m_camera->fieldOfView() ));
			ray.set_pos( Point3( cam_pos ) );
			Vec3 ver = cam_ver * (1.f / m_camera->screenHeight());
			Vec3 hor = cam_hor * (1.f / m_camera->screenHeight());
			Vec3 dir = (dep + hor * (px - m_camera->screenWidth()/2.f) + ver * (py - m_camera->screenHeight()/2.f));
			ray.set_dir(dir.normalized());
		}
		return ray;
	}

	// -----------------------------------------------------------------------------

	void Camera::print() const
	{
		if(isdebug)
		{
			std::cout << "position"  << get_pos().x << "," << get_pos().y << "," << get_pos().z << "\n";
			std::cout << "direction" << get_dir().x << "," << get_dir().y << "," << get_dir().z << "\n";
			std::cout << "up vector" << get_y().x   << "," << get_y().y   << "," << get_y().z   << "\n";
			std::cout << "x axis"    << get_x().x   << "," << get_x().y   << "," << get_x().z   << std::endl;
		}
	}


}
// -----------------------------------------------------------------------------
