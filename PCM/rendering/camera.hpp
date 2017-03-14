#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "toolbox/maths/ray_cu.hpp"
#include "toolbox/maths/vec3.hpp"
#include "toolbox/maths/transfo.hpp"
namespace qglviewer
{
	class Camera;
}
/** @class Camera
  @brief Handles camera movements, and various related parameters
*/
namespace Tbx
{
	class Camera {
	public:

		Camera();
		explicit Camera( qglviewer::Camera* _camera);
		explicit Camera( const qglviewer::Camera* _camera);
		Tbx::Camera& operator = ( const Tbx::Camera& _camera);
		// =========================================================================
		/// @name Moving the camera
		// =========================================================================

		/// update the direction from a rotation
		/// @param yaw_    rotation around y
		/// @param pitch_  rotation around x
		void update_dir(float yaw, float pitch, float roll);

		void set_dir(const Tbx::Vec3& dir);

		/// update the position from a direction in local space
		void update_pos(Tbx::Vec3 mod_local_);

		/// Sets origin of the camera
		void set_pos(const Tbx::Vec3& org);

		/// dir and up must be orthogonal to each other
		void set_dir_and_up(const Tbx::Vec3& dir, const Tbx::Vec3& up);

		/// This sets the camera frame as to look at the point 'aimed'
		void lookat(const Tbx::Vec3& aimed);

		void roll(float theta);

		void local_strafe(float x, float y, float z);

		Tbx::Vec3 get_pos() const;

		Tbx::Vec3 get_dir() const;

		Tbx::Vec3 get_y() const;

		Tbx::Vec3 get_x() const;

		Tbx::Transfo get_frame() const;

		/// Apply a global transformation to the camera frame
		void transform(const Tbx::Transfo& gtransfo);

		// =========================================================================
		/// @name Common camera operations
		// =========================================================================

		/// This call gluLookAt
		void lookat() const;

		/// Multiply the current OpenGL matrix with the ortho projection matrix
		/// @see gl_perspective_mult() gl_mult_projection()
		void gl_ortho_mult() const;

		/// Multiply the current OpenGL matrix with the perspective projection matrix
		/// @see gl_ortho_mult() gl_mult_projection()
		void gl_perspective_mult() const;

		/// Multiply the current OpenGL matrix with the projection matrix of the
		/// camera (either perspective or ortho acording to is_ortho() state)
		void gl_mult_projection() const;

		/// Compute the projection matrix. (as in OpenGL from eye coordinates
		/// to normalized device coordinates)
		Tbx::Transfo get_proj_transfo() const;

		/// Get the view matrix (As computed with gluLookAt from model coordinates
		/// to eye coordinates
		Tbx::Transfo get_eye_transfo() const;

		/// Get the viewport matrix (as in OpenGL from normalized device coordinates
		/// to window coordinates)
		/// @note z is mapped between [0 1] as in the default value of openGL
		/// for glDepthRange()
		Tbx::Transfo get_viewport_transfo() const;

		/// Project 'p' in world coordinates to the camera image plane
		/// (in window coordinates)
		/// @return p' = viewport_transfo * proj_transfo * eye_transfo * p
		Tbx::Point3 project(const Tbx::Point3& p) const;

		/// unProject 'p' in window coordinates to world coordinates
		/// @return p' = (viewport_transfo * proj_transfo * eye_transfo)^-1 * p
		Tbx::Point3 un_project(const Tbx::Point3& p) const;

		// TODO: (py) seems to be inverted (its in qt coords system 0,0 at upper left)
		// -> would be more consistent to be in opengl coords 0,0 at bottom left
		/// Cast a ray from camera : px, py are the pixel position you want
		/// to cast a ray from
		Tbx::Ray_cu cast_ray(int px, int py) const;

		// =========================================================================
		/// @name Frustum characteristics
		// =========================================================================

		/// Set the camera aperture in degrees
		void set_fov_deg(float f);

		/// Set the camera aperture in radians
		void set_fov_rad(float f);


		int width()  const;
		int height() const;

		float get_near() const;
		float get_far()  const;

		/// Aperture along y axis;
		float fovy() const;

		/// Aperture along x axis;
		float fovx() const;

		bool is_ortho() const;

		void set_ortho(bool s);

		void set_near(float n);

		void set_far(float f);

		void set_viewport(int x, int y, int w, int h);

		/// Frustum width in orthogonal projection @see is_ortho()
		/// @{
		float get_ortho_zoom() const;
		void  set_ortho_zoom(float width);
		/// @}

		void print() const;

		// -------------------------------------------------------------------------
		/// @name Attributes
		// -------------------------------------------------------------------------
	private:
		//Tbx::Vec3 _pos;    ///< position
		//Tbx::Vec3 _dir;    ///< sight direction (z axis)
		//Tbx::Vec3 _x, _y;  ///< other vectors to get the frame

		//float _fov;           ///< openGl opening angle along Y axis (in radian)
		//float _near;          ///< near plane distance
		//float _far;           ///< far plane distance
		//float _ortho_zoom;    ///< frustum width zoom in orthogonal projection

		//bool _ortho_proj;     ///< does orthogonal projection enabled ?

		//int _width;  ///< pixel width of the camera viewport
		//int _height; ///< pixel height of the camera viewport

		//int _offx;   ///< x offset of the camera viewport
		//int _offy;   ///< y offset of the camera viewport
		qglviewer::Camera*  m_camera;
	};
}


#endif
